"""
Live shadow-observation hook — the call path that actually feeds `shadow_logs`.

`ShadowRouter` implements shadow sampling, scoring and persistence, but nothing
ever constructed one, so `shadow_logs` stayed empty, `ABCollector.collect_window`
always returned `n_requests=0`, `ready` was never True, and the graph looped
`ab_test_node -> END` forever: the pipeline could neither promote nor roll back.

This module is the missing glue. `observe_production_call()` is called from the
ingestion path (the traffic simulator's DB emitter and the FastAPI interceptor
middleware) once per production LLM call. It is deliberately cheap and total:
when no challenger is in shadow it returns after a single Redis GET, and it never
raises into the caller's request path.

The challenger invoke fn is resolved from the challenger version's stored adapter
path. Loading a real challenger needs a GPU and the gated base weights, so when
`eval_real_inference` is off the challenger echoes production output (delta ~ 0):
that keeps the loop moving end-to-end but produces no meaningful promotion
signal, and is logged as such.
"""

from __future__ import annotations

import structlog
import redis.asyncio as aioredis

from src.config.settings import settings

log = structlog.get_logger()

_redis: aioredis.Redis | None = None
_router = None
# Challenger invoke fns are cached per version tag: building one loads (and
# merges) a multi-GB model, which must not happen per request.
_invoke_cache: dict[str, object] = {}
_stub_warned = False


def _get_router():
    global _redis, _router
    if _router is None:
        from src.shadow.router import ShadowRouter

        _redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        _router = ShadowRouter(_redis)
    return _router


async def _stub_challenger(prompt: str, production_output: str) -> str:
    """Dev stand-in: echo production so the delta is 0 rather than fabricated."""
    return production_output


async def _resolve_invoke_fn(version_tag: str, production_output: str):
    """Invoke fn for the challenger version, or None when one can't be built."""
    global _stub_warned

    cached = _invoke_cache.get(version_tag)
    if cached is not None:
        return cached

    if not settings.eval_real_inference:
        if not settings.shadow_allow_stub_challenger:
            return None
        if not _stub_warned:
            log.warning(
                "shadow_using_stub_challenger",
                note=(
                    "EVAL_REAL_INFERENCE is off, so the shadow 'challenger' echoes "
                    "production and every quality delta is 0. The A/B window will "
                    "fill and the loop will complete, but the promotion decision "
                    "carries no signal. Enable real inference on a GPU box for a "
                    "meaningful result."
                ),
            )
            _stub_warned = True

        async def _fn(prompt: str, _out: str = production_output) -> str:
            return await _stub_challenger(prompt, _out)

        return _fn

    try:
        from src.db.connection import get_db
        from src.db.repositories.model_versions import ModelRepository
        from src.inference.challenger import build_challenger_invoke_fn

        async with get_db() as db:
            version = await ModelRepository(db).get_by_tag(version_tag)
        adapter = getattr(version, "lora_weights_path", None) if version else None
        if not adapter:
            log.warning("shadow_challenger_has_no_adapter", version_tag=version_tag)
            return None
        base_model = settings.base_model_name
        fn = build_challenger_invoke_fn(base_model, adapter)
        _invoke_cache[version_tag] = fn
        return fn
    except Exception:
        log.exception("shadow_challenger_build_failed", version_tag=version_tag)
        return None


async def observe_production_call(prompt: str, production_output: str) -> float | None:
    """Offer one production call to the shadow router.

    Returns the quality delta when the call was shadowed and scored, else None.
    Never raises — a shadow failure must not affect serving or ingestion.
    """
    if not settings.shadow_observation_enabled:
        return None
    if not prompt or not production_output:
        return None

    try:
        router = _get_router()
        version_tag = await router.get_challenger_version()
        if not version_tag:
            return None  # no shadow test running — the common case, one Redis GET

        invoke_fn = await _resolve_invoke_fn(version_tag, production_output)
        if invoke_fn is None:
            return None

        return await router.maybe_shadow(prompt, production_output, invoke_fn)
    except Exception:
        log.warning("shadow_observation_failed")
        return None


def reset_for_tests() -> None:
    """Drop cached clients/fns so tests don't leak state between cases."""
    global _redis, _router, _stub_warned
    _redis = None
    _router = None
    _stub_warned = False
    _invoke_cache.clear()
