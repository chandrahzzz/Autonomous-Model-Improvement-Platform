"""
Ingestion adapters: land generated Calls in `llm_logs` the way a real app would.

Three modes:
  db    — insert rows directly via LLMLogRepository (most reliable for a laptop
          demo; no Kafka consumer needed). RECOMMENDED.
  kafka — produce LLMEvent messages to the llm.production.events topic via the
          real AsyncKafkaProducer. Only reaches llm_logs if a Kafka->DB consumer
          is running; warns loudly otherwise.
  http  — POST to the FastAPI app so the real LLMInterceptorMiddleware captures
          the call. Same consumer caveat as kafka (middleware emits to Kafka).

All `src` imports are lazy so importing `sim` (e.g. in unit tests that mock the
emit layer) never requires a database, Kafka, or the ML stack.
"""

from __future__ import annotations

import asyncio
import json
import random
from datetime import datetime, timedelta

import structlog

from sim.answers import Call
from sim.config import SimConfig, TOKEN_COSTS

log = structlog.get_logger()


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    c = TOKEN_COSTS.get(model, TOKEN_COSTS["default"])
    return prompt_tokens * c["input"] + completion_tokens * c["output"]


def _latency_ms(call: Call, rng: random.Random) -> int:
    # Healthy stays well under 3000 so it satisfies the known-good replay filter.
    if call.failure_type is None:
        return rng.randint(120, 1400)
    return rng.randint(150, 2200)


def _created_at(cfg: SimConfig, i: int, n: int) -> datetime:
    now = datetime.utcnow()
    if cfg.backfill_minutes <= 0:
        return now
    # Spread rows evenly across the past `backfill_minutes` (oldest first).
    span = timedelta(minutes=cfg.backfill_minutes)
    return now - span + (span * (i / max(1, n)))


def _session_for(i: int, rng: random.Random, state: dict) -> str:
    """Group a few consecutive calls into one session, like real usage."""
    if state.get("left", 0) <= 0:
        state["id"] = f"sess-{rng.randint(10_000, 99_999)}"
        state["left"] = rng.randint(1, 5)
    state["left"] -= 1
    return state["id"]


def _metadata(cfg: SimConfig, call: Call) -> dict:
    return {
        "model": cfg.model_name,
        "is_rag": call.is_rag,          # detector reads is_rag from here
        "status_code": 200,
        "sim": True,                    # marks synthetic traffic
        "sim_failure_label": call.failure_type,
        "domain": call.domain,
        "expect_json": call.expect_json,
    }


def call_to_row(cfg: SimConfig, call: Call, rng: random.Random, i: int, n: int,
                session_state: dict) -> dict:
    """Map a Call to an LLMLog constructor dict (keys = ORM attribute names)."""
    pt = _tokens(call.prompt)
    ct = _tokens(call.completion)
    return {
        "session_id": _session_for(i, rng, session_state),
        "user_cohort": rng.choice(cfg.cohorts),
        "model_version": cfg.model_version,
        "prompt": call.prompt,
        "completion": call.completion,
        "retrieved_context": call.retrieved_context or None,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "latency_ms": _latency_ms(call, rng),
        "finish_reason": call.finish_reason,
        "cost_usd": _cost(cfg.model_name, pt, ct),
        "embedding_hash": "",
        "metadata_": _metadata(cfg, call),
        "created_at": _created_at(cfg, i, n),
    }


def call_to_event(cfg: SimConfig, call: Call, rng: random.Random, i: int, n: int,
                  session_state: dict) -> dict:
    """Map a Call to an LLMEvent payload dict for Kafka."""
    from src.kafka.schemas.llm_event import LLMEvent
    pt = _tokens(call.prompt)
    ct = _tokens(call.completion)
    event = LLMEvent(
        session_id=_session_for(i, rng, session_state),
        user_cohort=rng.choice(cfg.cohorts),
        model_version=cfg.model_version,
        prompt=call.prompt,
        completion=call.completion,
        retrieved_context=call.retrieved_context or "",
        is_rag=call.is_rag,
        prompt_tokens=pt,
        completion_tokens=ct,
        latency_ms=_latency_ms(call, rng),
        finish_reason=call.finish_reason,
        cost_usd=_cost(cfg.model_name, pt, ct),
        metadata=_metadata(cfg, call),
    )
    return event.model_dump()


# ── Mode: db ─────────────────────────────────────────────────────────────────
async def _emit_db(calls: list[Call], cfg: SimConfig) -> dict:
    from src.db.connection import get_db
    from src.db.repositories.llm_logs import LLMLogRepository

    from src.shadow.service import observe_production_call

    rng = random.Random(cfg.seed + 7)
    session_state: dict = {}
    n = len(calls)
    inserted = 0
    shadowed = 0
    for start in range(0, n, cfg.db_batch_size):
        chunk = calls[start:start + cfg.db_batch_size]
        async with get_db() as db:
            repo = LLMLogRepository(db)
            for j, call in enumerate(chunk):
                row = call_to_row(cfg, call, rng, start + j, n, session_state)
                await repo.insert(row)
            inserted += len(chunk)
        # The simulator stands in for the production app, so it also stands in
        # for its serving layer: offer each call to the shadow router so a
        # challenger under test accumulates A/B samples. No-ops (one Redis GET)
        # whenever no shadow test is running.
        for call in chunk:
            if await observe_production_call(call.prompt, call.completion) is not None:
                shadowed += 1
        log.info("sim_db_batch_committed", inserted=inserted, total=n)
    if shadowed:
        log.info("sim_shadow_samples_recorded", shadowed=shadowed)
    return {"inserted": inserted, "shadowed": shadowed}


# ── Mode: kafka ──────────────────────────────────────────────────────────────
async def _emit_kafka(calls: list[Call], cfg: SimConfig) -> dict:
    from src.kafka.producer import get_producer
    from src.config.settings import settings

    log.warning(
        "sim_kafka_mode_consumer_required",
        message=(
            "Producing to the llm.production.events topic. These reach llm_logs "
            "ONLY if a Kafka->DB consumer is running. If nothing consumes the "
            "topic, use --mode db instead."
        ),
    )
    producer = get_producer()
    rng = random.Random(cfg.seed + 7)
    session_state: dict = {}
    n = len(calls)
    produced = 0
    for i, call in enumerate(calls):
        payload = call_to_event(cfg, call, rng, i, n, session_state)
        await producer.produce(
            topic=settings.kafka_topic_llm_events,
            value=payload,
            key=payload["session_id"],
        )
        produced += 1
        if cfg.rate > 0:
            await asyncio.sleep(1.0 / cfg.rate)
    await producer.flush()
    return {"produced": produced}


# ── Mode: http ───────────────────────────────────────────────────────────────
async def _emit_http(calls: list[Call], cfg: SimConfig) -> dict:
    import httpx

    log.warning(
        "sim_http_mode_consumer_required",
        message=(
            "POSTing through the LLMInterceptorMiddleware. The middleware emits "
            "to Kafka, so rows reach llm_logs ONLY if a Kafka->DB consumer runs. "
            "For a self-contained laptop demo prefer --mode db."
        ),
    )
    rng = random.Random(cfg.seed + 7)
    session_state: dict = {}
    n = len(calls)
    url = cfg.base_url.rstrip("/") + "/" + cfg.http_path.lstrip("/")
    sent = 0
    async with httpx.AsyncClient(timeout=15.0) as client:
        for i, call in enumerate(calls):
            pt, ct = _tokens(call.prompt), _tokens(call.completion)
            meta = {
                "model": cfg.model_name,
                "content": call.completion,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "finish_reason": call.finish_reason,
                "retrieved_context": call.retrieved_context or "",
            }
            headers = {
                "X-LLM-Call": "1",
                "X-Session-ID": _session_for(i, rng, session_state),
                "X-User-Cohort": rng.choice(cfg.cohorts),
                "X-Model-Version": cfg.model_version,
                "X-Prompt-Hash": call.prompt,     # misnamed by the middleware: carries prompt TEXT
                "X-LLM-Meta": json.dumps(meta),
            }
            try:
                await client.post(url, headers=headers, json={"sim": True})
                sent += 1
            except Exception as e:
                log.warning("sim_http_post_failed", error=str(e))
            if cfg.rate > 0:
                await asyncio.sleep(1.0 / cfg.rate)
    return {"sent": sent}


_EMITTERS = {"db": _emit_db, "kafka": _emit_kafka, "http": _emit_http}


async def emit_calls(calls: list[Call], cfg: SimConfig) -> dict:
    """Dispatch to the configured ingestion mode. Returns a small stats dict."""
    emitter = _EMITTERS.get(cfg.mode)
    if emitter is None:
        raise ValueError(f"unknown mode: {cfg.mode!r} (expected db|kafka|http)")
    return await emitter(calls, cfg)
