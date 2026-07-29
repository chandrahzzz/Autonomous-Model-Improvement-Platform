"""
Log monitor node: consumes LLM event batch from the DB.
Pulls the last N unprocessed events for the current cycle.
"""
import asyncio
import uuid
from datetime import datetime
import structlog
import redis.asyncio as aioredis

from src.config.settings import settings
from src.graph.state import PipelineState
from src.db.connection import AsyncSessionLocal, get_db
from src.db.repositories.llm_logs import LLMLogRepository

log = structlog.get_logger()
BATCH_SIZE = 500

# Lazily-built shared clients for the eval factory (RFC-002).
_openai_client = None
_factory_redis = None


def _get_openai():
    """Chat-completions client for the eval factory. $0 stack: AsyncGroq (free
    tier), which is interface-compatible with AsyncOpenAI. Falls back to OpenAI
    only when a Groq key is absent but an OpenAI key is set."""
    global _openai_client
    if _openai_client is None:
        if settings.groq_api_key:
            from groq import AsyncGroq
            _openai_client = AsyncGroq(api_key=settings.groq_api_key)
        else:
            from openai import AsyncOpenAI
            _openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _openai_client


def _get_factory_redis():
    global _factory_redis
    if _factory_redis is None:
        _factory_redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _factory_redis


async def _run_eval_factory_task(log_ids: list) -> None:
    """Fire-and-forget: fetch prompts, run the eval factory, persist results."""
    try:
        async with get_db() as db:
            logs = await LLMLogRepository(db).get_by_ids(log_ids)
            recent_prompts = [
                lg.prompt for lg in logs if lg.prompt and len(lg.prompt.strip()) > 10
            ]
        if len(recent_prompts) < 10:
            return  # not enough data for clustering
        from src.evaluation.eval_factory import EvalFactory
        from src.db.repositories.eval_set import EvalSetRepository
        async with get_db() as factory_db:
            factory = EvalFactory(
                eval_repo=EvalSetRepository(factory_db),
                openai_client=_get_openai(),
                redis_client=_get_factory_redis(),
            )
            added = await factory.run(recent_prompts=recent_prompts, db=factory_db)
            log.info("eval_factory_run_complete", examples_added=added)
    except Exception:
        log.exception("eval_factory_task_error")


async def _filter_unprocessed(log_ids: list[str]) -> list[str]:
    """Drop ids already handed to the detectors in an earlier cycle.

    The query window is a rolling hour but the cycle is a minute, so without
    this the same rows are re-classified every cycle — burning NLI/embedding
    compute and feeding the drift detector duplicate samples. Fails OPEN (returns
    everything) on any Redis problem: re-processing is wasteful, dropping logs is
    a correctness bug.
    """
    if not settings.log_dedup_enabled or not log_ids:
        return log_ids
    try:
        r = _get_factory_redis()
        key = settings.log_dedup_key
        seen_flags = await r.smismember(key, log_ids)
        fresh = [lid for lid, seen in zip(log_ids, seen_flags) if not seen]
        if fresh:
            await r.sadd(key, *fresh)
            await r.expire(key, settings.log_dedup_ttl_seconds)
        return fresh
    except Exception:
        log.warning("log_dedup_failed_open", n_ids=len(log_ids))
        return log_ids


async def log_monitor_node(state: PipelineState) -> PipelineState:
    cycle_id = str(uuid.uuid4())
    log.info("log_monitor_node_start", cycle_id=cycle_id)
    async with AsyncSessionLocal() as db:
        repo = LLMLogRepository(db)
        recent = await repo.get_recent(limit=BATCH_SIZE, hours=1)
    fetched_ids = [str(r.id) for r in recent]
    recent_log_ids = await _filter_unprocessed(fetched_ids)
    if len(recent_log_ids) != len(fetched_ids):
        log.info(
            "log_monitor_skipped_already_processed",
            fetched=len(fetched_ids),
            fresh=len(recent_log_ids),
        )
    updates = {
        "cycle_id": cycle_id,
        "cycle_start_at": datetime.utcnow().isoformat(),
        "recent_log_ids": recent_log_ids,
        "log_batch_size": len(recent_log_ids),
        "error": None,
    }

    # ── Eval factory trigger (RFC-002) — additive, off via the flag ──
    if settings.eval_factory_enabled and recent_log_ids:
        try:
            r = _get_factory_redis()
            await r.incrby(settings.eval_factory_request_counter_key, len(recent_log_ids))
            counter = await r.get(settings.eval_factory_request_counter_key)
            if counter is not None and int(counter) >= settings.eval_factory_trigger_every_n_requests:
                asyncio.create_task(_run_eval_factory_task(recent_log_ids))
        except Exception:
            log.warning("eval_factory_trigger_failed")

    log.info("log_monitor_node_complete", n_logs=len(recent_log_ids))
    return {**state, **updates}
