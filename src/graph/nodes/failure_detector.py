"""Failure detector node: runs all 4 detectors on the current log batch."""
import asyncio
import json
from datetime import datetime, timezone

import structlog
import redis.asyncio as aioredis

from src.config.settings import settings
from src.graph.state import PipelineState
from src.db.connection import AsyncSessionLocal, get_db
from src.db.repositories.llm_logs import LLMLogRepository
from src.detection.hallucination import HallucinationDetector
from src.detection.drift import DriftDetector
from src.detection.refusal import RefusalDetector
from src.detection.format_validator import FormatValidator
from src.detection.failure_classifier import FailureClassifier
from src.detection.drift_predictor import DriftPredictor, DriftTrend, get_drift_predictor
from src.monitoring.metrics import (
    drift_trend_slope,
    drift_r_squared,
    drift_predicted_trigger_hours,
)

log = structlog.get_logger()

_hall = HallucinationDetector()
_drift = DriftDetector()
_refusal = RefusalDetector()
_fmt = FormatValidator()
_classifier = FailureClassifier(_hall, _drift, _refusal, _fmt)


async def _publish_trend_to_redis(trend: DriftTrend) -> None:
    """Write current trend snapshot to Redis for dashboard polling."""
    payload = {
        "current_score": trend.current_score,
        "threshold": trend.threshold,
        "slope_per_cycle": trend.slope_per_cycle,
        "r_squared": trend.r_squared,
        "predicted_trigger_hours": trend.predicted_trigger_hours,
        "window_size": trend.window_size,
        "trend_direction": trend.trend_direction,
        "is_alarming": trend.is_alarming,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    r = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        await r.set(
            settings.drift_trend_redis_key,
            json.dumps(payload),
            ex=settings.drift_trend_redis_ttl_seconds,
        )
    finally:
        await r.aclose()


async def _handle_drift_prediction(
    trend: DriftTrend,
    cycle_id: str | None,
    model_version: str | None,
) -> None:
    """Fire-and-forget: alert + persist drift trend. Not on the critical path."""
    try:
        from src.monitoring.alerts import alerter
        from src.db.repositories.drift_trend import DriftTrendRepository

        alert_sent = await get_drift_predictor(None).maybe_send_alert(trend, alerter)

        async with get_db() as db:
            await DriftTrendRepository(db).insert({
                "cycle_id": cycle_id,
                "model_version": model_version,
                "current_score": trend.current_score,
                "threshold": trend.threshold,
                "slope_per_cycle": trend.slope_per_cycle,
                "r_squared": trend.r_squared,
                "predicted_trigger_hours": trend.predicted_trigger_hours,
                "window_size": trend.window_size,
                "trend_direction": trend.trend_direction,
                "is_alarming": trend.is_alarming,
                "alert_sent": alert_sent,
            })

        if trend.is_alarming:
            await _publish_trend_to_redis(trend)
    except Exception:
        log.exception("drift_prediction_background_error")

_UUID_LEN = 36


def _looks_like_uuid(v: str | None) -> bool:
    return bool(v) and len(v) == _UUID_LEN and v.count("-") == 4


async def _persist_classifications(events: list) -> None:
    """Write one failure_classifications row per event (reusing an existing row
    for logs already classified), and record the id on event.metadata under
    'failure_classification_id'. Best-effort: a persistence error must not stop
    detection/curation, so it's logged and swallowed."""
    if not events:
        return
    log_ids = [e.llm_log_id for e in events if _looks_like_uuid(e.llm_log_id)]
    try:
        from src.db.repositories.failure_classifications import FailureClassificationRepository
        async with get_db() as db:
            repo = FailureClassificationRepository(db)
            existing = await repo.existing_for_logs(log_ids)
            for e in events:
                if not _looks_like_uuid(e.llm_log_id):
                    continue
                if e.llm_log_id in existing:
                    e.metadata["failure_classification_id"] = existing[e.llm_log_id]
                    continue
                cid = await repo.insert({
                    "llm_log_id": e.llm_log_id,
                    "failure_type": e.failure_type,
                    "score": float(e.score),
                    "cluster_id": e.metadata.get("cluster_id"),
                    "cluster_label": e.metadata.get("cluster_label"),
                    "metadata_": {k: v for k, v in e.metadata.items()
                                  if k != "failure_classification_id"},
                })
                e.metadata["failure_classification_id"] = cid
                existing[e.llm_log_id] = cid
    except Exception:
        log.warning("failure_classification_persist_failed")


async def failure_detector_node(state: PipelineState) -> PipelineState:
    log_ids = state.get("recent_log_ids", [])
    if not log_ids:
        return {**state, "failure_events": [], "failure_count": 0, "has_failures": False, "drift_score": 0.0}

    async with AsyncSessionLocal() as db:
        repo = LLMLogRepository(db)
        # Load baseline for drift detector on first run
        await _drift.load_baseline(db)

        # Fetch full log records
        events = []
        for log_id in log_ids[:500]:
            record = await repo.get_by_id(log_id)
            if record:
                # Prefer the first-class column; fall back to metadata for
                # producers that haven't migrated yet.
                meta = getattr(record, "metadata_", None)
                meta = meta if isinstance(meta, dict) else {}
                ctx = getattr(record, "retrieved_context", None) or meta.get("retrieved_context")
                # A call is RAG if it carried context or was explicitly flagged.
                is_rag = bool(meta.get("is_rag")) or bool(ctx)
                events.append({
                    "id": str(record.id),
                    "prompt": record.prompt,
                    "completion": record.completion,
                    "retrieved_context": ctx or "",
                    "is_rag": is_rag,
                    "model_version": record.model_version,
                })

    batch = await _classifier.classify_batch(events)

    # Persist a failure_classifications row per detected failure so the record
    # exists in the DB (the known-good replay filter, attribution, lineage and
    # the calibrator all depend on it — previously it was never written, so bad
    # logs leaked into the replay buffer as "known-good"). Reuse an existing row
    # when this same log was already classified in an earlier cycle, and stash the
    # id in the event metadata so the curator can set training_examples.failure_id.
    await _persist_classifications(batch.events)

    failure_dicts = [
        {
            "llm_log_id": f.llm_log_id,
            "prompt": f.prompt,
            "completion": f.completion,
            "failure_type": f.failure_type,
            "score": f.score,
            "metadata": f.metadata,
        }
        for f in batch.events
    ]
    updates: dict = {
        "failure_events": failure_dicts,
        "drift_score": batch.drift_score,
        "failure_count": len(failure_dicts),
        "has_failures": batch.has_failures,
    }

    # ── Predictive drift early warning (RFC-001) — additive, observational ──
    # Never affects failure classification; alert/persist run off the critical
    # path via asyncio.create_task. Entire block gated by the kill switch.
    if settings.drift_prediction_enabled:
        predictor = get_drift_predictor(_drift)
        if predictor.should_compute_this_cycle():
            trend = predictor.compute_trend()
            if trend is not None:
                drift_trend_slope.set(trend.slope_per_cycle)
                drift_r_squared.set(trend.r_squared)
                drift_predicted_trigger_hours.set(
                    trend.predicted_trigger_hours
                    if trend.predicted_trigger_hours is not None
                    else -1.0
                )
                asyncio.create_task(_handle_drift_prediction(
                    trend, state.get("cycle_id"), state.get("production_version")
                ))
                updates.update({
                    "drift_slope": trend.slope_per_cycle,
                    "drift_predicted_trigger_hours": trend.predicted_trigger_hours,
                    "drift_is_alarming": trend.is_alarming,
                    "drift_trend_direction": trend.trend_direction,
                })

    # ── Failure attribution (RFC-003) — additive, fire-and-forget ──
    if settings.attribution_enabled and batch.has_failures:
        failures_to_attribute = batch.events[: settings.attribution_max_failures_per_cycle]

        async def _attribution_task(failures=failures_to_attribute):
            count = 0
            try:
                from src.attribution.attributor import get_attributor  # loads model lazily
                attributor = get_attributor()
                async with get_db() as adb:
                    for f in failures:
                        if await attributor.attribute(f, adb):
                            count += 1
            except Exception:
                log.exception("attribution_task_error")
            finally:
                log.info("attribution_cycle_complete", attributed=count)

        asyncio.create_task(_attribution_task())
        updates.update({
            "attribution_count_this_cycle": len(failures_to_attribute),
            "last_attribution_run_at": datetime.now(timezone.utc).isoformat(),
        })

    log.info("failure_detector_node_complete", failures=len(failure_dicts), drift=batch.drift_score)
    return {**state, **updates}
