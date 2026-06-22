from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
import structlog

from src.api.dependencies import get_db_session, get_redis
from src.db.connection import check_database_health

log = structlog.get_logger()
router = APIRouter()


@router.get("/health")
async def health(
    db: AsyncSession = Depends(get_db_session),
    redis: aioredis.Redis = Depends(get_redis),
) -> dict:
    db_ok = await check_database_health()
    try:
        redis_ok = await redis.ping()
    except Exception:
        redis_ok = False

    # Kafka liveness via producer (non-blocking)
    kafka_ok = True  # producer failure surfaces in metrics, not health check

    from src.audit.hmac_signer import is_degraded
    vault_degraded = is_degraded()

    # ── Detection-layer health (June 2026 hardening) ───────────────────────────
    # Surface stale baselines and ungrounded RAG calls so operators see them
    # before they cause false alarms / silent missed hallucinations.
    warnings: list[str] = []
    detection: dict = {}
    try:
        from src.graph.nodes.failure_detector import _drift, _fmt
        from src.monitoring.metrics import hallucination_premise_missing_total

        drift_age = _drift.baseline_age_hours
        fmt_age = _fmt.baseline_age_hours
        premise_missing = hallucination_premise_missing_total._value.get()
        detection = {
            "drift_baseline_age_hours": round(drift_age, 2) if drift_age is not None else None,
            "drift_baseline_loaded": _drift._loaded,
            "drift_baseline_stale": _drift.is_baseline_stale(),
            "drift_window_size": _drift.window_size,
            "format_baseline_age_hours": round(fmt_age, 2) if fmt_age is not None else None,
            "format_baseline_stale": _fmt.is_baseline_stale(),
            "hallucination_premise_missing_total": premise_missing,
        }
        if _drift.is_baseline_stale():
            warnings.append(
                f"DRIFT_BASELINE_STALE: active baseline is {drift_age:.1f}h old "
                f"(> {_drift_max_age()}h) — drift detection may be unreliable."
            )
        if _fmt.is_baseline_stale():
            warnings.append(
                f"FORMAT_BASELINE_STALE: length baseline is {fmt_age:.1f}h old — "
                "format regression detection may be unreliable."
            )
        if premise_missing > 0:
            warnings.append(
                f"HALLUCINATION_PREMISE_MISSING: {int(premise_missing)} RAG-flagged "
                "calls lacked retrieved_context; grounding fell back to the prompt."
            )
    except Exception:
        log.warning("health_detection_probe_failed")

    if vault_degraded:
        warnings.append(
            "VAULT_DEGRADED: audit-trail HMAC is using the local fallback key, "
            "not Vault — signatures are not cryptographically guaranteed."
        )

    status = "ok" if (db_ok and redis_ok) else "degraded"
    response: dict = {
        "status": status,
        "database": db_ok,
        "redis": bool(redis_ok),
        "kafka": kafka_ok,
        "vault_degraded_mode": vault_degraded,
        "detection": detection,
    }
    if warnings:
        response["warnings"] = warnings
    return response


def _drift_max_age() -> float:
    from src.config.settings import settings
    return settings.drift_baseline_max_age_hours


@router.get("/ready")
async def readiness() -> dict:
    """Kubernetes readiness probe — returns 200 when app is ready to serve."""
    return {"ready": True}


@router.get("/live")
async def liveness() -> dict:
    """Kubernetes liveness probe — returns 200 if process is alive."""
    return {"alive": True}
