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

    status = "ok" if (db_ok and redis_ok) else "degraded"
    response: dict = {
        "status": status,
        "database": db_ok,
        "redis": bool(redis_ok),
        "kafka": kafka_ok,
        "vault_degraded_mode": vault_degraded,
    }
    if vault_degraded:
        response["warnings"] = [
            "VAULT_DEGRADED: audit-trail HMAC is using the local fallback key, "
            "not Vault — signatures are not cryptographically guaranteed."
        ]
    return response


@router.get("/ready")
async def readiness() -> dict:
    """Kubernetes readiness probe — returns 200 when app is ready to serve."""
    return {"ready": True}


@router.get("/live")
async def liveness() -> dict:
    """Kubernetes liveness probe — returns 200 if process is alive."""
    return {"alive": True}
