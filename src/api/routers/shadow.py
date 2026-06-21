from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis

from src.api.dependencies import get_db_session, get_redis

router = APIRouter()

SHADOW_ACTIVE_KEY = "shadow:active_version"
SHADOW_ABORT_KEY = "shadow:abort"


@router.get("/status")
async def shadow_status(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    challenger = await redis.get(SHADOW_ACTIVE_KEY)
    aborted = await redis.exists(SHADOW_ABORT_KEY)
    return {
        "active": bool(challenger),
        "challenger_version": challenger,
        "aborted": bool(aborted),
    }


@router.post("/abort")
async def abort_shadow(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    """Abort the current shadow test and reset the shadow slot."""
    await redis.set(SHADOW_ABORT_KEY, "1", ex=3600)
    await redis.delete(SHADOW_ACTIVE_KEY)
    return {"aborted": True}


@router.get("/canary/status")
async def canary_status(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    """Current canary rollout state (active version + recorded metrics)."""
    from src.shadow.canary import CanaryController
    return await CanaryController(redis).status()


@router.post("/canary/abort")
async def abort_canary(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    """Abort the current canary rollout."""
    from src.shadow.canary import CanaryController
    await CanaryController(redis).clear()
    return {"aborted": True}
