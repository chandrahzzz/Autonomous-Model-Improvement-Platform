import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as aioredis
import structlog

from src.api.dependencies import get_db_session, get_redis

log = structlog.get_logger()
router = APIRouter()

PIPELINE_STATE_KEY = "pipeline:state"
PIPELINE_PAUSED_KEY = "pipeline:paused"


@router.get("/status")
async def pipeline_status(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    raw = await redis.get(PIPELINE_STATE_KEY)
    paused = await redis.get(PIPELINE_PAUSED_KEY)
    state = json.loads(raw) if raw else {}
    return {
        "paused": bool(paused),
        "state": state,
    }


@router.post("/trigger")
async def manual_trigger(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    """Manually trigger a training cycle check (sets a Redis flag)."""
    await redis.set("pipeline:manual_trigger", "1", ex=3600)
    log.info("pipeline_manual_trigger_set")
    return {"triggered": True}


@router.post("/pause")
async def pause_pipeline(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    await redis.set(PIPELINE_PAUSED_KEY, "1")
    log.info("pipeline_paused")
    return {"paused": True}


@router.post("/resume")
async def resume_pipeline(redis: aioredis.Redis = Depends(get_redis)) -> dict:
    await redis.delete(PIPELINE_PAUSED_KEY)
    log.info("pipeline_resumed")
    return {"paused": False}
