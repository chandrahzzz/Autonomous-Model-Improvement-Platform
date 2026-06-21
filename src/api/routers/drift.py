"""Predictive drift early-warning endpoints (RFC-001). Read-only."""

import json
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session, get_redis
from src.config.settings import settings
from src.db.repositories.drift_trend import DriftTrendRepository

router = APIRouter()


class DriftTrendRow(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    cycle_id: str | None
    model_version: str | None
    current_score: float
    threshold: float
    slope_per_cycle: float
    r_squared: float
    predicted_trigger_hours: float | None
    window_size: int
    trend_direction: str
    is_alarming: bool
    alert_sent: bool
    created_at: datetime


@router.get("/trend")
async def get_drift_trend() -> dict:
    """Latest predicted drift trend (from Redis). `no_data` if not yet computed."""
    raw = await get_redis().get(settings.drift_trend_redis_key)
    if not raw:
        return {
            "status": "no_data",
            "reason": "window_too_small_or_prediction_disabled",
        }
    return json.loads(raw)


@router.get("/trend/history", response_model=list[DriftTrendRow])
async def get_drift_trend_history(
    hours: int = 24,
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session),
) -> list[DriftTrendRow]:
    rows = await DriftTrendRepository(db).get_recent(hours=hours, limit=limit)
    return [DriftTrendRow.model_validate(r) for r in rows]


@router.get("/trend/alarms", response_model=list[DriftTrendRow])
async def get_drift_trend_alarms(
    limit: int = 50,
    db: AsyncSession = Depends(get_db_session),
) -> list[DriftTrendRow]:
    rows = await DriftTrendRepository(db).get_alarming(limit=limit)
    return [DriftTrendRow.model_validate(r) for r in rows]
