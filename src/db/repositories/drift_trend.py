from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import DriftTrendHistory


class DriftTrendRepository:
    """Thin read/write layer for drift_trend_history. No business logic."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def insert(self, data: dict[str, Any]) -> DriftTrendHistory:
        row = DriftTrendHistory(**data)
        self._db.add(row)
        await self._db.flush()
        return row

    async def get_recent(self, hours: int = 24, limit: int = 100) -> list[DriftTrendHistory]:
        since = datetime.utcnow() - timedelta(hours=hours)
        result = await self._db.execute(
            select(DriftTrendHistory)
            .where(DriftTrendHistory.created_at >= since)
            .order_by(DriftTrendHistory.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_alarming(self, limit: int = 50) -> list[DriftTrendHistory]:
        result = await self._db.execute(
            select(DriftTrendHistory)
            .where(DriftTrendHistory.is_alarming.is_(True))
            .order_by(DriftTrendHistory.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def delete_older_than(self, days: int = 30) -> int:
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self._db.execute(
            delete(DriftTrendHistory).where(DriftTrendHistory.created_at < cutoff)
        )
        return result.rowcount or 0
