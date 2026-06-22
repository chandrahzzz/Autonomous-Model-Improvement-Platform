from datetime import datetime, timedelta
from typing import Any
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import LLMLog


class LLMLogRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def insert(self, data: dict[str, Any]) -> LLMLog:
        log = LLMLog(**data)
        self._db.add(log)
        await self._db.flush()
        return log

    async def get_recent(self, limit: int = 1000, hours: int = 1) -> list[LLMLog]:
        since = datetime.utcnow() - timedelta(hours=hours)
        result = await self._db.execute(
            select(LLMLog)
            .where(LLMLog.created_at >= since)
            .order_by(LLMLog.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_since(self, hours: int = 24) -> int:
        since = datetime.utcnow() - timedelta(hours=hours)
        result = await self._db.execute(
            select(func.count()).where(LLMLog.created_at >= since)
        )
        return result.scalar_one()

    async def get_by_id(self, log_id: str) -> LLMLog | None:
        result = await self._db.execute(select(LLMLog).where(LLMLog.id == log_id))
        return result.scalar_one_or_none()

    async def get_by_ids(self, ids: list) -> list[LLMLog]:
        if not ids:
            return []
        result = await self._db.execute(select(LLMLog).where(LLMLog.id.in_(ids)))
        return list(result.scalars().all())

    async def get_recent_completions(
        self, limit: int, model_version: str | None = None
    ) -> list[str]:
        """Recent completion texts, optionally for a specific model version.
        Used to recompute the drift baseline after a promotion."""
        stmt = select(LLMLog.completion).order_by(LLMLog.created_at.desc()).limit(limit)
        if model_version:
            stmt = stmt.where(LLMLog.model_version == model_version)
        result = await self._db.execute(stmt)
        return [row[0] for row in result.fetchall()]

    async def get_known_good_sample(
        self, limit: int, model_version: str | None = None
    ) -> list[LLMLog]:
        """Recent logs that were NOT flagged as failures — the replay buffer used
        to counter catastrophic forgetting. Filters to clean, complete responses
        and samples randomly for diversity."""
        params: dict[str, Any] = {"limit": limit}
        version_clause = ""
        if model_version:
            version_clause = "AND l.model_version = :model_version"
            params["model_version"] = model_version
        result = await self._db.execute(
            text(
                f"""
                SELECT l.* FROM llm_logs l
                WHERE l.finish_reason = 'stop'
                  AND l.latency_ms < 3000
                  {version_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM failure_classifications fc
                      WHERE fc.llm_log_id = l.id
                  )
                ORDER BY RANDOM()
                LIMIT :limit
                """
            ),
            params,
        )
        return list(result.fetchall())

    async def get_known_good_candidates(
        self, limit: int, model_version: str | None = None
    ) -> list[LLMLog]:
        """Known-good logs ordered NEWEST first (not random) so a caller can apply
        recency-weighted sampling (#T2). Same clean/complete filters as
        get_known_good_sample."""
        params: dict[str, Any] = {"limit": limit}
        version_clause = ""
        if model_version:
            version_clause = "AND l.model_version = :model_version"
            params["model_version"] = model_version
        result = await self._db.execute(
            text(
                f"""
                SELECT l.* FROM llm_logs l
                WHERE l.finish_reason = 'stop'
                  AND l.latency_ms < 3000
                  {version_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM failure_classifications fc
                      WHERE fc.llm_log_id = l.id
                  )
                ORDER BY l.created_at DESC
                LIMIT :limit
                """
            ),
            params,
        )
        return list(result.fetchall())
