import uuid
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import FailureAttribution


class FailureAttributionRepository:
    """Thin data layer for failure_attributions (RFC-003)."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def insert(self, data: dict[str, Any]) -> FailureAttribution:
        row = FailureAttribution(**data)
        self._db.add(row)
        await self._db.flush()
        return row

    async def get_by_log_id(self, log_id: uuid.UUID) -> list[FailureAttribution]:
        result = await self._db.execute(
            select(FailureAttribution)
            .where(FailureAttribution.log_id == log_id)
            .order_by(FailureAttribution.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_by_model_version(self, version: str, limit: int = 100, offset: int = 0) -> list[FailureAttribution]:
        result = await self._db.execute(
            select(FailureAttribution)
            .where(FailureAttribution.model_version == version)
            .order_by(FailureAttribution.created_at.desc())
            .limit(limit).offset(offset)
        )
        return list(result.scalars().all())

    async def get_top_influential_examples(self, model_version: str, min_score: float = 0.70) -> list[dict]:
        """Aggregate across attribution rows: which training examples appear most
        often as high-influence. Raw SQL — JSONB unnest isn't clean in the ORM."""
        result = await self._db.execute(
            text(
                """
                SELECT elem->>'example_id' AS example_id,
                       COUNT(*) AS appearances,
                       AVG((elem->>'influence_score')::float) AS mean_influence_score,
                       MAX(elem->>'failure_type') AS failure_type,
                       MAX(elem->>'prompt_preview') AS prompt_preview
                FROM failure_attributions,
                     jsonb_array_elements(top_k_examples) AS elem
                WHERE model_version = :version
                  AND (elem->>'influence_score')::float >= :min_score
                GROUP BY example_id
                ORDER BY appearances DESC, mean_influence_score DESC
                LIMIT 50
                """
            ),
            {"version": model_version, "min_score": min_score},
        )
        return [
            {
                "example_id": r.example_id,
                "appearances": int(r.appearances),
                "mean_influence_score": round(float(r.mean_influence_score), 4),
                "failure_type": r.failure_type,
                "prompt_preview": r.prompt_preview,
            }
            for r in result.fetchall()
        ]

    async def delete_older_than(self, days: int = 30) -> int:
        cutoff = datetime.utcnow() - timedelta(days=days)
        result = await self._db.execute(
            delete(FailureAttribution).where(FailureAttribution.created_at < cutoff)
        )
        return result.rowcount or 0
