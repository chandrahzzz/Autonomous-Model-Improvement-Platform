from datetime import datetime
from typing import Any
from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert

from src.db.models import TrainingExample


class TrainingExampleRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def upsert(self, data: dict[str, Any]) -> TrainingExample | None:
        """Insert, ignoring duplicates by dedup_hash. Returns None on conflict."""
        stmt = (
            insert(TrainingExample)
            .values(**data)
            .on_conflict_do_nothing(index_elements=["dedup_hash"])
            .returning(TrainingExample)
        )
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def count_pending(self) -> int:
        """Count examples not yet assigned to a training run."""
        result = await self._db.execute(
            select(func.count()).where(TrainingExample.included_in_run.is_(None))
        )
        return result.scalar_one()

    async def all_for_dedup(self, limit: int = 50000) -> list[tuple[str, str]]:
        """(prompt, corrected_completion) for every stored example — used to
        rehydrate the in-memory MinHash near-dup index on startup. Newest first
        so the cap keeps the most recent examples."""
        result = await self._db.execute(
            select(TrainingExample.prompt, TrainingExample.corrected_completion)
            .order_by(TrainingExample.created_at.desc())
            .limit(limit)
        )
        return [(r[0], r[1]) for r in result.fetchall()]

    async def get_pending(self, limit: int = 2000) -> list[TrainingExample]:
        result = await self._db.execute(
            select(TrainingExample)
            .where(TrainingExample.included_in_run.is_(None))
            .where(TrainingExample.retracted_at.is_(None))  # don't train on retracted examples
            .order_by(TrainingExample.quality_score.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_used(self, ids: list[str], run_id: int) -> None:
        await self._db.execute(
            update(TrainingExample)
            .where(TrainingExample.id.in_(ids))
            .values(included_in_run=run_id)
        )

    async def get_used_for_run(self, training_run_id: int, limit: int = 500) -> list[TrainingExample]:
        """Examples used in a specific training run — candidates for influence
        scoring (RFC-003). Excludes retracted examples."""
        result = await self._db.execute(
            select(TrainingExample)
            .where(TrainingExample.included_in_run == training_run_id)
            .where(TrainingExample.retracted_at.is_(None))
            .order_by(TrainingExample.quality_score.desc())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_source(self, source_id: str) -> list[dict[str, Any]]:
        """Examples grounded on a given source document/chunk (not yet retracted)."""
        result = await self._db.execute(
            select(
                TrainingExample.id,
                TrainingExample.grounding_score,
                TrainingExample.failure_type,
                TrainingExample.created_at,
            )
            .where(TrainingExample.grounding_sources.any(source_id))
            .where(TrainingExample.retracted_at.is_(None))
            .order_by(TrainingExample.created_at.desc())
        )
        return [
            {
                "id": str(r.id),
                "grounding_score": r.grounding_score,
                "failure_type": r.failure_type,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in result.fetchall()
        ]

    async def retract_by_ids(self, ids: list) -> int:
        """Retract specific examples by id (RFC-003 surgical removal). Re-queues
        them as pending (included_in_run=NULL) but retracted_at excludes them
        from get_pending(). Returns rows affected."""
        if not ids:
            return 0
        result = await self._db.execute(
            update(TrainingExample)
            .where(TrainingExample.id.in_(ids))
            .where(TrainingExample.retracted_at.is_(None))
            .values(retracted_at=datetime.utcnow(), included_in_run=None)
        )
        return result.rowcount or 0

    async def retract_by_source(self, source_id: str) -> int:
        """Mark all (non-retracted) examples grounded on a source as retracted.
        Returns the number of rows affected."""
        result = await self._db.execute(
            update(TrainingExample)
            .where(TrainingExample.grounding_sources.any(source_id))
            .where(TrainingExample.retracted_at.is_(None))
            .values(retracted_at=datetime.utcnow())
        )
        return result.rowcount or 0
