"""Repository for the held-out evaluation set (`eval_set` table).

The eval set is what the promotion gate's RAGAS scoring runs against. The
factory (RFC-002) adds living, traffic-derived examples (source='factory')
alongside the seed examples; RAGAS picks them up automatically.
"""

import uuid as _uuid

import numpy as np
from sqlalchemy import select, func, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import EvalSet


class EvalSetRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_eval_set(self, version: str | None = None) -> list[dict]:
        """Return active (non-evicted) eval examples as dicts with
        id/question/context/ground_truth. `id` lets the eval node mark them
        accessed for LRU tracking."""
        stmt = select(
            EvalSet.id, EvalSet.question, EvalSet.context, EvalSet.ground_truth
        ).where(EvalSet.evicted_at.is_(None))
        if version:
            stmt = stmt.where(EvalSet.version == version)
        stmt = stmt.order_by(EvalSet.id)
        result = await self._db.execute(stmt)
        return [
            {
                "id": row.id,
                "question": row.question,
                "context": row.context,
                "ground_truth": row.ground_truth,
            }
            for row in result.fetchall()
        ]

    async def count(self, version: str | None = None) -> int:
        stmt = select(func.count()).select_from(EvalSet).where(EvalSet.evicted_at.is_(None))
        if version:
            stmt = stmt.where(EvalSet.version == version)
        result = await self._db.execute(stmt)
        return int(result.scalar() or 0)

    # ── Eval factory (RFC-002) ──────────────────────────────────────────────

    async def get_active(self, limit: int = 500) -> list[EvalSet]:
        """All non-evicted examples, oldest-accessed first (NULLS FIRST so
        never-accessed examples sort before recently-accessed ones)."""
        result = await self._db.execute(
            select(EvalSet)
            .where(EvalSet.evicted_at.is_(None))
            .order_by(EvalSet.last_accessed_at.asc().nullsfirst())
            .limit(limit)
        )
        return list(result.scalars().all())

    async def mark_accessed(self, ids: list[int]) -> None:
        if not ids:
            return
        await self._db.execute(
            update(EvalSet)
            .where(EvalSet.id.in_(ids))
            .values(access_count=EvalSet.access_count + 1, last_accessed_at=func.now())
        )

    async def evict_oldest(self, count: int) -> int:
        """Soft-delete the `count` oldest-accessed FACTORY examples (never seed).
        Returns rows actually evicted."""
        if count <= 0:
            return 0
        subq = (
            select(EvalSet.id)
            .where(EvalSet.source == "factory", EvalSet.evicted_at.is_(None))
            .order_by(EvalSet.last_accessed_at.asc().nullsfirst())
            .limit(count)
            .scalar_subquery()
        )
        result = await self._db.execute(
            update(EvalSet).where(EvalSet.id.in_(subq)).values(evicted_at=func.now())
        )
        return result.rowcount or 0

    async def count_active_by_source(self) -> dict[str, int]:
        result = await self._db.execute(
            select(EvalSet.source, func.count())
            .where(EvalSet.evicted_at.is_(None))
            .group_by(EvalSet.source)
        )
        counts = {"seed": 0, "factory": 0}
        for source, n in result.fetchall():
            counts[source] = int(n)
        return counts

    async def insert_factory_example(self, data: dict) -> EvalSet:
        row = EvalSet(
            version="v1",
            source="factory",
            question=data["question"],
            context=data.get("context", ""),
            ground_truth=data["ground_truth"],
            cluster_id=data.get("cluster_id"),
            cluster_label=data.get("cluster_label"),
            factory_confidence=data.get("factory_confidence"),
            embedding=data.get("embedding"),
        )
        self._db.add(row)
        await self._db.flush()
        return row

    async def exists_similar(self, embedding: list[float], threshold: float = 0.90) -> bool:
        """True if any active example's stored embedding has cosine similarity
        >= threshold with the candidate embedding (numpy scan, no pgvector)."""
        result = await self._db.execute(
            select(EvalSet.embedding).where(
                EvalSet.evicted_at.is_(None), EvalSet.embedding.isnot(None)
            )
        )
        stored = [row[0] for row in result.fetchall() if row[0]]
        if not stored:
            return False
        cand = np.asarray(embedding, dtype=float)
        cand_norm = np.linalg.norm(cand)
        if cand_norm == 0:
            return False
        cand = cand / cand_norm
        mat = np.asarray(stored, dtype=float)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1.0
        mat = mat / norms[:, None]
        sims = mat @ cand
        return bool(np.max(sims) >= threshold)

    # ── API helpers (RFC-002) ───────────────────────────────────────────────

    async def list_active(
        self, source: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[EvalSet]:
        stmt = select(EvalSet).where(EvalSet.evicted_at.is_(None))
        if source:
            stmt = stmt.where(EvalSet.source == source)
        stmt = stmt.order_by(EvalSet.created_at.desc()).limit(limit).offset(offset)
        result = await self._db.execute(stmt)
        return list(result.scalars().all())

    async def summary(self) -> dict:
        counts = await self.count_active_by_source()
        newest = await self._db.execute(
            select(func.max(EvalSet.created_at)).where(EvalSet.evicted_at.is_(None))
        )
        oldest_acc = await self._db.execute(
            select(func.min(EvalSet.last_accessed_at)).where(EvalSet.evicted_at.is_(None))
        )
        return {
            "total_active": sum(counts.values()),
            "by_source": counts,
            "newest_created_at": newest.scalar(),
            "oldest_last_accessed_at": oldest_acc.scalar(),
        }

    async def get_one(self, example_id: int) -> EvalSet | None:
        result = await self._db.execute(select(EvalSet).where(EvalSet.id == example_id))
        return result.scalar_one_or_none()

    async def soft_delete(self, example_id: int) -> None:
        await self._db.execute(
            update(EvalSet).where(EvalSet.id == example_id).values(evicted_at=func.now())
        )
