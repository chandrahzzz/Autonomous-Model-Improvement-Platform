"""
Persistence for detected failures.

Before this, the `failure_classifications` table was READ in several places
(the known-good replay filter, attribution, audit lineage, the calibrator) but
never WRITTEN — so every failure log still counted as "known-good" and could be
sampled into the replay buffer as a positive example (training on the model's own
bad outputs). This repository writes one row per detected failure and lets the
detector reuse an existing row across cycles (the pipeline re-scans the same
recent logs each cycle, so we must not insert a duplicate every time).
"""

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import FailureClassification


class FailureClassificationRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def insert(self, data: dict[str, Any]) -> str:
        """Insert one failure classification; returns its id (as str)."""
        obj = FailureClassification(**data)
        self._db.add(obj)
        await self._db.flush()
        return str(obj.id)

    async def existing_for_logs(self, log_ids: list[str]) -> dict[str, str]:
        """{llm_log_id: classification_id} for logs that already have a row, so
        re-scanning the same logs across cycles reuses rather than duplicates.
        The pipeline collapses correlated failures to one event per log, so at
        most one classification per log is expected."""
        if not log_ids:
            return {}
        result = await self._db.execute(
            select(FailureClassification.llm_log_id, FailureClassification.id)
            .where(FailureClassification.llm_log_id.in_(log_ids))
        )
        return {str(lid): str(cid) for lid, cid in result.fetchall()}
