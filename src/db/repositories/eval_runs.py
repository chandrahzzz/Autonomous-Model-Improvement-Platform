from datetime import datetime
from typing import Any
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import EvalRun


class EvalRunRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, data: dict[str, Any]) -> EvalRun:
        run = EvalRun(**data)
        self._db.add(run)
        await self._db.flush()
        return run

    async def update(self, run_id: int, data: dict[str, Any]) -> None:
        data["completed_at"] = datetime.utcnow()
        await self._db.execute(
            update(EvalRun).where(EvalRun.id == run_id).values(**data)
        )

    async def get_latest_for_version(self, version_tag: str, eval_type: str) -> EvalRun | None:
        result = await self._db.execute(
            select(EvalRun)
            .where(EvalRun.version_tag == version_tag, EvalRun.eval_type == eval_type)
            .order_by(EvalRun.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_all_for_run(self, training_run_id: int) -> list[EvalRun]:
        result = await self._db.execute(
            select(EvalRun).where(EvalRun.training_run_id == training_run_id)
        )
        return list(result.scalars().all())
