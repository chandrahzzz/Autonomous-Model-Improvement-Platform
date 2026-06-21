from datetime import datetime
from typing import Any
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import ModelVersion, DriftBaseline, TrainingRun


class ModelRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def get_production_version(self) -> ModelVersion | None:
        result = await self._db.execute(
            select(ModelVersion).where(ModelVersion.is_production == True)
        )
        return result.scalar_one_or_none()

    async def create_version(self, data: dict[str, Any]) -> ModelVersion:
        mv = ModelVersion(**data)
        self._db.add(mv)
        await self._db.flush()
        return mv

    async def promote(self, version_tag: str) -> None:
        """Atomically swap production pointer."""
        await self._db.execute(
            update(ModelVersion).values(is_production=False)
        )
        await self._db.execute(
            update(ModelVersion)
            .where(ModelVersion.version_tag == version_tag)
            .values(is_production=True, promoted_at=datetime.utcnow())
        )

    async def rollback(self, version_tag: str) -> None:
        await self._db.execute(
            update(ModelVersion)
            .where(ModelVersion.version_tag == version_tag)
            .values(rolled_back_at=datetime.utcnow(), is_production=False)
        )

    async def get_active_baseline(self) -> dict | None:
        result = await self._db.execute(
            select(DriftBaseline).where(DriftBaseline.is_active == True)
        )
        baseline = result.scalar_one_or_none()
        if baseline is None:
            return None
        import json
        return {
            "centroid": json.dumps(baseline.centroid),
            "covariance_inv": json.dumps(baseline.covariance_inv),
            "sample_size": baseline.sample_size,
        }

    async def save_baseline(self, model_version: str, data: dict[str, Any]) -> None:
        """Deactivate existing baselines and save new one."""
        await self._db.execute(
            update(DriftBaseline).values(is_active=False)
        )
        baseline = DriftBaseline(
            model_version=model_version,
            sample_size=data["sample_size"],
            centroid=data["centroid"],
            covariance_inv=data["covariance_inv"],
            is_active=True,
        )
        self._db.add(baseline)
        await self._db.flush()

    async def create_training_run(self, data: dict[str, Any]) -> TrainingRun:
        run = TrainingRun(**data)
        self._db.add(run)
        await self._db.flush()
        return run

    async def update_training_run(self, run_id: int, data: dict[str, Any]) -> None:
        await self._db.execute(
            update(TrainingRun).where(TrainingRun.id == run_id).values(**data)
        )

    async def get_training_run(self, run_id: int) -> TrainingRun | None:
        result = await self._db.execute(
            select(TrainingRun).where(TrainingRun.id == run_id)
        )
        return result.scalar_one_or_none()

    async def get_training_run_for_version(self, version_tag: str) -> int | None:
        """Return the completed training run id for a model version, or None
        (e.g. the seed model v7 was never trained by this pipeline). RFC-003."""
        result = await self._db.execute(
            select(TrainingRun.id)
            .where(TrainingRun.version_tag == version_tag, TrainingRun.status == "completed")
            .order_by(TrainingRun.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_version_history(self, limit: int = 20) -> list[ModelVersion]:
        result = await self._db.execute(
            select(ModelVersion)
            .order_by(ModelVersion.created_at.desc())
            .limit(limit)
        )
        return list(result.scalars().all())
