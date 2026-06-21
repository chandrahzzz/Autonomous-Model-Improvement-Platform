from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.api.dependencies import get_db_session
from src.db.repositories.model_versions import ModelRepository

log = structlog.get_logger()
router = APIRouter()


@router.get("/current")
async def current_model(db: AsyncSession = Depends(get_db_session)) -> dict:
    repo = ModelRepository(db)
    version = await repo.get_production_version()
    if not version:
        raise HTTPException(status_code=404, detail="No production model found")
    return {
        "version_tag": version.version_tag,
        "base_model": version.base_model,
        "lora_weights_path": version.lora_weights_path,
        "promoted_at": version.promoted_at.isoformat() if version.promoted_at else None,
        "created_at": version.created_at.isoformat(),
    }


@router.get("/history")
async def model_history(
    limit: int = 20,
    db: AsyncSession = Depends(get_db_session),
) -> list[dict]:
    repo = ModelRepository(db)
    versions = await repo.get_version_history(limit=limit)
    return [
        {
            "version_tag": v.version_tag,
            "is_production": v.is_production,
            "is_archived": v.is_archived,
            "promoted_at": v.promoted_at.isoformat() if v.promoted_at else None,
            "rolled_back_at": v.rolled_back_at.isoformat() if v.rolled_back_at else None,
            "created_at": v.created_at.isoformat(),
        }
        for v in versions
    ]


@router.post("/rollback/{version_tag}")
async def rollback_to_version(
    version_tag: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Emergency manual rollback. Autonomous rollback goes through the graph."""
    repo = ModelRepository(db)
    await repo.rollback(version_tag)
    log.warning("manual_model_rollback", version_tag=version_tag)
    return {"rolled_back": True, "version_tag": version_tag}
