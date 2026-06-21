"""Training-example source tracking + retraction.

If a source document is updated or found wrong, these endpoints find every
training example grounded on it and let an operator retract them before the next
training run (retracted examples are excluded from `get_pending`).
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.db.repositories.training_examples import TrainingExampleRepository

router = APIRouter()


@router.get("/examples/by-source")
async def examples_by_source(
    source_id: str = Query(..., description="Grounding source/document/chunk ID"),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    rows = await TrainingExampleRepository(db).get_by_source(source_id)
    return {"source_id": source_id, "count": len(rows), "examples": rows}


@router.delete("/examples/by-source")
async def retract_examples_by_source(
    source_id: str = Query(..., description="Grounding source/document/chunk ID"),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    retracted = await TrainingExampleRepository(db).retract_by_source(source_id)
    return {"source_id": source_id, "retracted": retracted}
