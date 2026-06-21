"""Failure-attribution endpoints (RFC-003)."""

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.db.repositories.attribution import FailureAttributionRepository

router = APIRouter()


class AttributionRow(BaseModel):
    id: uuid.UUID
    log_id: uuid.UUID
    model_version: str
    training_run_id: int | None
    top_k_examples: list
    total_candidates_scored: int
    backend_used: str
    computation_ms: int
    created_at: datetime


class RetractRequest(BaseModel):
    example_ids: list[uuid.UUID]
    reason: str


def _to_row(r) -> AttributionRow:
    return AttributionRow(
        id=r.id, log_id=r.log_id, model_version=r.model_version,
        training_run_id=r.training_run_id, top_k_examples=r.top_k_examples,
        total_candidates_scored=r.total_candidates_scored, backend_used=r.backend_used,
        computation_ms=r.computation_ms, created_at=r.created_at,
    )


@router.get("/log/{log_id}", response_model=list[AttributionRow])
async def attribution_by_log(log_id: uuid.UUID, db: AsyncSession = Depends(get_db_session)):
    rows = await FailureAttributionRepository(db).get_by_log_id(log_id)
    return [_to_row(r) for r in rows]


@router.get("/model/{version}", response_model=list[AttributionRow])
async def attribution_by_model(
    version: str, limit: int = 50, offset: int = 0, db: AsyncSession = Depends(get_db_session)
):
    rows = await FailureAttributionRepository(db).get_by_model_version(version, limit=limit, offset=offset)
    return [_to_row(r) for r in rows]


@router.get("/influential")
async def influential_examples(
    version: str = Query(...),
    min_score: float = 0.70,
    limit: int = 20,
    db: AsyncSession = Depends(get_db_session),
) -> list[dict]:
    rows = await FailureAttributionRepository(db).get_top_influential_examples(version, min_score=min_score)
    return rows[:limit]


@router.post("/retract")
async def retract_examples(req: RetractRequest, db: AsyncSession = Depends(get_db_session)) -> dict:
    from src.audit.logger import AuditLogger
    from src.audit.schemas import AuditEvent
    from src.db.repositories.training_examples import TrainingExampleRepository
    from src.monitoring.metrics import retracted_examples_total

    if not req.example_ids:
        return {"retracted": 0, "audit_id": None}

    # Write-before-act: audit entry committed before the UPDATE.
    audit_id = await AuditLogger(db).log(AuditEvent(
        event_type="training_data_retracted",
        decision=f"Retract {len(req.example_ids)} training examples: {req.reason}",
        rationale={
            "example_ids": [str(i) for i in req.example_ids],
            "reason": req.reason,
            "requested_at": datetime.utcnow().isoformat(),
        },
        state_snapshot={"count": len(req.example_ids)},
        operator="human_operator",
    ))
    retracted = await TrainingExampleRepository(db).retract_by_ids(req.example_ids)
    retracted_examples_total.inc(retracted)
    return {"retracted": retracted, "audit_id": audit_id}
