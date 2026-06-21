"""Eval factory / eval-set management endpoints (RFC-002)."""

import asyncio
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.config.settings import settings
from src.db.models import EvalSet
from src.db.repositories.eval_set import EvalSetRepository

router = APIRouter()


class EvalExample(BaseModel):
    id: int
    question: str
    ground_truth: str
    source: str
    factory_confidence: float | None = None
    cluster_id: int | None = None
    access_count: int
    last_accessed_at: datetime | None = None
    created_at: datetime


@router.get("/set/summary")
async def eval_set_summary(db: AsyncSession = Depends(get_db_session)) -> dict:
    return await EvalSetRepository(db).summary()


@router.get("/set/examples", response_model=list[EvalExample])
async def eval_set_examples(
    source: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db: AsyncSession = Depends(get_db_session),
) -> list[EvalExample]:
    rows = await EvalSetRepository(db).list_active(source=source, limit=limit, offset=offset)
    return [
        EvalExample(
            id=r.id, question=r.question, ground_truth=r.ground_truth, source=r.source,
            factory_confidence=r.factory_confidence, cluster_id=r.cluster_id,
            access_count=r.access_count, last_accessed_at=r.last_accessed_at,
            created_at=r.created_at,
        )
        for r in rows
    ]


@router.get("/set/factory/history")
async def eval_factory_history(
    limit: int = 20, db: AsyncSession = Depends(get_db_session)
) -> list[dict]:
    result = await db.execute(
        text(
            "SELECT id, decision, rationale, created_at FROM audit_trail "
            "WHERE event_type = 'eval_set_updated' ORDER BY created_at DESC LIMIT :lim"
        ),
        {"lim": limit},
    )
    return [
        {
            "id": r.id, "decision": r.decision, "rationale": r.rationale,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in result.fetchall()
    ]


@router.post("/set/factory/trigger")
async def eval_factory_trigger(db: AsyncSession = Depends(get_db_session)) -> dict:
    """Manually run one factory pass over the last 1000 prompts (90s timeout)."""
    from src.db.repositories.llm_logs import LLMLogRepository
    from src.graph.nodes.log_monitor import _get_openai, _get_factory_redis
    from src.evaluation.eval_factory import EvalFactory

    log_rows = (await db.execute(
        text("SELECT prompt FROM llm_logs ORDER BY created_at DESC LIMIT 1000")
    )).fetchall()
    prompts = [r.prompt for r in log_rows if r.prompt and len(r.prompt.strip()) > 10]
    if len(prompts) < 10:
        return {"added": 0, "skipped": 0, "reason": "not_enough_prompts"}

    factory = EvalFactory(
        eval_repo=EvalSetRepository(db),
        openai_client=_get_openai(),
        redis_client=_get_factory_redis(),
    )
    try:
        added = await asyncio.wait_for(factory.run(recent_prompts=prompts, db=db), timeout=90.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Factory run timed out")
    return {"added": added}


@router.delete("/set/examples/{example_id}")
async def delete_eval_example(
    example_id: int,
    reason: str = Query(..., description="Why this example is being removed"),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    repo = EvalSetRepository(db)
    example = await repo.get_one(example_id)
    if example is None:
        raise HTTPException(status_code=404, detail="Eval example not found")
    if example.source == "seed":
        raise HTTPException(status_code=403, detail="Seed examples cannot be deleted")

    # Write-before-act: audit, then soft-delete.
    from src.audit.logger import AuditLogger
    from src.audit.schemas import AuditEvent
    await AuditLogger(db).log(AuditEvent(
        event_type="eval_set_updated",
        decision=f"Manually evicted eval example {example_id}: {reason}",
        rationale={"example_id": example_id, "reason": reason},
        state_snapshot={"example_id": example_id},
        operator="human_operator",
    ))
    await repo.soft_delete(example_id)
    return {"id": example_id, "evicted_at": datetime.utcnow().isoformat(), "reason": reason}
