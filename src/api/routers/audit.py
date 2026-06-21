from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.db.repositories.audit_trail import AuditRepository
from src.audit.hmac_signer import HMACSigner

router = APIRouter()


@router.get("/trail")
async def get_audit_trail(
    limit: int = 100,
    db: AsyncSession = Depends(get_db_session),
) -> list[dict]:
    repo = AuditRepository(db)
    entries = await repo.get_recent(limit=limit)
    return [
        {
            "id": e.id,
            "event_type": e.event_type,
            "decision": e.decision,
            "rationale": e.rationale,
            "model_version_before": e.model_version_before,
            "model_version_after": e.model_version_after,
            "operator": e.operator,
            "created_at": e.created_at.isoformat(),
        }
        for e in entries
    ]


@router.get("/verify/{row_id}")
async def verify_audit_entry(
    row_id: int,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Verify the HMAC signature of a single audit row."""
    repo = AuditRepository(db)
    entry = await repo.get_by_id(row_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Audit entry not found")

    signer = HMACSigner()
    payload = {
        "id": entry.id,
        "event_type": entry.event_type,
        "decision": entry.decision,
        "rationale": entry.rationale,
        "state_snapshot": entry.state_snapshot,
        "operator": entry.operator,
        "created_at": entry.created_at.isoformat(),
    }
    valid = signer.verify(payload, entry.hmac_sha256)
    return {"row_id": row_id, "valid": valid}


@router.get("/lineage/{model_version}")
async def get_model_lineage(
    model_version: str,
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Full data lineage for a model version:
    llm_logs -> failure_classifications -> training_examples -> training_run -> model_version.
    Used by auditors/engineers to trace why a model learned a given behavior.
    """
    mv = (await db.execute(
        text("SELECT version_tag, training_run_id FROM model_versions WHERE version_tag = :v"),
        {"v": model_version},
    )).fetchone()
    if not mv:
        raise HTTPException(status_code=404, detail="Model version not found")

    run_id = mv.training_run_id
    dataset_uri = None
    count = 0
    sample: list[dict] = []
    if run_id is not None:
        tr = (await db.execute(
            text("SELECT dataset_uri FROM training_runs WHERE id = :r"), {"r": run_id}
        )).fetchone()
        dataset_uri = tr.dataset_uri if tr else None
        count = int((await db.execute(
            text("SELECT COUNT(*) FROM training_examples WHERE included_in_run = :r"),
            {"r": run_id},
        )).scalar() or 0)
        rows = (await db.execute(
            text(
                """
                SELECT te.id AS training_example_id, te.failure_id, te.llm_log_id,
                       te.failure_type, fc.score AS failure_score, l.prompt AS original_prompt
                FROM training_examples te
                LEFT JOIN failure_classifications fc ON te.failure_id = fc.id
                LEFT JOIN llm_logs l ON te.llm_log_id = l.id
                WHERE te.included_in_run = :r
                LIMIT 10
                """
            ),
            {"r": run_id},
        )).fetchall()
        sample = [
            {
                "training_example_id": str(r.training_example_id),
                "failure_id": str(r.failure_id) if r.failure_id else None,
                "original_log_id": str(r.llm_log_id) if r.llm_log_id else None,
                "original_prompt_preview": (r.original_prompt or "")[:160],
                "failure_type": r.failure_type,
                "failure_score": r.failure_score,
            }
            for r in rows
        ]

    return {
        "model_version": mv.version_tag,
        "training_run_id": run_id,
        "dataset_uri": dataset_uri,
        "training_examples_count": count,
        "example_sample": sample,
    }
