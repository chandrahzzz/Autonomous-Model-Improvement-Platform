"""Audit logger node: writes HMAC-signed audit entry before each major action."""
import structlog
from src.graph.state import PipelineState
from src.db.connection import get_db
from src.audit.logger import AuditLogger
from src.audit.schemas import AuditEvent

log = structlog.get_logger()

async def audit_logger_node(state: PipelineState) -> PipelineState:
    """Write an audit entry capturing the current state decision."""
    # Determine what event we're auditing
    if state.get("promotion_decision") is True:
        event_type = "model_promoted"
        decision = f"promote {state.get('version_tag')} to production"
    elif state.get("rollback_reason"):
        event_type = "model_rolled_back"
        decision = f"rollback {state.get('version_tag')}: {state.get('rollback_reason')}"
    elif state.get("training_triggered"):
        event_type = "training_triggered"
        decision = f"trigger training run for {state.get('version_tag')}"
    elif state.get("has_failures"):
        event_type = "failure_batch_detected"
        decision = f"detected {state.get('failure_count', 0)} failures"
    else:
        event_type = "drift_detected"
        decision = f"drift_score={state.get('drift_score', 0.0):.3f}"

    # State snapshot (exclude large arrays)
    snapshot = {
        k: v for k, v in state.items()
        if k not in ("failure_events", "curated_examples", "recent_log_ids")
        and not isinstance(v, (list, bytes))
    }

    event = AuditEvent(
        event_type=event_type,
        decision=decision,
        rationale=state.get("eval_result") or {},
        state_snapshot=snapshot,
        model_version_before=state.get("production_version"),
        model_version_after=state.get("version_tag") if state.get("promotion_decision") else None,
    )

    async with get_db() as db:
        audit = AuditLogger(db)
        row_id = await audit.log(event)

    log.info("audit_logger_node_complete", row_id=row_id, event_type=event_type)
    return {**state, "last_audit_id": row_id}
