"""Rollback node: reverts to previous production version."""
import structlog
from src.graph.state import PipelineState
from src.db.connection import get_db
from src.db.repositories.model_versions import ModelRepository
from src.monitoring.metrics import rollbacks_total

log = structlog.get_logger()

async def rollback_node(state: PipelineState) -> PipelineState:
    version_tag = state.get("version_tag", "unknown")
    reason = state.get("rollback_reason", "unknown")

    async with get_db() as db:
        repo = ModelRepository(db)
        await repo.rollback(version_tag)
        prod = await repo.get_production_version()
        prod_tag = prod.version_tag if prod else "unknown"

    # End the shadow test so the next challenger starts a clean window and the
    # router stops sampling against a version that is no longer a candidate.
    try:
        from src.shadow.service import _get_router
        await _get_router().clear_challenger()
    except Exception:
        log.warning("shadow_challenger_clear_failed", version=version_tag)

    rollbacks_total.labels(reason=reason).inc()
    log.warning("rollback_node_complete", rolled_back=version_tag, restored=prod_tag, reason=reason)
    # Clear transient error + in-flight markers so the NEXT cycle starts clean and
    # the conditional entry point can't misroute on stale state (#L2).
    return {
        **state,
        "production_version": prod_tag,
        "training_triggered": False,
        "modal_job_id": None,
        "training_status": "idle",
        "shadow_active": False,
        "shadow_ready_for_decision": False,
        "canary_active": False,
        "version_tag": None,
        "rollback_reason": None,
        # audit_logger_node infers its event type from state; a sticky decision
        # would mislabel the next cycle's failure-detection audit entry.
        "promotion_decision": None,
        "error": None,
        "error_node": None,
    }
