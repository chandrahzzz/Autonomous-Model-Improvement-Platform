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

    rollbacks_total.labels(reason=reason).inc()
    log.warning("rollback_node_complete", rolled_back=version_tag, restored=prod_tag, reason=reason)
    return {**state, "production_version": prod_tag, "training_triggered": False}
