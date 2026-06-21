"""A/B test node: manages shadow traffic window."""
import structlog
from src.graph.state import PipelineState
from src.db.connection import AsyncSessionLocal
from src.shadow.ab_collector import ABCollector

log = structlog.get_logger()

async def ab_test_node(state: PipelineState) -> PipelineState:
    version_tag = state.get("version_tag", "unknown")

    async with AsyncSessionLocal() as db:
        collector = ABCollector(db)
        ab_data = await collector.collect_window(version_tag)

    log.info("ab_test_node_complete", version=version_tag, ready=ab_data["ready"], n=ab_data["n_requests"])
    return {
        **state,
        "shadow_active": True,
        "shadow_ready_for_decision": ab_data["ready"],
        "ab_data": ab_data,
    }
