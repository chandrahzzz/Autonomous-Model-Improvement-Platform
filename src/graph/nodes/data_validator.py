"""Data validator node: counts pending training examples."""
import structlog
from src.graph.state import PipelineState
from src.db.connection import AsyncSessionLocal
from src.db.repositories.training_examples import TrainingExampleRepository
from src.monitoring.metrics import pending_examples_gauge

log = structlog.get_logger()

async def data_validator_node(state: PipelineState) -> PipelineState:
    async with AsyncSessionLocal() as db:
        repo = TrainingExampleRepository(db)
        count = await repo.count_pending()

    pending_examples_gauge.set(count)
    log.info("data_validator_node_complete", pending=count)
    return {**state, "pending_examples": count, "data_validated": True}
