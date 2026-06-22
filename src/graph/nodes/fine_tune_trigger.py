"""Fine-tune trigger node: decides whether to start a training run."""
import structlog
from datetime import datetime
from src.graph.state import PipelineState
from src.db.connection import get_db
from src.db.repositories.training_examples import TrainingExampleRepository
from src.training.trigger import TrainingTrigger

log = structlog.get_logger()
_trigger = TrainingTrigger()

async def fine_tune_trigger_node(state: PipelineState) -> PipelineState:
    pending = state.get("pending_examples", 0)
    drift = state.get("drift_score", 0.0)
    last_at_str = state.get("training_submitted_at")
    last_at = datetime.fromisoformat(last_at_str) if last_at_str else None

    # Failure-type distribution of pending examples drives the soft drift gate:
    # a format/refusal-dominant backlog can trigger on count alone (#T3).
    failure_type_counts: dict[str, int] = {}
    try:
        async with get_db() as db:
            failure_type_counts = await TrainingExampleRepository(db).pending_failure_type_counts()
    except Exception:
        log.warning("pending_failure_type_counts_failed")

    should, reason = _trigger.should_trigger(pending, drift, last_at, failure_type_counts)
    log.info("fine_tune_trigger_node_complete", triggered=should, reason=reason)
    return {**state, "training_triggered": should}
