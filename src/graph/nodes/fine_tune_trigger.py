"""Fine-tune trigger node: decides whether to start a training run."""
import structlog
from datetime import datetime
from src.graph.state import PipelineState
from src.training.trigger import TrainingTrigger

log = structlog.get_logger()
_trigger = TrainingTrigger()

async def fine_tune_trigger_node(state: PipelineState) -> PipelineState:
    pending = state.get("pending_examples", 0)
    drift = state.get("drift_score", 0.0)
    last_at_str = state.get("training_submitted_at")
    last_at = datetime.fromisoformat(last_at_str) if last_at_str else None

    should, reason = _trigger.should_trigger(pending, drift, last_at)
    log.info("fine_tune_trigger_node_complete", triggered=should, reason=reason)
    return {**state, "training_triggered": should}
