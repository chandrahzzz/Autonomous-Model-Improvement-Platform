"""Example curator node: curates failures into training examples."""
import structlog
from src.graph.state import PipelineState
from src.db.connection import get_db
from src.curation.curator import CurationPipeline
from src.detection.failure_classifier import FailureBatch, FailureEvent

log = structlog.get_logger()
_curator = CurationPipeline()

async def example_curator_node(state: PipelineState) -> PipelineState:
    failure_events = state.get("failure_events", [])
    if not failure_events:
        return {**state, "curated_examples": [], "curated_count": 0}

    failures = [
        FailureEvent(
            llm_log_id=f["llm_log_id"],
            prompt=f["prompt"],
            completion=f["completion"],
            failure_type=f["failure_type"],
            score=f["score"],
            metadata=f.get("metadata", {}),
        )
        for f in failure_events
    ]
    batch = FailureBatch(events=failures, drift_score=state.get("drift_score", 0.0))

    async with get_db() as db:
        saved = await _curator.curate(batch, db)

    log.info("example_curator_node_complete", saved=len(saved))
    return {**state, "curated_examples": saved, "curated_count": len(saved)}
