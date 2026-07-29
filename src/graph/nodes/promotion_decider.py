"""Promotion decider node: runs all promotion gates."""
import structlog
from src.graph.state import PipelineState
from src.shadow.promotion_gate import PromotionGate
from src.evaluation.eval_orchestrator import EvalResult

log = structlog.get_logger()
_gate = PromotionGate()

async def promotion_decider_node(state: PipelineState) -> PipelineState:
    ab_data = state.get("ab_data", {})
    eval_result_dict = state.get("eval_result", {})

    # Prefer the incumbent re-scored on the SAME eval-set snapshot as the
    # challenger (computed by EvalOrchestrator but previously only stashed in the
    # rationale and never used). It is the apples-to-apples comparison; the
    # stored state scores are the fallback when no re-eval ran.
    snapshot_incumbent = eval_result_dict.get("incumbent_scores_on_snapshot")
    if isinstance(snapshot_incumbent, dict) and snapshot_incumbent:
        incumbent_scores = snapshot_incumbent
        incumbent_source = "snapshot_reeval"
    else:
        incumbent_scores = state.get("incumbent_scores", {})
        incumbent_source = "stored_state"

    # Reconstruct EvalResult from state dict
    eval_result = EvalResult(
        version_tag=state.get("version_tag", "unknown"),
        passed=state.get("eval_passed", False),
        faithfulness=eval_result_dict.get("faithfulness", 0.0),
        answer_relevancy=eval_result_dict.get("answer_relevancy", 0.0),
        context_recall=eval_result_dict.get("context_recall", 0.0),
        safety_score=eval_result_dict.get("safety_score", 0.0),
    )

    decision = _gate.evaluate(ab_data, eval_result, incumbent_scores)
    log.info(
        "promotion_decider_node_complete",
        promote=decision.promote,
        reason=decision.reason,
        incumbent_source=incumbent_source,
    )
    return {**state, "promotion_decision": decision.promote, "rollback_reason": None if decision.promote else decision.reason}
