"""
All conditional edge functions for the LangGraph pipeline.

Each function returns the name of the next node to route to.
Returning END exits the graph for this cycle; the outer run_forever loop
re-invokes after 60 seconds with the merged state.
"""

from langgraph.graph import END
from src.graph.state import PipelineState


def after_failure_detector(state: PipelineState) -> str:
    if state.get("has_failures"):
        return "example_curator"
    return "data_validator"


def after_data_validator(state: PipelineState) -> str:
    if state.get("paused"):
        return END
    return "fine_tune_trigger"


def after_fine_tune_trigger(state: PipelineState) -> str:
    if state.get("training_triggered"):
        return "audit_logger_pre_train"
    return END


def after_training_poller(state: PipelineState) -> str:
    status = state.get("training_status", "running")
    if status == "completed":
        return "eval_runner"
    if status in ("failed", "timeout"):
        return "rollback_node"
    return END  # still running; re-check next cycle


def after_eval_runner(state: PipelineState) -> str:
    if state.get("eval_passed"):
        return "ab_test_node"
    return "audit_logger_rollback"


def after_ab_test(state: PipelineState) -> str:
    if state.get("shadow_ready_for_decision"):
        return "promotion_decider"
    return END  # window not complete; re-check next cycle


def after_promotion_decider(state: PipelineState) -> str:
    if state.get("promotion_decision"):
        return "canary_node"   # gate full promotion behind a live canary
    return "audit_logger_rollback"


def after_canary(state: PipelineState) -> str:
    decision = state.get("canary_decision")
    if decision == "promote":
        return "audit_logger_promote"
    if decision == "rollback":
        return "audit_logger_rollback"
    return END   # canary window still open; re-check next cycle
