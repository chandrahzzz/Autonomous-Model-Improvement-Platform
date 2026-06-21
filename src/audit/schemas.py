from pydantic import BaseModel
from typing import Literal, Any


AuditEventType = Literal[
    "training_triggered",
    "training_completed",
    "training_failed",
    "eval_started",
    "eval_passed",
    "eval_failed",
    "shadow_started",
    "shadow_completed",
    "model_promoted",
    "model_rolled_back",
    "safety_regression_detected",
    "pipeline_paused",
    "pipeline_resumed",
    "drift_detected",
    "failure_batch_detected",
    "eval_set_updated",          # RFC-002: eval factory added examples
    "training_data_retracted",   # RFC-003: training examples surgically retracted
]


class AuditEvent(BaseModel):
    event_type: AuditEventType
    decision: str
    rationale: dict[str, Any]
    state_snapshot: dict[str, Any]
    model_version_before: str | None = None
    model_version_after: str | None = None
    operator: str = "autonomous_pipeline"
