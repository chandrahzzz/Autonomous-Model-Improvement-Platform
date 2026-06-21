from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal


class TrainingEvent(BaseModel):
    event_type: Literal[
        "training_triggered",
        "training_started",
        "training_completed",
        "training_failed",
        "eval_started",
        "eval_completed",
        "model_promoted",
        "model_rolled_back",
    ]
    version_tag: str
    training_run_id: int | None = None
    payload: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
