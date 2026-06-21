from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class LLMEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    user_cohort: str = "default"
    model_version: str
    prompt: str
    completion: str
    # Retrieved grounding context for RAG calls. Enables true factual-consistency
    # (NLI entailment) checks in the hallucination detector instead of falling
    # back to the prompt. Empty for non-RAG traffic.
    retrieved_context: str = ""
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    finish_reason: str
    cost_usd: float
    embedding_hash: str = ""
    metadata: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
