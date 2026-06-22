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
    # Whether this call was a retrieval-augmented (RAG) generation. When True the
    # hallucination detector EXPECTS retrieved_context; if it's missing the call
    # is counted (hallucination_premise_missing_total) and the resulting NLI score
    # is flagged ungrounded rather than silently graded against the bare prompt.
    is_rag: bool = False
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    finish_reason: str
    cost_usd: float
    embedding_hash: str = ""
    metadata: dict = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
