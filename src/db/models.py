"""
SQLAlchemy 2.0 async ORM models. Mirrors the Alembic schema exactly.

MULTI-TENANCY: intentionally omitted. This is a single-tenant deployment, so
tables carry no `tenant_id`. To support multi-tenant SaaS later, add a
`tenant_id` (indexed) to every table, thread it through the Kafka event schema,
repositories, the LLM interceptor, and the LangGraph pipeline state, and
namespace per-tenant work. That is a dedicated migration, not a small change.
"""

from datetime import datetime
from sqlalchemy import (
    BigInteger, Boolean, Column, DateTime, Float, Integer,
    Numeric, String, Text, ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.orm import DeclarativeBase
import uuid


class Base(DeclarativeBase):
    pass


class LLMLog(Base):
    __tablename__ = "llm_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(Text, nullable=False)
    user_cohort = Column(Text)
    model_version = Column(Text, nullable=False)
    prompt = Column(Text, nullable=False)
    completion = Column(Text, nullable=False)
    retrieved_context = Column(Text)  # RAG grounding context for hallucination NLI
    prompt_tokens = Column(Integer, nullable=False)
    completion_tokens = Column(Integer, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    finish_reason = Column(Text, nullable=False)
    cost_usd = Column(Numeric(12, 8), nullable=False)
    embedding_hash = Column(Text)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class FailureClassification(Base):
    __tablename__ = "failure_classifications"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    llm_log_id = Column(UUID(as_uuid=True), ForeignKey("llm_logs.id", ondelete="CASCADE"), nullable=False)
    failure_type = Column(Text, nullable=False)
    score = Column(Float, nullable=False)
    cluster_id = Column(Integer)
    cluster_label = Column(Text)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class TrainingExample(Base):
    __tablename__ = "training_examples"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    failure_id = Column(UUID(as_uuid=True), ForeignKey("failure_classifications.id"))
    llm_log_id = Column(UUID(as_uuid=True), ForeignKey("llm_logs.id"))
    prompt = Column(Text, nullable=False)
    bad_completion = Column(Text, nullable=False)
    corrected_completion = Column(Text, nullable=False)
    failure_type = Column(Text, nullable=False)
    cluster_id = Column(Integer)
    teacher_model = Column(Text, nullable=False)
    teacher_confidence = Column(Float, nullable=False)
    pii_scrubbed = Column(Boolean, nullable=False, default=False)
    dedup_hash = Column(Text, nullable=False)
    quality_score = Column(Float, nullable=False)
    # NLI entailment of the teacher correction vs the retrieved_context that the
    # production answer used. NULL when no RAG context was available (general
    # knowledge failures are still valid training data).
    grounding_score = Column(Float)
    # Document/chunk IDs from retrieved_context that grounded this example —
    # enables retraction if a source doc changes. NULL/empty when no context.
    grounding_sources = Column(ARRAY(Text))
    # Set when an example is retracted (e.g. its grounding source was withdrawn).
    retracted_at = Column(DateTime(timezone=True))
    included_in_run = Column(Integer)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)

    __table_args__ = (UniqueConstraint("dedup_hash", name="uq_training_dedup"),)


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_tag = Column(Text, nullable=False, unique=True)
    base_model = Column(Text, nullable=False)
    lora_weights_path = Column(Text)
    is_production = Column(Boolean, nullable=False, default=False)
    is_archived = Column(Boolean, nullable=False, default=False)
    training_run_id = Column(Integer)
    promoted_at = Column(DateTime(timezone=True))
    rolled_back_at = Column(DateTime(timezone=True))
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version_tag = Column(Text, nullable=False)
    modal_job_id = Column(Text)
    status = Column(Text, nullable=False, default="submitted")
    dataset_size = Column(Integer, nullable=False)
    dataset_path = Column(Text)
    dataset_uri = Column(Text)  # Durable, reproducible pointer to the exact dataset
    lora_config = Column(JSONB, nullable=False)
    final_loss = Column(Float)
    wandb_run_id = Column(Text)
    wandb_run_url = Column(Text)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    training_run_id = Column(Integer, ForeignKey("training_runs.id"))
    version_tag = Column(Text, nullable=False)
    eval_type = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="running")
    faithfulness = Column(Float)
    answer_relevancy = Column(Float)
    context_recall = Column(Float)
    safety_score = Column(Float)
    ab_quality_delta = Column(Float)
    ab_pvalue = Column(Float)
    ab_cohens_d = Column(Float)
    ab_n_requests = Column(Integer)
    gate_passed = Column(Boolean)
    rationale = Column(JSONB, default=dict)
    eval_set_version = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    completed_at = Column(DateTime(timezone=True))


class AuditTrail(Base):
    __tablename__ = "audit_trail"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    event_type = Column(Text, nullable=False)
    decision = Column(Text, nullable=False)
    rationale = Column(JSONB, nullable=False)
    state_snapshot = Column(JSONB, nullable=False)
    model_version_before = Column(Text)
    model_version_after = Column(Text)
    operator = Column(Text, nullable=False, default="autonomous_pipeline")
    hmac_sha256 = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class DriftBaseline(Base):
    __tablename__ = "drift_baselines"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(Text, nullable=False)
    sample_size = Column(Integer, nullable=False)
    centroid = Column(JSONB, nullable=False)
    covariance_inv = Column(JSONB, nullable=False)
    computed_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    is_active = Column(Boolean, nullable=False, default=False)


class DriftTrendHistory(Base):
    """Append-only snapshots of the predictive drift early-warning system
    (RFC-001). Operational data, not audit — safe to prune."""
    __tablename__ = "drift_trend_history"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    cycle_id = Column(Text)
    model_version = Column(Text)
    current_score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    slope_per_cycle = Column(Float, nullable=False)
    r_squared = Column(Float, nullable=False)
    predicted_trigger_hours = Column(Float)  # None if slope <= 0
    window_size = Column(Integer, nullable=False)
    trend_direction = Column(Text, nullable=False)  # stable | increasing | decreasing
    is_alarming = Column(Boolean, nullable=False, default=False)
    alert_sent = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class EvalSet(Base):
    """Held-out evaluation set. Originally seed-only (migration 003); the eval
    factory (RFC-002, migration 006) adds living, traffic-derived examples."""
    __tablename__ = "eval_set"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(Text, nullable=False, server_default="v1")
    question = Column(Text, nullable=False)
    context = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=False)
    domain = Column(Text)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    # ── Added in migration 006 (eval factory) ──
    source = Column(Text, nullable=False, server_default="seed")  # seed | factory
    cluster_id = Column(Integer)
    cluster_label = Column(Text)
    factory_confidence = Column(Float)
    access_count = Column(Integer, nullable=False, server_default="0")
    last_accessed_at = Column(DateTime(timezone=True))
    evicted_at = Column(DateTime(timezone=True))
    embedding = Column(JSONB)  # list[float], for dedup cosine checks


class FailureAttribution(Base):
    """Embedding-influence attribution of a failure to training examples (RFC-003).
    Operational data, not audit — safe to prune."""
    __tablename__ = "failure_attributions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    log_id = Column(UUID(as_uuid=True), ForeignKey("llm_logs.id", ondelete="CASCADE"), nullable=False)
    failure_classification_id = Column(
        UUID(as_uuid=True), ForeignKey("failure_classifications.id", ondelete="SET NULL"), nullable=True
    )
    model_version = Column(Text, nullable=False)
    # training_runs.id is an integer (autoincrement), so this FK is Integer, not UUID.
    training_run_id = Column(Integer, ForeignKey("training_runs.id", ondelete="SET NULL"), nullable=True)
    top_k_examples = Column(JSONB, nullable=False)
    total_candidates_scored = Column(Integer, nullable=False)
    backend_used = Column(Text, nullable=False, server_default="embedding_cosine")
    computation_ms = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


class KnowledgeDocument(Base):
    """Domain documents/chunks for retrieval. Lets the grounded teacher fetch
    context when the upstream app didn't attach `retrieved_context`. Embeddings
    are stored as JSONB and scanned with numpy cosine (no pgvector)."""
    __tablename__ = "knowledge_documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_id = Column(Text, nullable=False)  # document / chunk identifier
    content = Column(Text, nullable=False)
    embedding = Column(JSONB)  # list[float] (MiniLM, 384-dim)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
