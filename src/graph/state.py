"""
PipelineState — single source of truth for the LangGraph state machine.

Every node reads from and writes to this TypedDict.
LangGraph persists this to PostgreSQL checkpoints between cycles.
"""

from typing import TypedDict, Annotated
from datetime import datetime
import operator


class PipelineState(TypedDict, total=False):
    # ── Cycle metadata ──────────────────────────────────────────────────────
    cycle_id: str
    cycle_start_at: str
    last_cycle_at: str

    # ── Log monitoring ───────────────────────────────────────────────────────
    recent_log_ids: list[str]
    log_batch_size: int

    # ── Failure detection ────────────────────────────────────────────────────
    failure_events: list[dict]
    drift_score: float
    failure_count: int
    has_failures: bool

    # ── Predictive drift early warning (RFC-001) — snapshot, not accumulator ──
    drift_slope: float | None
    drift_predicted_trigger_hours: float | None
    drift_is_alarming: bool | None
    drift_trend_direction: str | None

    # ── Curation ────────────────────────────────────────────────────────────
    curated_examples: list[dict]
    curated_count: int

    # ── Data validation ──────────────────────────────────────────────────────
    pending_examples: int
    data_validated: bool

    # ── Training ─────────────────────────────────────────────────────────────
    training_triggered: bool
    training_run_id: int | None
    modal_job_id: str | None
    training_status: str
    training_submitted_at: str | None
    version_tag: str | None
    lora_weights_path: str | None
    final_loss: float | None

    # ── Evaluation ───────────────────────────────────────────────────────────
    eval_passed: bool
    eval_result: dict
    incumbent_scores: dict

    # ── Eval factory (RFC-002) — snapshot ────────────────────────────────────
    eval_set_size: int | None
    eval_factory_last_run_at: str | None
    eval_factory_examples_added_last_run: int | None

    # ── Failure attribution (RFC-003) — snapshot ─────────────────────────────
    attribution_count_this_cycle: int | None
    last_attribution_run_at: str | None

    # ── Shadow A/B ───────────────────────────────────────────────────────────
    shadow_active: bool
    shadow_ready_for_decision: bool
    ab_data: dict

    # ── Promotion / Rollback ─────────────────────────────────────────────────
    production_version: str | None
    promotion_decision: bool
    rollback_reason: str | None

    # ── Canary ───────────────────────────────────────────────────────────────
    canary_active: bool
    canary_decision: str   # "pending" | "promote" | "rollback"
    canary_metrics: dict

    # ── Audit ────────────────────────────────────────────────────────────────
    last_audit_id: int | None

    # ── Error tracking ───────────────────────────────────────────────────────
    error: str | None
    error_node: str | None

    # ── Pipeline control ─────────────────────────────────────────────────────
    paused: bool
    cycles_completed: Annotated[int, operator.add]
