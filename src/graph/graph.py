"""
LangGraph pipeline graph construction and compilation.

The graph runs forever in production. Every node is async.
Checkpoints are persisted to PostgreSQL so the pipeline can
resume after crashes without losing state.
"""

import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.graph.state import PipelineState
from src.graph.nodes.log_monitor import log_monitor_node
from src.graph.nodes.failure_detector import failure_detector_node
from src.graph.nodes.example_curator import example_curator_node
from src.graph.nodes.data_validator import data_validator_node
from src.graph.nodes.fine_tune_trigger import fine_tune_trigger_node
from src.graph.nodes.lora_trainer import lora_trainer_node
from src.graph.nodes.training_poller import training_poller_node
from src.graph.nodes.eval_runner import eval_runner_node
from src.graph.nodes.ab_test_node import ab_test_node
from src.graph.nodes.promotion_decider import promotion_decider_node
from src.graph.nodes.rollback_node import rollback_node
from src.graph.nodes.audit_logger import audit_logger_node
from src.graph.edges import (
    after_failure_detector,
    after_data_validator,
    after_fine_tune_trigger,
    after_training_poller,
    after_eval_runner,
    after_ab_test,
    after_promotion_decider,
    after_canary,
)

log = structlog.get_logger()


async def promote_model_node(state: PipelineState) -> PipelineState:
    """Executes the actual DB promotion after audit is written."""
    from src.db.connection import get_db
    from src.db.repositories.model_versions import ModelRepository
    from src.monitoring.metrics import promotions_total

    version_tag = state.get("version_tag", "unknown")
    async with get_db() as db:
        repo = ModelRepository(db)
        await repo.promote(version_tag)
        prod = await repo.get_production_version()

    promotions_total.inc()
    log.info("model_promoted", version_tag=version_tag)

    # Refresh the drift AND format baselines against the newly promoted model so
    # neither is measured against a stale seed. Failure here must NOT undo the
    # promotion — log loudly and continue with the old baselines.
    try:
        from src.graph.nodes.failure_detector import _drift, _fmt
        from src.config.settings import settings as _settings
        from src.db.repositories.llm_logs import LLMLogRepository
        async with get_db() as db:
            await _drift.refresh_baseline(
                db, model_version=version_tag,
                min_samples=_settings.drift_baseline_min_samples,
            )
            # Format length baseline (#5): recompute from the new model's outputs.
            completions = await LLMLogRepository(db).get_recent_completions(
                limit=10000, model_version=version_tag
            )
            if len(completions) < _settings.format_min_samples:
                completions = await LLMLogRepository(db).get_recent_completions(limit=10000)
            _fmt.refresh_baseline(completions)
    except Exception:
        log.error("baseline_refresh_failed", version_tag=version_tag)

    return {
        **state,
        "production_version": version_tag,
        "training_triggered": False,
        "modal_job_id": None,
        "training_status": "idle",
        "shadow_active": False,
        "shadow_ready_for_decision": False,
        "canary_active": False,
        "version_tag": None,
        "rollback_reason": None,
        # Clear transient error markers so they don't leak into the next cycle (#L2).
        "error": None,
        "error_node": None,
        "cycles_completed": 1,
    }


async def audit_pre_train_node(state: PipelineState) -> PipelineState:
    """Audit entry before training starts."""
    from src.db.connection import get_db
    from src.audit.logger import AuditLogger
    from src.audit.schemas import AuditEvent
    event = AuditEvent(
        event_type="training_triggered",
        decision=f"start training for {state.get('version_tag', 'next')}",
        rationale={"pending_examples": state.get("pending_examples"), "drift": state.get("drift_score")},
        state_snapshot={"pending": state.get("pending_examples"), "drift": state.get("drift_score")},
    )
    async with get_db() as db:
        audit = AuditLogger(db)
        row_id = await audit.log(event)
    return {**state, "last_audit_id": row_id}


async def audit_rollback_node(state: PipelineState) -> PipelineState:
    """Audit entry before rollback."""
    from src.db.connection import get_db
    from src.audit.logger import AuditLogger
    from src.audit.schemas import AuditEvent
    event = AuditEvent(
        event_type="model_rolled_back",
        decision=f"rollback {state.get('version_tag')}: {state.get('rollback_reason')}",
        rationale=state.get("eval_result") or {},
        state_snapshot={"version": state.get("version_tag"), "reason": state.get("rollback_reason")},
        model_version_before=state.get("version_tag"),
        model_version_after=state.get("production_version"),
    )
    async with get_db() as db:
        audit = AuditLogger(db)
        row_id = await audit.log(event)
    return {**state, "last_audit_id": row_id}


async def audit_promote_node(state: PipelineState) -> PipelineState:
    """Audit entry before promotion."""
    from src.db.connection import get_db
    from src.audit.logger import AuditLogger
    from src.audit.schemas import AuditEvent
    event = AuditEvent(
        event_type="model_promoted",
        decision=f"promote {state.get('version_tag')} to production",
        rationale=state.get("eval_result") or {},
        state_snapshot={"version": state.get("version_tag"), "ab": state.get("ab_data", {})},
        model_version_before=state.get("production_version"),
        model_version_after=state.get("version_tag"),
    )
    async with get_db() as db:
        audit = AuditLogger(db)
        row_id = await audit.log(event)
    return {**state, "last_audit_id": row_id}


async def canary_node(state: PipelineState) -> PipelineState:
    """Gate full promotion behind a small live canary rollout.

    Disabled by default (no serving layer to feed it) — when off it passes
    straight through. When on, it starts the canary, waits across cycles, and
    decides promote/rollback from the serving layer's recorded results.
    """
    from src.config.settings import settings as _settings
    if not _settings.canary_enabled:
        return {**state, "canary_active": False, "canary_decision": "promote"}

    from src.shadow.canary import CanaryController
    cc = CanaryController()
    version_tag = state.get("version_tag", "unknown")
    active = await cc.get_active()
    if not active or active.get("version") != version_tag:
        await cc.start(version_tag)
        log.info("canary_node_started", version=version_tag)
        return {**state, "canary_active": True, "canary_decision": "pending"}

    decision, metrics = await cc.evaluate()
    if decision in ("promote", "rollback"):
        await cc.clear()
        log.info("canary_node_decided", version=version_tag, decision=decision, **metrics)
        return {
            **state,
            "canary_active": False,
            "canary_decision": decision,
            "canary_metrics": metrics,
            "rollback_reason": None if decision == "promote"
            else f"canary_failed: {metrics.get('reason', 'unknown')}",
        }
    return {**state, "canary_active": True, "canary_decision": "pending", "canary_metrics": metrics}


def route_cycle_start(state: PipelineState) -> str:
    """Pick the entry node for this cycle.

    The previous cycle may have ended at END while a Modal training job was
    still running, a shadow A/B window was open, or a canary was in flight.
    Because a cycle always re-invokes the graph from its entry point, a fixed
    `log_monitor` entry could never get back to those nodes, so in-flight work
    was never observed to completion. Resume it here before starting fresh work.
    """
    if state.get("modal_job_id") and state.get("training_status") in ("submitted", "running"):
        return "training_poller"
    if (
        state.get("shadow_active")
        and not state.get("shadow_ready_for_decision")
        and state.get("version_tag")
    ):
        return "ab_test_node"
    if state.get("canary_active") and state.get("canary_decision") == "pending":
        return "canary_node"
    return "log_monitor"


def build_graph() -> StateGraph:
    """Construct and compile the perpetual LangGraph state machine."""
    builder = StateGraph(PipelineState)

    # Add all nodes
    builder.add_node("log_monitor", log_monitor_node)
    builder.add_node("failure_detector", failure_detector_node)
    builder.add_node("example_curator", example_curator_node)
    builder.add_node("data_validator", data_validator_node)
    builder.add_node("fine_tune_trigger", fine_tune_trigger_node)
    builder.add_node("audit_logger_pre_train", audit_pre_train_node)
    builder.add_node("lora_trainer", lora_trainer_node)
    builder.add_node("training_poller", training_poller_node)
    builder.add_node("eval_runner", eval_runner_node)
    builder.add_node("ab_test_node", ab_test_node)
    builder.add_node("promotion_decider", promotion_decider_node)
    builder.add_node("canary_node", canary_node)
    builder.add_node("audit_logger_promote", audit_promote_node)
    builder.add_node("promote_model", promote_model_node)
    builder.add_node("audit_logger_rollback", audit_rollback_node)
    builder.add_node("rollback_node", rollback_node)
    builder.add_node("audit_logger", audit_logger_node)

    # Conditional entry point: resume an in-flight training poll or shadow test
    # if one is active, otherwise begin a fresh monitoring cycle at log_monitor.
    builder.set_conditional_entry_point(
        route_cycle_start,
        {
            "training_poller": "training_poller",
            "ab_test_node": "ab_test_node",
            "canary_node": "canary_node",
            "log_monitor": "log_monitor",
        },
    )

    # Linear edges
    builder.add_edge("log_monitor", "failure_detector")
    builder.add_edge("example_curator", "data_validator")
    builder.add_edge("audit_logger_pre_train", "lora_trainer")
    builder.add_edge("lora_trainer", "training_poller")
    builder.add_edge("audit_logger_promote", "promote_model")
    builder.add_edge("promote_model", END)
    builder.add_edge("audit_logger_rollback", "rollback_node")
    builder.add_edge("rollback_node", END)

    # Conditional edges
    builder.add_conditional_edges(
        "failure_detector",
        after_failure_detector,
        {"example_curator": "example_curator", "data_validator": "data_validator"},
    )
    builder.add_conditional_edges(
        "data_validator",
        after_data_validator,
        {"fine_tune_trigger": "fine_tune_trigger", END: END},
    )
    builder.add_conditional_edges(
        "fine_tune_trigger",
        after_fine_tune_trigger,
        {"audit_logger_pre_train": "audit_logger_pre_train", END: END},
    )
    builder.add_conditional_edges(
        "training_poller",
        after_training_poller,
        {
            "eval_runner": "eval_runner",
            "rollback_node": "rollback_node",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "eval_runner",
        after_eval_runner,
        {"ab_test_node": "ab_test_node", "audit_logger_rollback": "audit_logger_rollback"},
    )
    builder.add_conditional_edges(
        "ab_test_node",
        after_ab_test,
        {"promotion_decider": "promotion_decider", END: END},
    )
    builder.add_conditional_edges(
        "promotion_decider",
        after_promotion_decider,
        {"canary_node": "canary_node", "audit_logger_rollback": "audit_logger_rollback"},
    )
    builder.add_conditional_edges(
        "canary_node",
        after_canary,
        {
            "audit_logger_promote": "audit_logger_promote",
            "audit_logger_rollback": "audit_logger_rollback",
            END: END,
        },
    )

    return builder


def build_checkpointer():
    """Return the configured LangGraph checkpointer (#L1).

    `CHECKPOINTER_BACKEND=postgres` gives crash-resilient resume (the checkpoint
    that records which node to resume from survives a restart), but requires the
    `langgraph-checkpoint-postgres` extra and a one-time `.setup()`. When it's
    unavailable we fall back to MemorySaver and warn loudly — the Redis
    `pipeline:full_state` snapshot + conditional entry point still recover
    in-flight Modal jobs / shadow windows, so this is degraded, not broken.
    """
    from src.config.settings import settings
    from src.monitoring.metrics import checkpointer_backend_info

    if settings.checkpointer_backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
            conn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)
            saver_cm = PostgresSaver.from_conn_string(conn)
            saver = saver_cm.__enter__()  # lifetime tied to the process
            try:
                saver.setup()  # idempotent: creates checkpoint tables if absent
            except Exception:
                log.warning("postgres_checkpointer_setup_skipped")
            checkpointer_backend_info.labels(backend="postgres").set(1)
            log.info("checkpointer_backend_selected", backend="postgres")
            return saver
        except Exception as e:
            log.error(
                "postgres_checkpointer_unavailable_falling_back_to_memory",
                error=str(e),
                hint="pip install langgraph-checkpoint-postgres to enable durable checkpoints",
            )

    checkpointer_backend_info.labels(backend="memory").set(1)
    return MemorySaver()


def compile_graph(checkpointer=None):
    """Compile the graph. Pass a checkpointer to inject one (tests/custom); by
    default the backend is chosen from settings (#L1)."""
    builder = build_graph()
    if checkpointer is None:
        checkpointer = build_checkpointer()
    return builder.compile(checkpointer=checkpointer)
