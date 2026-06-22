"""Eval runner node: runs RAGAS + safety battery."""
import structlog
from src.graph.state import PipelineState
from src.config.settings import settings
from src.db.connection import get_db
from src.db.repositories.eval_runs import EvalRunRepository
from src.db.repositories.eval_set import EvalSetRepository
from src.evaluation.eval_orchestrator import EvalOrchestrator
from datetime import datetime

log = structlog.get_logger()
_orchestrator = EvalOrchestrator()

# Dev-only fallback. The promotion gate must NOT run on this in production —
# two toy examples cannot distinguish a good model from a bad one.
MOCK_EVAL_SET = [
    {"question": "What is the capital of France?", "context": "France is a country in Europe. Paris is its capital.", "ground_truth": "Paris"},
    {"question": "What year did WWII end?", "context": "World War II ended in 1945 with the surrender of Germany and Japan.", "ground_truth": "1945"},
]


def _rollback_state(state: PipelineState, reason: str) -> PipelineState:
    """Fail the eval (routes to rollback) without crashing the cycle."""
    log.error("eval_runner_blocked", reason=reason)
    return {
        **state,
        "eval_passed": False,
        "eval_result": {"fail_reasons": [reason], "blocked": True},
        "rollback_reason": reason,
    }


async def _stub_invoke(prompt: str) -> str:
    return f"This is a safe and helpful response to: {prompt}"


async def _build_invoke_fns(state: PipelineState):
    """Returns (challenger_fn, incumbent_fn, mode).

    Production (EVAL_REAL_INFERENCE): challenger = base + trained adapter merged;
    incumbent = base + the current production adapter (for same-set re-eval).
    Returns (None, ...) when real inference is required but no adapter path exists.
    Dev: returns the stub for the challenger and no incumbent fn.
    """
    if not settings.eval_real_inference:
        return _stub_invoke, None, "stub"

    lora_path = state.get("lora_weights_path")
    if not lora_path:
        return None, None, "real"

    from src.inference.challenger import build_challenger_invoke_fn, build_base_invoke_fn
    base_model = state.get("base_model") or settings.base_model_name
    challenger_fn = build_challenger_invoke_fn(base_model, lora_path)

    # Incumbent = current production model (its adapter if any, else the base).
    incumbent_fn = None
    try:
        from src.db.repositories.model_versions import ModelRepository
        async with get_db() as db:
            prod = await ModelRepository(db).get_production_version()
        prod_adapter = getattr(prod, "lora_weights_path", None) if prod else None
        incumbent_fn = (
            build_challenger_invoke_fn(base_model, prod_adapter) if prod_adapter
            else build_base_invoke_fn(base_model)
        )
    except Exception:
        log.warning("incumbent_invoke_build_failed")
    return challenger_fn, incumbent_fn, "real"


async def _verify_adapter(state, challenger_invoke, incumbent_invoke, mode) -> str | None:
    """Confirm the challenger differs from the base model (adapter applied).
    Returns a rollback reason string if it must be blocked, else None."""
    from src.monitoring.metrics import challenger_adapter_checks_total

    if mode != "real":
        # Dev stub: only enforce in production.
        if settings.environment == "production" and settings.eval_require_adapter_verification:
            challenger_adapter_checks_total.labels(result="skipped").inc()
            return ("adapter_verification_required_in_production: real inference is "
                    "off, refusing to gate on a stub model.")
        challenger_adapter_checks_total.labels(result="skipped").inc()
        return None

    try:
        from src.inference.challenger import build_base_invoke_fn, verify_adapter_distinct
        base_model = state.get("base_model") or settings.base_model_name
        base_fn = build_base_invoke_fn(base_model)
        distinct, _details = await verify_adapter_distinct(challenger_invoke, base_fn)
    except Exception:
        log.exception("adapter_verification_error")
        challenger_adapter_checks_total.labels(result="error").inc()
        if settings.environment == "production" and settings.eval_require_adapter_verification:
            return "adapter_verification_error: could not confirm the adapter was applied."
        return None

    if not distinct:
        challenger_adapter_checks_total.labels(result="identical").inc()
        return ("adapter_not_applied: challenger output is identical to the base "
                "model on all probes — the LoRA adapter was not applied or is a no-op.")
    challenger_adapter_checks_total.labels(result="verified").inc()
    return None


async def eval_runner_node(state: PipelineState) -> PipelineState:
    version_tag = state.get("version_tag", "unknown")
    lora_path = state.get("lora_weights_path")
    incumbent_scores = state.get("incumbent_scores", {})

    # Load the real, seeded eval set. Fall back to the tiny mock only in dev.
    async with get_db() as db:
        eval_set = await EvalSetRepository(db).get_eval_set()

    if len(eval_set) < settings.min_eval_examples:
        if settings.environment == "development":
            log.warning(
                "eval_set_below_minimum_using_mock",
                found=len(eval_set),
                minimum=settings.min_eval_examples,
                note="DEV ONLY — seed eval_set with scripts/seed_eval_set.py",
            )
            eval_set = MOCK_EVAL_SET
        else:
            return _rollback_state(
                state,
                f"eval_set_too_small: {len(eval_set)} < {settings.min_eval_examples} "
                f"required. Run scripts/seed_eval_set.py.",
            )

    # Build the challenger (and incumbent) invocation paths. In production this
    # loads the base model + the trained LoRA adapter (merge_and_unload); in dev
    # it falls back to a stub. Either way, the adapter-applied check below gates
    # the run so scores are never trusted from the wrong model (#T1).
    challenger_invoke, incumbent_invoke, mode = await _build_invoke_fns(state)
    if challenger_invoke is None:
        return _rollback_state(
            state,
            "real_inference_required: EVAL_REAL_INFERENCE is on but no LoRA adapter "
            "path is available for the challenger.",
        )

    # Hard gate: verify the adapter actually changed the model vs. the base.
    verify = await _verify_adapter(state, challenger_invoke, incumbent_invoke, mode)
    if verify is not None:
        return _rollback_state(state, verify)

    # Lock the eval-set snapshot (the exact ids scored) for auditability and
    # apples-to-apples incumbent comparison (#E2).
    snapshot_ids = [
        e["id"] for e in eval_set if isinstance(e, dict) and e.get("id") is not None
    ]

    async with get_db() as db:
        eval_repo = EvalRunRepository(db)
        eval_run = await eval_repo.create({
            "version_tag": version_tag,
            "eval_type": "ragas",
            "training_run_id": state.get("training_run_id"),
        })

    eval_result = await _orchestrator.run(
        version_tag=version_tag,
        challenger_invoke_fn=challenger_invoke,
        incumbent_scores=incumbent_scores or None,
        eval_set=eval_set,
        incumbent_invoke_fn=incumbent_invoke,
    )
    eval_result.rationale["eval_set_snapshot_ids"] = snapshot_ids
    eval_result.rationale["eval_set_snapshot_size"] = len(snapshot_ids)
    eval_result.rationale["inference_mode"] = mode

    async with get_db() as db:
        eval_repo = EvalRunRepository(db)
        await eval_repo.update(eval_run.id, {
            "status": "passed" if eval_result.passed else "failed",
            "faithfulness": eval_result.faithfulness,
            "answer_relevancy": eval_result.answer_relevancy,
            "context_recall": eval_result.context_recall,
            "safety_score": eval_result.safety_score,
            "gate_passed": eval_result.passed,
            "rationale": eval_result.rationale,
        })

    # Mark accessed for LRU tracking + update the eval-set-size gauge (RFC-002).
    eval_set_total = None
    try:
        from src.db.repositories.eval_set import EvalSetRepository
        from src.monitoring.metrics import eval_set_size
        ids = [e["id"] for e in eval_set if isinstance(e, dict) and e.get("id") is not None]
        async with get_db() as db:
            es_repo = EvalSetRepository(db)
            await es_repo.mark_accessed(ids)
            counts = await es_repo.count_active_by_source()
        for source, count in counts.items():
            eval_set_size.labels(source=source).set(count)
        eval_set_total = sum(counts.values())
    except Exception:
        log.warning("eval_set_mark_accessed_failed")

    log.info("eval_runner_node_complete", version=version_tag, passed=eval_result.passed)
    return {
        **state,
        "eval_passed": eval_result.passed,
        "eval_result": eval_result.rationale,
        "eval_set_size": eval_set_total,
    }
