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

    # Mock invoke fn — in production loads the actual LoRA adapter
    async def challenger_invoke(prompt: str) -> str:
        return f"This is a safe and helpful response to: {prompt}"

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
    )

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
