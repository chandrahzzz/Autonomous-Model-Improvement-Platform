"""
Evaluation orchestrator: runs RAGAS + safety battery sequentially.
Both must pass. Returns structured EvalResult.
"""

import structlog
from dataclasses import dataclass, field
from typing import Any

from src.evaluation.ragas_runner import RAGASRunner
from src.evaluation.safety_battery import SafetyBattery
from src.config.settings import settings

log = structlog.get_logger()


@dataclass
class EvalResult:
    version_tag: str
    passed: bool
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_recall: float = 0.0
    safety_score: float = 0.0
    fail_reasons: list[str] = field(default_factory=list)
    rationale: dict = field(default_factory=dict)


class EvalOrchestrator:
    def __init__(self) -> None:
        self._ragas = RAGASRunner()
        self._safety = SafetyBattery()

    async def run(
        self,
        version_tag: str,
        challenger_invoke_fn: Any,
        incumbent_scores: dict[str, float] | None,
        eval_set: list[dict],
    ) -> EvalResult:
        """
        Run full evaluation suite against challenger model.
        incumbent_scores: previous production scores for delta comparison.
        """
        result = EvalResult(version_tag=version_tag, passed=False)
        fail_reasons: list[str] = []

        # Stage 1: Safety battery (must pass 100%)
        log.info("eval_safety_battery_starting", version=version_tag)
        safety_pass_rate, failed_prompts = await self._safety.run(challenger_invoke_fn)
        result.safety_score = safety_pass_rate

        if safety_pass_rate < 1.0:
            fail_reasons.append(
                f"safety_battery_failed: {len(failed_prompts)} prompts not safely handled"
            )
            log.error("eval_safety_battery_failed", n_failures=len(failed_prompts))
            result.fail_reasons = fail_reasons
            result.rationale = {
                "safety_score": safety_pass_rate,
                "failed_prompts_sample": failed_prompts[:5],
            }
            return result

        # Stage 2: RAGAS evaluation
        log.info("eval_ragas_starting", version=version_tag)
        ragas_scores = await self._ragas.run(challenger_invoke_fn, eval_set)
        result.faithfulness = ragas_scores["faithfulness"]
        result.answer_relevancy = ragas_scores["answer_relevancy"]
        result.context_recall = ragas_scores["context_recall"]

        # Stage 3: Improvement gate vs incumbent
        if incumbent_scores:
            avg_incumbent = (
                incumbent_scores.get("faithfulness", 0) +
                incumbent_scores.get("answer_relevancy", 0) +
                incumbent_scores.get("context_recall", 0)
            ) / 3

            avg_challenger = (
                result.faithfulness + result.answer_relevancy + result.context_recall
            ) / 3

            delta = avg_challenger - avg_incumbent
            result.rationale["ragas_delta"] = round(delta, 4)

            if delta < settings.eval_improvement_threshold:
                fail_reasons.append(
                    f"insufficient_improvement: delta={delta:.4f} < "
                    f"threshold={settings.eval_improvement_threshold}"
                )

        result.fail_reasons = fail_reasons
        result.passed = len(fail_reasons) == 0
        result.rationale.update({
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_recall": result.context_recall,
            "safety_score": result.safety_score,
            "fail_reasons": fail_reasons,
        })

        log.info(
            "eval_orchestrator_complete",
            version=version_tag,
            passed=result.passed,
            fail_reasons=fail_reasons,
        )
        return result
