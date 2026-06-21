"""
Promotion gate: all conditions must pass for the challenger to be promoted.

Gates:
  1. A/B window complete (≥1000 requests, ≥48h elapsed)
  2. Statistical significance (Welch p < 0.05, Cohen's d ≥ 0.10)
  3. Quality improvement ≥ threshold (3% absolute)
  4. Safety score = 1.0 (no regressions)
"""

import structlog
from dataclasses import dataclass, field

from src.evaluation.statistical_tests import passes_significance_gate_from_deltas
from src.config.settings import settings

log = structlog.get_logger()


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    metrics: dict = field(default_factory=dict)


class PromotionGate:
    def evaluate(
        self,
        ab_data: dict,
        eval_result,
        incumbent_scores: dict,
    ) -> PromotionDecision:
        """
        ab_data: output of ABCollector.collect_window()
        eval_result: EvalResult from EvalOrchestrator
        incumbent_scores: ragas scores for the current production model
        """
        metrics: dict = {}

        # Gate 1: safety must be perfect
        if eval_result.safety_score < 1.0:
            return PromotionDecision(
                promote=False,
                reason=f"safety_regression: score={eval_result.safety_score:.2f}",
                metrics={"safety_score": eval_result.safety_score},
            )

        # Gate 2: A/B window must be complete
        if not ab_data.get("ready"):
            return PromotionDecision(
                promote=False,
                reason=(
                    f"ab_window_incomplete: "
                    f"requests={ab_data.get('n_requests', 0)}, "
                    f"hours={ab_data.get('elapsed_hours', 0):.1f}"
                ),
                metrics=ab_data,
            )

        # Gate 3: statistical significance (paired one-sample test on per-request
        # deltas; each delta is challenger_score - production_score).
        sig_passes, sig_metrics = passes_significance_gate_from_deltas(
            quality_deltas=ab_data["quality_deltas"],
            n_requests=ab_data["n_requests"],
        )
        metrics.update(sig_metrics)

        if not sig_passes:
            return PromotionDecision(
                promote=False,
                reason=f"significance_gate_failed: {sig_metrics.get('fail_reason', '')}",
                metrics=metrics,
            )

        # Gate 4: absolute improvement threshold
        avg_challenger = (
            eval_result.faithfulness +
            eval_result.answer_relevancy +
            eval_result.context_recall
        ) / 3

        avg_incumbent = (
            incumbent_scores.get("faithfulness", 0) +
            incumbent_scores.get("answer_relevancy", 0) +
            incumbent_scores.get("context_recall", 0)
        ) / 3

        delta = avg_challenger - avg_incumbent
        metrics["ragas_delta"] = round(delta, 4)

        if delta < settings.eval_improvement_threshold:
            return PromotionDecision(
                promote=False,
                reason=(
                    f"improvement_threshold_not_met: "
                    f"delta={delta:.4f} < {settings.eval_improvement_threshold}"
                ),
                metrics=metrics,
            )

        log.info("promotion_gate_passed", **metrics)
        return PromotionDecision(
            promote=True,
            reason="all_gates_passed",
            metrics=metrics,
        )
