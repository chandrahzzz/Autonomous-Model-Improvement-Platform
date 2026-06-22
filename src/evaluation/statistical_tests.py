"""
Statistical significance tests for A/B comparison.

Uses Welch's t-test (unequal variance) and Cohen's d effect size.
Both must pass their thresholds before promotion is allowed.
"""

import math
import numpy as np
from scipy import stats
import structlog

from src.config.settings import settings

log = structlog.get_logger()


def welch_t_test(
    production_scores: list[float],
    challenger_scores: list[float],
) -> tuple[float, float]:
    """Returns (t_statistic, p_value) for a one-sided Welch's t-test
    (H1: challenger > production).

    Guards the degenerate zero-variance case: when both samples are constant,
    SciPy returns NaN (and `nan >= threshold` is False, which would silently
    let a non-improvement slip through the p-value gate). We instead treat a
    strictly higher constant challenger as significant and an equal/lower one
    as not significant.
    """
    if len(production_scores) < 2 or len(challenger_scores) < 2:
        return 0.0, 1.0

    prod = np.asarray(production_scores, dtype=float)
    chal = np.asarray(challenger_scores, dtype=float)

    if np.std(prod) == 0 and np.std(chal) == 0:
        if chal.mean() > prod.mean():
            return float("inf"), 0.0
        return 0.0, 1.0

    t_stat, p_val = stats.ttest_ind(
        chal,
        prod,
        equal_var=False,  # Welch's
        alternative="greater",  # challenger > production
    )
    if math.isnan(p_val):
        return 0.0, 1.0
    return float(t_stat), float(p_val)


def cohens_d(
    production_scores: list[float],
    challenger_scores: list[float],
) -> float:
    """Cohen's d effect size. Positive = challenger is better.

    When neither group has within-group variance the standardized effect size
    is undefined; we report it as infinite when the means differ (so the effect
    is unambiguous) and 0.0 when the means are equal.
    """
    prod = np.array(production_scores, dtype=float)
    chal = np.array(challenger_scores, dtype=float)
    mean_diff = float(np.mean(chal) - np.mean(prod))
    pooled_std = math.sqrt(
        (np.std(prod, ddof=1) ** 2 + np.std(chal, ddof=1) ** 2) / 2
    )
    if pooled_std == 0:
        if mean_diff == 0:
            return 0.0
        return math.inf if mean_diff > 0 else -math.inf
    return mean_diff / pooled_std


def _round_safe(x: float, ndigits: int = 4) -> float:
    """Round, but pass non-finite values through unchanged (round() handles
    inf/nan, but this keeps intent explicit for metrics dicts)."""
    return round(x, ndigits) if math.isfinite(x) else x


def passes_significance_gate(
    production_scores: list[float],
    challenger_scores: list[float],
    n_requests: int,
) -> tuple[bool, dict]:
    """
    Returns (passes, metrics_dict).
    Requires:
      - n_requests ≥ ab_min_requests
      - p_value < ab_pvalue_threshold
      - cohen's d ≥ ab_cohens_d_threshold
    """
    metrics: dict = {
        "n_requests": n_requests,
        "n_production": len(production_scores),
        "n_challenger": len(challenger_scores),
    }

    if n_requests < settings.ab_min_requests:
        metrics["fail_reason"] = "insufficient_requests"
        return False, metrics

    if not production_scores or not challenger_scores:
        metrics["fail_reason"] = "empty_score_arrays"
        return False, metrics

    quality_delta = float(np.mean(challenger_scores) - np.mean(production_scores))

    # Explicit regression guard (#E4): block a clearly-worse challenger before the
    # one-sided significance test (which would just report "not significant").
    if quality_delta < 0:
        from src.monitoring.metrics import challenger_regression_blocks_total
        challenger_regression_blocks_total.inc()
        metrics.update({"quality_delta": _round_safe(quality_delta), "fail_reason": "challenger_regression"})
        return False, metrics

    t_stat, p_val = welch_t_test(production_scores, challenger_scores)
    d = cohens_d(production_scores, challenger_scores)

    metrics.update({
        "t_statistic": _round_safe(t_stat),
        "p_value": _round_safe(p_val),
        "cohens_d": _round_safe(d),
        "quality_delta": _round_safe(quality_delta),
    })

    if p_val >= settings.ab_pvalue_threshold:
        metrics["fail_reason"] = f"p_value {p_val:.4f} >= {settings.ab_pvalue_threshold}"
        return False, metrics

    if d < settings.ab_cohens_d_threshold:
        metrics["fail_reason"] = f"cohens_d {d} < {settings.ab_cohens_d_threshold}"
        return False, metrics

    log.info("ab_significance_passed", **metrics)
    return True, metrics


def passes_significance_gate_from_deltas(
    quality_deltas: list[float],
    n_requests: int,
) -> tuple[bool, dict]:
    """Significance gate for *paired* A/B data.

    Each element of `quality_deltas` is (challenger_score - production_score) for
    one shadow request, so the correct test is a one-sample t-test of the deltas
    against 0 (H1: mean delta > 0), with a one-sample standardized effect size
    (mean / std). This replaces the previous approach of running a two-sample
    test against a synthetic all-zeros array, which inflated Cohen's d by ~√2
    (the zero array contributed zero variance to the pooled denominator).
    """
    metrics: dict = {"n_requests": n_requests, "n_deltas": len(quality_deltas)}

    if n_requests < settings.ab_min_requests:
        metrics["fail_reason"] = "insufficient_requests"
        return False, metrics

    if len(quality_deltas) < 2:
        metrics["fail_reason"] = "insufficient_samples"
        return False, metrics

    deltas = np.asarray(quality_deltas, dtype=float)
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std(ddof=1))

    # Explicit regression guard (#E4): a one-sided test (H1: challenger > prod)
    # gives a clearly-worse challenger a high p-value (not significant), so without
    # this it could only be caught downstream. Block immediately and unambiguously
    # when the mean delta is negative — a worse model must never reach promotion.
    if mean_delta < 0:
        from src.monitoring.metrics import challenger_regression_blocks_total
        challenger_regression_blocks_total.inc()
        metrics.update({"quality_delta": _round_safe(mean_delta), "fail_reason": "challenger_regression"})
        return False, metrics

    if std_delta == 0:
        # Constant deltas: significant iff strictly positive.
        t_stat, p_val = (math.inf, 0.0) if mean_delta > 0 else (0.0, 1.0)
        d = math.inf if mean_delta > 0 else (0.0 if mean_delta == 0 else -math.inf)
    else:
        t_stat, p_val = stats.ttest_1samp(deltas, 0.0, alternative="greater")
        t_stat, p_val = float(t_stat), float(p_val)
        if math.isnan(p_val):
            p_val = 1.0
        d = mean_delta / std_delta

    metrics.update({
        "t_statistic": _round_safe(t_stat),
        "p_value": _round_safe(p_val),
        "cohens_d": _round_safe(d),
        "quality_delta": _round_safe(mean_delta),
    })

    if p_val >= settings.ab_pvalue_threshold:
        metrics["fail_reason"] = f"p_value {p_val:.4f} >= {settings.ab_pvalue_threshold}"
        return False, metrics

    if d < settings.ab_cohens_d_threshold:
        metrics["fail_reason"] = f"cohens_d {d} < {settings.ab_cohens_d_threshold}"
        return False, metrics

    log.info("ab_significance_passed", **metrics)
    return True, metrics
