"""
Confidence / threshold calibration.

Detection thresholds (hallucination, drift, refusal, format) are otherwise
static and arbitrary. This calibrator derives a *false-positive proxy* from data
the pipeline already produces and suggests threshold adjustments:

  drop_rate = 1 - (curated examples / detected failures), per failure type.

A high drop rate means the detector flagged many "failures" that curation then
discarded (low teacher confidence / quality) — i.e. the detector is over-firing,
so its threshold should be made *less* sensitive. A very low drop rate suggests
the detector may be too lax.

Suggestions are written to `calibration_history` and logged. They are NOT
auto-applied unless ALLOW_AUTO_CALIBRATION is set — and even then, applying a new
threshold means updating configuration, so this stays advisory by design.
"""

from datetime import datetime, timedelta
from typing import Any

import structlog
from sqlalchemy import text

from src.config.settings import settings

log = structlog.get_logger()

# failure_type -> (settings attribute, min bound, max bound)
_CALIBRATABLE = {
    "hallucination": ("hallucination_threshold", 0.2, 0.9),
    "semantic_drift": ("drift_mahalanobis_threshold", 0.05, 0.5),
    "format_regression": ("format_kl_threshold", 0.1, 1.5),
    "refusal_creep": ("refusal_rate_multiplier", 1.2, 5.0),
}


class ThresholdCalibrator:
    async def run(self, db: Any, window_days: int = 7) -> list[dict]:
        since = datetime.utcnow() - timedelta(days=window_days)
        suggestions: list[dict] = []

        for failure_type, (attr, lo, hi) in _CALIBRATABLE.items():
            detected = int((await db.execute(
                text(
                    "SELECT COUNT(*) FROM failure_classifications "
                    "WHERE created_at >= :s AND failure_type = :t"
                ),
                {"s": since, "t": failure_type},
            )).scalar() or 0)
            if detected < settings.calibration_min_samples:
                continue
            curated = int((await db.execute(
                text(
                    "SELECT COUNT(*) FROM training_examples "
                    "WHERE created_at >= :s AND failure_type = :t"
                ),
                {"s": since, "t": failure_type},
            )).scalar() or 0)

            drop_rate = 1.0 - min(1.0, curated / detected)
            current = float(getattr(settings, attr))
            step = settings.calibration_step

            if drop_rate > settings.calibration_fp_target:
                suggested = min(hi, round(current + step, 4))  # less sensitive
                signal = f"drop_rate={drop_rate:.2f} > target — over-firing"
            elif drop_rate < settings.calibration_fp_target / 2:
                suggested = max(lo, round(current - step, 4))  # more sensitive
                signal = f"drop_rate={drop_rate:.2f} low — may be too lax"
            else:
                continue

            if suggested == current:
                continue

            await db.execute(
                text(
                    "INSERT INTO calibration_history "
                    "(metric, current_value, suggested_value, signal, applied) "
                    "VALUES (:m, :c, :s, :sig, :ap)"
                ),
                {
                    "m": attr, "c": current, "s": suggested,
                    "sig": signal, "ap": settings.allow_auto_calibration,
                },
            )
            suggestions.append({
                "metric": attr, "current": current, "suggested": suggested,
                "signal": signal, "drop_rate": round(drop_rate, 3),
            })
            log.warning(
                "threshold_calibration_suggestion",
                metric=attr, current=current, suggested=suggested, signal=signal,
            )

        log.info("threshold_calibration_complete", suggestions=len(suggestions))
        return suggestions
