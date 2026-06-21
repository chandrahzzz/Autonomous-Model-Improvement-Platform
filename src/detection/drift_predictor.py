"""
Predictive drift early warning (RFC-001).

Fits a linear regression on the DriftDetector's rolling window of Mahalanobis
distances and extrapolates how many hours until the drift threshold is crossed —
so the pipeline can warn *before* it reactively fires `is_drifting()`.

Purely observational: it never changes failure classification, training triggers,
or any other pipeline behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import structlog

from src.config.settings import settings
from src.monitoring.metrics import drift_early_warnings_total

if TYPE_CHECKING:  # avoid importing drift.py (and its heavy deps) at runtime
    from src.detection.drift import DriftDetector

log = structlog.get_logger()

# The detection loop runs on the "monitoring" cadence in runner.py (60s/cycle).
CYCLE_SECONDS = 60


@dataclass(frozen=True)
class DriftTrend:
    current_score: float
    threshold: float
    slope_per_cycle: float
    r_squared: float
    predicted_trigger_hours: float | None
    window_size: int
    trend_direction: str        # "stable" | "increasing" | "decreasing"
    is_alarming: bool


class DriftPredictor:
    def __init__(self, drift_detector: "DriftDetector") -> None:
        # Store reference to the EXISTING DriftDetector instance — never create one.
        self._detector = drift_detector
        self._last_alert_at: datetime | None = None
        self._cycles_since_last_prediction: int = 0

    def compute_trend(self) -> DriftTrend | None:
        """Synchronous (numpy) — fast. Returns None if the window is too small."""
        values = list(self._detector._window)  # raw per-request Mahalanobis distances
        if len(values) < settings.drift_min_window_for_prediction:
            return None

        x = np.arange(len(values), dtype=float)
        y = np.array(values, dtype=float)

        coeffs = np.polyfit(x, y, deg=1)
        slope = float(coeffs[0])  # positive = worsening

        y_hat = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        current_score = float(np.mean(y[-20:]))  # mean of most-recent points
        threshold = settings.drift_mahalanobis_threshold

        if slope > 1e-5:
            trend_direction = "increasing"
        elif slope < -1e-5:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"

        if slope > 0 and current_score < threshold:
            cycles_to_trigger = (threshold - current_score) / slope
            hours_to_trigger = cycles_to_trigger * (CYCLE_SECONDS / 3600.0)
            is_alarming = hours_to_trigger <= settings.drift_alert_horizon_hours
        else:
            hours_to_trigger = None
            is_alarming = False

        return DriftTrend(
            current_score=current_score,
            threshold=threshold,
            slope_per_cycle=slope,
            r_squared=r_squared,
            predicted_trigger_hours=hours_to_trigger,
            window_size=len(values),
            trend_direction=trend_direction,
            is_alarming=is_alarming,
        )

    async def maybe_send_alert(self, trend: DriftTrend, alerter) -> bool:
        """Fire a predictive PagerDuty warning, deduplicated within a window."""
        if not trend.is_alarming:
            return False

        now = datetime.now(timezone.utc)
        if self._last_alert_at is not None:
            elapsed = (now - self._last_alert_at).total_seconds()
            if elapsed < settings.drift_alert_dedupe_seconds:
                return False  # deduplication — don't spam

        hours = trend.predicted_trigger_hours
        title = f"Drift Early Warning — Trigger in ~{hours:.1f}h"
        body = (
            f"Current score: {trend.current_score:.4f} / threshold: {trend.threshold}\n"
            f"Trend: slope={trend.slope_per_cycle:+.5f}/cycle "
            f"(R²={trend.r_squared:.2f}, window={trend.window_size})\n"
            f"Predicted trigger: ~{hours:.1f} hours from now\n"
            f"This is a predictive alert. No action required yet."
        )
        # Existing alerter API (Phase 0, Q4): alerter.trigger(summary, severity, details).
        await alerter.trigger(summary=title, severity="warning", details={"body": body})

        self._last_alert_at = now
        drift_early_warnings_total.inc()
        return True

    def should_compute_this_cycle(self) -> bool:
        """Throttle: only run the regression every N cycles."""
        self._cycles_since_last_prediction += 1
        if self._cycles_since_last_prediction >= settings.drift_prediction_interval_cycles:
            self._cycles_since_last_prediction = 0
            return True
        return False


_predictor_instance: DriftPredictor | None = None


def get_drift_predictor(drift_detector: "DriftDetector | None") -> DriftPredictor:
    """Module-level singleton. `drift_detector` is used only on first call."""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = DriftPredictor(drift_detector)
    return _predictor_instance
