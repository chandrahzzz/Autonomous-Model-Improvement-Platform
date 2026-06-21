"""Unit tests for the predictive drift early-warning system (RFC-001).

No real I/O — DriftDetector / alerter / settings are mocked. Imports only from
src.detection.drift_predictor + stdlib + pytest/mock/numpy.
"""

from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.detection.drift_predictor import DriftPredictor, DriftTrend


def make_detector_with_window(values: list[float]) -> MagicMock:
    detector = MagicMock()
    detector._window = deque(values, maxlen=1000)  # Phase 0, Q1: attribute is _window
    return detector


def make_predictor(window_values: list[float]) -> DriftPredictor:
    return DriftPredictor(drift_detector=make_detector_with_window(window_values))


@contextmanager
def patch_settings(threshold=0.15, horizon_hours=24.0, min_window=50):
    with patch("src.detection.drift_predictor.settings") as s:
        s.drift_mahalanobis_threshold = threshold   # actual setting name
        s.drift_alert_horizon_hours = horizon_hours
        s.drift_min_window_for_prediction = min_window
        s.drift_alert_dedupe_seconds = 7200
        yield s


# ── compute_trend() ──────────────────────────────────────────────────────────

def test_compute_trend_returns_none_when_window_too_small():
    predictor = make_predictor([0.05] * 10)  # 10 < min_window (50)
    with patch_settings():
        assert predictor.compute_trend() is None


def test_compute_trend_flat_window_is_stable():
    predictor = make_predictor([0.08] * 200)
    with patch_settings():
        trend = predictor.compute_trend()
    assert trend is not None
    assert trend.trend_direction == "stable"
    assert trend.is_alarming is False
    assert trend.predicted_trigger_hours is None
    assert abs(trend.slope_per_cycle) < 1e-5


def test_compute_trend_upward_trend_is_alarming():
    predictor = make_predictor(list(np.linspace(0.05, 0.13, 200)))
    with patch_settings(threshold=0.15, horizon_hours=24.0):
        trend = predictor.compute_trend()
    assert trend.trend_direction == "increasing"
    assert trend.slope_per_cycle > 0
    assert trend.predicted_trigger_hours is not None
    assert trend.predicted_trigger_hours > 0
    assert trend.is_alarming is True


def test_compute_trend_downward_trend_not_alarming():
    predictor = make_predictor(list(np.linspace(0.12, 0.04, 200)))
    with patch_settings():
        trend = predictor.compute_trend()
    assert trend.trend_direction == "decreasing"
    assert trend.is_alarming is False
    assert trend.predicted_trigger_hours is None


def test_compute_trend_already_above_threshold_no_prediction():
    predictor = make_predictor([0.20] * 200)
    with patch_settings(threshold=0.15):
        trend = predictor.compute_trend()
    assert trend.predicted_trigger_hours is None


def test_compute_trend_slow_trend_outside_horizon_not_alarming():
    predictor = make_predictor(list(np.linspace(0.05, 0.051, 200)))
    with patch_settings(threshold=0.15, horizon_hours=24.0):
        trend = predictor.compute_trend()
    if trend.predicted_trigger_hours is not None:
        assert trend.predicted_trigger_hours > 24.0
        assert trend.is_alarming is False


def test_compute_trend_r_squared_in_range():
    values = list(np.linspace(0.05, 0.12, 150)) + [0.09, 0.13, 0.08]
    predictor = make_predictor(values)
    with patch_settings():
        trend = predictor.compute_trend()
    assert 0.0 <= trend.r_squared <= 1.0


# ── maybe_send_alert() ───────────────────────────────────────────────────────

def _alarming_trend() -> DriftTrend:
    return DriftTrend(
        current_score=0.13, threshold=0.15, slope_per_cycle=0.001,
        r_squared=0.9, predicted_trigger_hours=5.5, window_size=200,
        trend_direction="increasing", is_alarming=True,
    )


@pytest.mark.asyncio
async def test_maybe_send_alert_non_alarming_returns_false():
    predictor = make_predictor([0.08] * 200)
    trend = DriftTrend(
        current_score=0.08, threshold=0.15, slope_per_cycle=0.0,
        r_squared=0.5, predicted_trigger_hours=None, window_size=200,
        trend_direction="stable", is_alarming=False,
    )
    alerter = AsyncMock()
    result = await predictor.maybe_send_alert(trend, alerter)
    assert result is False
    alerter.trigger.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_send_alert_alarming_sends_alert():
    predictor = make_predictor([0.08] * 200)
    alerter = AsyncMock()
    with patch("src.detection.drift_predictor.settings") as s:
        s.drift_alert_dedupe_seconds = 7200
        result = await predictor.maybe_send_alert(_alarming_trend(), alerter)
    assert result is True
    alerter.trigger.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_send_alert_deduplication():
    predictor = make_predictor([0.08] * 200)
    alerter = AsyncMock()
    with patch("src.detection.drift_predictor.settings") as s:
        s.drift_alert_dedupe_seconds = 7200
        await predictor.maybe_send_alert(_alarming_trend(), alerter)
        result2 = await predictor.maybe_send_alert(_alarming_trend(), alerter)
    assert result2 is False
    assert alerter.trigger.call_count == 1


@pytest.mark.asyncio
async def test_maybe_send_alert_dedup_expires():
    predictor = make_predictor([0.08] * 200)
    alerter = AsyncMock()
    predictor._last_alert_at = datetime.now(timezone.utc) - timedelta(hours=3)
    with patch("src.detection.drift_predictor.settings") as s:
        s.drift_alert_dedupe_seconds = 7200
        result = await predictor.maybe_send_alert(_alarming_trend(), alerter)
    assert result is True
    alerter.trigger.assert_called_once()


# ── should_compute_this_cycle() ──────────────────────────────────────────────

def test_should_compute_this_cycle_throttle():
    predictor = make_predictor([0.08] * 200)
    with patch("src.detection.drift_predictor.settings") as s:
        s.drift_prediction_interval_cycles = 5
        results = [predictor.should_compute_this_cycle() for _ in range(5)]
        assert results == [False, False, False, False, True]
        # After reset, the count starts again.
        assert predictor.should_compute_this_cycle() is False
