"""Unit tests for ThresholdCalibrator drop-rate logic (fake async DB)."""

import pytest

from src.detection.calibrator import ThresholdCalibrator
from src.config.settings import settings


class _Result:
    def __init__(self, value):
        self._value = value

    def scalar(self):
        return self._value


class FakeDB:
    """Routes queries by SQL text; records calibration_history inserts."""

    def __init__(self, detected: dict, curated: dict):
        self._detected = detected
        self._curated = curated
        self.inserts: list[dict] = []

    async def execute(self, stmt, params=None):
        sql = str(stmt)
        params = params or {}
        if "INSERT INTO calibration_history" in sql:
            self.inserts.append(params)
            return _Result(None)
        ftype = params.get("t")
        if "FROM failure_classifications" in sql:
            return _Result(self._detected.get(ftype, 0))
        if "FROM training_examples" in sql:
            return _Result(self._curated.get(ftype, 0))
        return _Result(0)


@pytest.mark.asyncio
async def test_high_drop_rate_suggests_less_sensitive_threshold():
    # 100 detected, 10 curated -> drop_rate 0.9 (>> target) -> raise threshold.
    db = FakeDB(detected={"hallucination": 100}, curated={"hallucination": 10})
    suggestions = await ThresholdCalibrator().run(db)
    hall = [s for s in suggestions if s["metric"] == "hallucination_threshold"]
    assert hall, "expected a hallucination suggestion"
    assert hall[0]["suggested"] > hall[0]["current"]
    assert db.inserts  # written to calibration_history


@pytest.mark.asyncio
async def test_insufficient_samples_no_suggestion():
    db = FakeDB(detected={"hallucination": settings.calibration_min_samples - 1}, curated={})
    suggestions = await ThresholdCalibrator().run(db)
    assert suggestions == []


@pytest.mark.asyncio
async def test_healthy_drop_rate_no_suggestion():
    # drop_rate within tolerated band -> no change.
    n = 100
    keep = int(n * (1 - settings.calibration_fp_target * 0.9))  # drop_rate ~ just under target
    db = FakeDB(detected={"hallucination": n}, curated={"hallucination": keep})
    suggestions = await ThresholdCalibrator().run(db)
    assert all(s["metric"] != "hallucination_threshold" for s in suggestions)
