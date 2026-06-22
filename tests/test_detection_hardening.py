"""Unit tests for the June 2026 detection-layer hardening.

These never load the real NLI / embedding models: detectors are either
constructed via ``__new__`` (skipping the heavy ``__init__``) or replaced with
lightweight fakes, so the suite stays fast and CPU-only.
"""

from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.config.settings import settings
from src.detection.drift import DriftDetector
from src.detection.refusal import RefusalDetector
from src.detection.format_validator import FormatValidator
from src.detection.failure_classifier import FailureClassifier, FailureEvent
from src.monitoring import metrics


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def set(self, k, v, **kw):
        self.store[k] = v

    async def get(self, k):
        return self.store.get(k)


# ── #6 correlation collapse (pure) ────────────────────────────────────────────

def _ev(ftype, score, log_id="L1"):
    return FailureEvent(llm_log_id=log_id, prompt="p", completion="c",
                        failure_type=ftype, score=score)


def test_collapse_keeps_highest_severity_and_records_all():
    cands = [_ev("format_regression", 0.9), _ev("hallucination", 0.6),
             _ev("refusal_creep", 0.7)]
    out = FailureClassifier._collapse(cands)
    assert len(out) == 1
    assert out[0].failure_type == "hallucination"  # highest priority wins
    assert out[0].metadata["is_correlated"] is True
    assert set(out[0].metadata["all_failure_types"]) == {
        "format_regression", "hallucination", "refusal_creep"}


def test_collapse_single_candidate_untouched():
    out = FailureClassifier._collapse([_ev("refusal_creep", 0.8)])
    assert len(out) == 1
    assert "is_correlated" not in out[0].metadata


def test_collapse_disabled_emits_all(monkeypatch):
    monkeypatch.setattr(settings, "correlate_failures_enabled", False)
    out = FailureClassifier._collapse([_ev("hallucination", 0.9), _ev("refusal_creep", 0.7)])
    assert len(out) == 2


# ── classify_batch: single drift score + correlation + premise (mocked) ───────

class _FakeDrift:
    def __init__(self, drifting):
        self.calls = 0
        self._drifting = drifting

    def score(self, text):
        self.calls += 1
        return 0.2

    @property
    def rolling_drift_score(self):
        return 0.2

    def is_drifting(self):
        return self._drifting


class _FakeHall:
    def __init__(self, scores):
        self.scores = scores

    async def score_batch(self, pairs):
        return self.scores


class _FakeRefusal:
    def __init__(self, results):
        self.results = results

    async def classify_batch(self, events):
        return self.results


class _FakeFormat:
    def __init__(self, results):
        self.results = results

    async def validate_batch(self, events):
        return self.results


@pytest.mark.asyncio
async def test_drift_scored_exactly_once_per_event():
    drift = _FakeDrift(drifting=True)
    clf = FailureClassifier(
        _FakeHall([0.0, 0.0]), drift,
        _FakeRefusal([("", 0.0), ("", 0.0)]),
        _FakeFormat([("", 0.0), ("", 0.0)]),
    )
    events = [{"id": "1", "completion": "a"}, {"id": "2", "completion": "b"}]
    await clf.classify_batch(events)
    assert drift.calls == 2  # previously this was 4 (double-scored when drifting)


@pytest.mark.asyncio
async def test_multi_detector_same_log_collapses_to_one(monkeypatch):
    monkeypatch.setattr(settings, "correlate_failures_enabled", True)
    drift = _FakeDrift(drifting=True)
    clf = FailureClassifier(
        _FakeHall([0.9]), drift,
        _FakeRefusal([("refusal_creep", 0.8)]),
        _FakeFormat([("format_regression", 0.7)]),
    )
    batch = await clf.classify_batch([{"id": "L9", "prompt": "p", "completion": "c"}])
    assert len(batch.events) == 1
    ev = batch.events[0]
    assert ev.failure_type == "hallucination"
    assert ev.metadata["is_correlated"] is True
    assert set(ev.metadata["all_failure_types"]) == {
        "hallucination", "semantic_drift", "refusal_creep", "format_regression"}


@pytest.mark.asyncio
async def test_rag_call_missing_context_counts_and_flags():
    before = metrics.hallucination_premise_missing_total._value.get()
    drift = _FakeDrift(drifting=False)
    clf = FailureClassifier(
        _FakeHall([0.9]), drift,
        _FakeRefusal([("", 0.0)]), _FakeFormat([("", 0.0)]),
    )
    # is_rag True but no retrieved_context → premise falls back to prompt.
    batch = await clf.classify_batch(
        [{"id": "R1", "prompt": "p", "completion": "c", "is_rag": True}]
    )
    after = metrics.hallucination_premise_missing_total._value.get()
    assert after == before + 1
    ev = batch.events[0]
    assert ev.metadata["premise_source"] == "prompt_fallback"
    assert ev.metadata["grounded"] is False


@pytest.mark.asyncio
async def test_grounded_rag_call_marked_grounded():
    drift = _FakeDrift(drifting=False)
    clf = FailureClassifier(
        _FakeHall([0.9]), drift,
        _FakeRefusal([("", 0.0)]), _FakeFormat([("", 0.0)]),
    )
    batch = await clf.classify_batch([{
        "id": "R2", "prompt": "p", "completion": "c",
        "retrieved_context": "the grounding doc", "is_rag": True,
    }])
    ev = batch.events[0]
    assert ev.metadata["premise_source"] == "context"
    assert ev.metadata["grounded"] is True


# ── #3 min-sample guards ───────────────────────────────────────────────────────

def _bare_drift():
    d = DriftDetector.__new__(DriftDetector)
    d._centroid = np.array([0.0])
    d._cov_inv = np.array([[1.0]])
    d._loaded = True
    d._computed_at = None
    d._window = deque(maxlen=1000)
    return d


def test_drift_not_drifting_below_min_window(monkeypatch):
    monkeypatch.setattr(settings, "drift_min_window", 50)
    monkeypatch.setattr(settings, "drift_mahalanobis_threshold", 0.15)
    d = _bare_drift()
    d._window.extend([0.9] * 10)  # high score but only 10 samples
    assert d.is_drifting() is False
    d._window.extend([0.9] * 50)  # now > 50 samples, still high
    assert d.is_drifting() is True


def _bare_refusal():
    r = RefusalDetector.__new__(RefusalDetector)
    r._window = deque(maxlen=500)
    r._baseline_rate = 0.05
    return r


def test_refusal_not_creeping_below_min_samples(monkeypatch):
    monkeypatch.setattr(settings, "refusal_min_samples", 50)
    monkeypatch.setattr(settings, "refusal_rate_multiplier", 2.0)
    r = _bare_refusal()
    r._window.extend([True] * 5)  # 100% rate but tiny window
    assert r.is_creeping() is False
    r._window.clear()
    r._window.extend([True] * 60)  # enough samples, well above baseline
    assert r.is_creeping() is True


# ── #2 / #5 baseline age ────────────────────────────────────────────────────────

def test_drift_baseline_age_and_staleness(monkeypatch):
    monkeypatch.setattr(settings, "drift_baseline_max_age_hours", 24.0)
    d = _bare_drift()
    assert d.baseline_age_hours is None
    assert d.is_baseline_stale() is False  # no baseline → not "stale"
    d._computed_at = datetime.now(timezone.utc) - timedelta(hours=30)
    assert d.baseline_age_hours > 29
    assert d.is_baseline_stale() is True
    d._computed_at = datetime.now(timezone.utc) - timedelta(hours=1)
    assert d.is_baseline_stale() is False


def test_format_refresh_baseline_and_staleness(monkeypatch):
    monkeypatch.setattr(settings, "format_min_samples", 100)
    monkeypatch.setattr(settings, "format_baseline_max_age_hours", 24.0)
    fv = FormatValidator()
    assert fv.refresh_baseline(["word " * 10] * 50) is False  # too few
    assert fv.baseline_age_hours is None
    assert fv.refresh_baseline(["word " * 10] * 120) is True
    assert fv.baseline_age_hours is not None and fv.baseline_age_hours < 1
    assert fv.is_baseline_stale() is False
    fv._baseline_computed_at = datetime.now(timezone.utc) - timedelta(hours=30)
    assert fv.is_baseline_stale() is True


# ── #4 window persistence round-trips ────────────────────────────────────────

@pytest.mark.asyncio
async def test_drift_window_persist_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "detector_state_persist_enabled", True)
    r = FakeRedis()
    d = _bare_drift()
    d._window.extend([0.1, 0.2, 0.3])
    await d.save_window_state(r)
    d2 = _bare_drift()
    await d2.load_window_state(r)
    assert list(d2._window) == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_refusal_window_persist_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "detector_state_persist_enabled", True)
    r = FakeRedis()
    rd = _bare_refusal()
    rd._window.extend([True, False, True])
    await rd.save_window_state(r)
    rd2 = _bare_refusal()
    await rd2.load_window_state(r)
    assert list(rd2._window) == [True, False, True]


@pytest.mark.asyncio
async def test_format_state_persist_roundtrip(monkeypatch):
    monkeypatch.setattr(settings, "detector_state_persist_enabled", True)
    monkeypatch.setattr(settings, "format_min_samples", 5)
    r = FakeRedis()
    fv = FormatValidator()
    fv.refresh_baseline(["a b c"] * 10)
    fv._length_window.extend([3, 4, 5])
    await fv.save_window_state(r)
    fv2 = FormatValidator()
    await fv2.load_window_state(r)
    assert list(fv2._length_window) == [3, 4, 5]
    assert fv2._baseline_length_dist is not None
    assert fv2.baseline_age_hours is not None


@pytest.mark.asyncio
async def test_persist_disabled_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "detector_state_persist_enabled", False)
    r = FakeRedis()
    d = _bare_drift()
    d._window.extend([0.5])
    await d.save_window_state(r)
    assert r.store == {}  # nothing written when disabled
