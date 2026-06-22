"""Unit tests for the June 2026 shadow / state-machine / infra hardening.

No Kafka / DB / Redis: pure helpers are tested directly and stateful components
use fakes or mocked clients.
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from src.config.settings import settings
from src.shadow.router import _bucket_keep_multiplier
from src.shadow.canary import CanaryController
from src.graph.runner import _should_fast_path
from src.kafka.dlq_consumer import DLQReplayer
from src.monitoring import metrics


# ── S1 stratified sampling multiplier ───────────────────────────────────────────

def test_bucket_multiplier_balances_overrepresented():
    counts = {"100": 50, "101": 10}  # bucket 100 is over-represented
    assert _bucket_keep_multiplier(counts, "100") < 1.0   # downsample peak bucket
    assert _bucket_keep_multiplier(counts, "101") == 1.0  # min bucket kept fully


def test_bucket_multiplier_empty_or_new_is_one():
    assert _bucket_keep_multiplier({}, "100") == 1.0
    assert _bucket_keep_multiplier({"100": 5}, "999") == 1.0  # new bucket = min


def test_bucket_multiplier_floor():
    counts = {"a": 1000, "b": 1}
    assert _bucket_keep_multiplier(counts, "a") >= 0.1  # never fully starves a bucket


# ── S2 canary real-time auto-abort ──────────────────────────────────────────────

def _canary_with_metrics(metrics_hash, active=True):
    cc = CanaryController(redis=Mock())
    # start_ts must match the naive datetime.utcnow() the controller subtracts against.
    cc.get_active = AsyncMock(
        return_value={"version": "v9", "start_ts": datetime.utcnow().isoformat(),
                      "traffic_pct": 0.05} if active else None
    )
    r = AsyncMock()
    r.hgetall = AsyncMock(return_value=metrics_hash)
    cc._r = AsyncMock(return_value=r)
    cc._alert_auto_abort = AsyncMock()
    return cc


@pytest.mark.asyncio
async def test_canary_auto_aborts_on_error_spike(monkeypatch):
    monkeypatch.setattr(settings, "canary_min_requests_for_abort", 20)
    monkeypatch.setattr(settings, "canary_max_error_rate", 0.01)
    monkeypatch.setattr(settings, "canary_abort_error_multiplier", 2.0)
    before = metrics.canary_auto_aborts_total._value.get()
    # 50 requests, 5 errors = 10% error rate, threshold = 2% → abort.
    cc = _canary_with_metrics({"requests": "50", "errors": "5", "safety_failures": "0"})
    decision, m = await cc.evaluate()
    assert decision == "rollback"
    assert "error_rate_spike" in m["reason"]
    assert metrics.canary_auto_aborts_total._value.get() == before + 1
    cc._alert_auto_abort.assert_awaited_once()


@pytest.mark.asyncio
async def test_canary_no_abort_when_healthy(monkeypatch):
    monkeypatch.setattr(settings, "canary_min_requests_for_abort", 20)
    monkeypatch.setattr(settings, "canary_max_error_rate", 0.01)
    monkeypatch.setattr(settings, "canary_abort_error_multiplier", 2.0)
    monkeypatch.setattr(settings, "canary_window_minutes", 60)
    # 100 requests, 0 errors, window not elapsed → pending, no abort.
    cc = _canary_with_metrics({"requests": "100", "errors": "0", "safety_failures": "0"})
    decision, _ = await cc.evaluate()
    assert decision == "pending"


@pytest.mark.asyncio
async def test_canary_aborts_on_safety_failure(monkeypatch):
    monkeypatch.setattr(settings, "canary_min_requests_for_abort", 20)
    cc = _canary_with_metrics({"requests": "30", "errors": "0", "safety_failures": "1"})
    decision, m = await cc.evaluate()
    assert decision == "rollback"
    assert m["reason"] == "safety_failure_during_canary"


# ── L3 fast-path decision ────────────────────────────────────────────────────────

def test_fast_path_engages_on_burst(monkeypatch):
    monkeypatch.setattr(settings, "high_severity_failure_threshold", 500)
    monkeypatch.setattr(settings, "max_consecutive_fast_cycles", 5)
    assert _should_fast_path(600, "monitoring", 0) is True


def test_fast_path_bounded_and_phase_scoped(monkeypatch):
    monkeypatch.setattr(settings, "high_severity_failure_threshold", 500)
    monkeypatch.setattr(settings, "max_consecutive_fast_cycles", 5)
    assert _should_fast_path(600, "monitoring", 5) is False   # bound reached
    assert _should_fast_path(600, "ab_testing", 0) is False   # only in monitoring
    assert _should_fast_path(10, "monitoring", 0) is False    # below threshold


# ── I1 DLQ replayer payload handling ────────────────────────────────────────────

def test_dlq_unwrap_envelope():
    topic, payload, attempts = DLQReplayer._next_payload(
        {"original_topic": "llm.production.events", "payload": {"x": 1}, "attempts": 2}
    )
    assert topic == "llm.production.events"
    assert payload == {"x": 1}
    assert attempts == 2


def test_dlq_unwrap_bare_payload():
    topic, payload, attempts = DLQReplayer._next_payload({"x": 1})
    assert payload == {"x": 1}
    assert attempts == 0


class _FakeKafkaMsg:
    def __init__(self, value):
        self._v = json.dumps(value).encode("utf-8")
    def error(self): return None
    def value(self): return self._v


@pytest.mark.asyncio
async def test_dlq_replays_to_original_topic(monkeypatch):
    monkeypatch.setattr(settings, "dlq_replay_max_attempts", 5)
    before = metrics.dlq_replayed_total._value.get()
    producer = AsyncMock()
    replayer = DLQReplayer(producer=producer)
    # One message then None (drained).
    consumer = Mock()
    consumer.poll = Mock(side_effect=[
        _FakeKafkaMsg({"original_topic": "llm.production.events", "payload": {"a": 1}, "attempts": 0}),
        None,
    ])
    consumer.commit = Mock()
    replayer._consumer = consumer
    replayed, dropped = await replayer.replay_batch(max_messages=10)
    assert (replayed, dropped) == (1, 0)
    producer.produce.assert_awaited_with("llm.production.events", {"a": 1})
    assert metrics.dlq_replayed_total._value.get() == before + 1


@pytest.mark.asyncio
async def test_dlq_drops_after_max_attempts(monkeypatch):
    monkeypatch.setattr(settings, "dlq_replay_max_attempts", 5)
    before = metrics.dlq_replay_failed_total._value.get()
    producer = AsyncMock()
    replayer = DLQReplayer(producer=producer)
    consumer = Mock()
    consumer.poll = Mock(side_effect=[
        _FakeKafkaMsg({"original_topic": "t", "payload": {"a": 1}, "attempts": 5}),
        None,
    ])
    consumer.commit = Mock()
    replayer._consumer = consumer
    replayed, dropped = await replayer.replay_batch(max_messages=10)
    assert (replayed, dropped) == (0, 1)
    producer.produce.assert_not_awaited()  # exhausted → dropped, not replayed
    assert metrics.dlq_replay_failed_total._value.get() == before + 1
