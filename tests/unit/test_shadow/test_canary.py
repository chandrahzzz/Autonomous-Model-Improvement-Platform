"""Unit tests for CanaryController decision logic (in-memory fake Redis)."""

import json
from datetime import datetime, timedelta

import pytest

from src.shadow.canary import CanaryController, ACTIVE_KEY, METRICS_KEY
from src.config.settings import settings


class FakeRedis:
    def __init__(self) -> None:
        self.kv: dict = {}
        self.hashes: dict = {}

    async def set(self, k, v):
        self.kv[k] = v

    async def get(self, k):
        return self.kv.get(k)

    async def delete(self, *keys):
        for k in keys:
            self.kv.pop(k, None)
            self.hashes.pop(k, None)

    async def hincrby(self, key, field, amount=1):
        h = self.hashes.setdefault(key, {})
        h[field] = int(h.get(field, 0)) + amount
        return h[field]

    async def hgetall(self, key):
        return {k: str(v) for k, v in self.hashes.get(key, {}).items()}


def _seed_active(redis, started_minutes_ago: float, version="v8"):
    redis.kv[ACTIVE_KEY] = json.dumps({
        "version": version,
        "start_ts": (datetime.utcnow() - timedelta(minutes=started_minutes_ago)).isoformat(),
        "traffic_pct": 0.05,
    })


@pytest.mark.asyncio
async def test_evaluate_none_when_inactive():
    cc = CanaryController(FakeRedis())
    decision, _ = await cc.evaluate()
    assert decision == "none"


@pytest.mark.asyncio
async def test_evaluate_pending_within_window():
    r = FakeRedis()
    _seed_active(r, started_minutes_ago=1)
    cc = CanaryController(r)
    decision, _ = await cc.evaluate()
    assert decision == "pending"


@pytest.mark.asyncio
async def test_evaluate_promote_clean_metrics():
    r = FakeRedis()
    _seed_active(r, started_minutes_ago=settings.canary_window_minutes + 1)
    r.hashes[METRICS_KEY] = {"requests": settings.canary_min_requests + 10, "errors": 0, "safety_failures": 0}
    cc = CanaryController(r)
    decision, metrics = await cc.evaluate()
    assert decision == "promote"
    assert metrics["error_rate"] == 0.0


@pytest.mark.asyncio
async def test_evaluate_rollback_on_safety_failure():
    r = FakeRedis()
    _seed_active(r, started_minutes_ago=settings.canary_window_minutes + 1)
    r.hashes[METRICS_KEY] = {"requests": settings.canary_min_requests + 10, "errors": 0, "safety_failures": 1}
    cc = CanaryController(r)
    decision, metrics = await cc.evaluate()
    assert decision == "rollback"
    assert "safety" in metrics["reason"]


@pytest.mark.asyncio
async def test_evaluate_rollback_on_high_error_rate():
    r = FakeRedis()
    _seed_active(r, started_minutes_ago=settings.canary_window_minutes + 1)
    n = settings.canary_min_requests + 10
    r.hashes[METRICS_KEY] = {"requests": n, "errors": n, "safety_failures": 0}
    cc = CanaryController(r)
    decision, _ = await cc.evaluate()
    assert decision == "rollback"


@pytest.mark.asyncio
async def test_evaluate_rollback_when_no_traffic_after_grace():
    r = FakeRedis()
    _seed_active(r, started_minutes_ago=2 * settings.canary_window_minutes + 1)
    r.hashes[METRICS_KEY] = {"requests": 0}
    cc = CanaryController(r)
    decision, metrics = await cc.evaluate()
    assert decision == "rollback"
    assert metrics["reason"] == "insufficient_canary_traffic"
