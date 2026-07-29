"""
Regression tests for the seven defects that stopped the pipeline loop from
completing. Each test fails against the pre-fix code.

Numbering matches the audit:
  1. ShadowRouter was never constructed -> shadow_logs empty -> loop hung
  2. ABCollector.elapsed_hours was `now - (now - ab_min_hours)`, a constant
  3. incumbent_scores was frozen at its hardcoded seed forever
  4. next version tag came from production, colliding after a rollback
  5. cycles_completed only moved on promotion, so maintenance jobs never ran
  6. log_monitor re-fed the same rows to the detectors every cycle
  7. audit_logger node was registered but unreachable
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import settings


# ── 2. A/B window elapsed time is measured, not assumed ──────────────────────
class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


def _row(delta, created_at):
    return SimpleNamespace(quality_delta=delta, created_at=created_at)


def _collector_with(rows):
    from src.shadow.ab_collector import ABCollector

    db = MagicMock()
    db.execute = AsyncMock(return_value=_FakeResult(rows))
    return ABCollector(db)


@pytest.mark.asyncio
async def test_elapsed_hours_is_measured_from_window_start(monkeypatch):
    """Pre-fix this returned exactly ab_min_hours regardless of reality."""
    monkeypatch.setattr(settings, "ab_min_hours", 48.0)
    monkeypatch.setattr(settings, "ab_min_requests", 10)

    started = datetime.now(timezone.utc) - timedelta(hours=3)
    rows = [_row(0.1, started + timedelta(minutes=i)) for i in range(20)]

    data = await _collector_with(rows).collect_window("v8", started_at=started)

    assert 2.5 < data["elapsed_hours"] < 3.5, data["elapsed_hours"]
    assert data["elapsed_hours"] != pytest.approx(48.0)


@pytest.mark.asyncio
async def test_window_not_ready_before_min_hours(monkeypatch):
    """The time gate must be able to FAIL — pre-fix it never could."""
    monkeypatch.setattr(settings, "ab_min_hours", 48.0)
    monkeypatch.setattr(settings, "ab_min_requests", 5)

    started = datetime.now(timezone.utc) - timedelta(hours=1)
    rows = [_row(0.1, started) for _ in range(50)]

    data = await _collector_with(rows).collect_window("v8", started_at=started)

    assert data["n_requests"] >= settings.ab_min_requests
    assert data["ready"] is False


@pytest.mark.asyncio
async def test_empty_window_reports_zero_elapsed(monkeypatch):
    monkeypatch.setattr(settings, "ab_min_hours", 48.0)
    data = await _collector_with([]).collect_window("v8", started_at=None)
    assert data["elapsed_hours"] == 0.0
    assert data["ready"] is False


@pytest.mark.asyncio
async def test_starved_window_times_out_so_the_graph_cannot_hang(monkeypatch):
    """No traffic must still reach a decision instead of looping forever."""
    monkeypatch.setattr(settings, "ab_min_hours", 48.0)
    monkeypatch.setattr(settings, "ab_min_requests", 1000)
    monkeypatch.setattr(settings, "ab_max_wait_hours", 72.0)

    started = datetime.now(timezone.utc) - timedelta(hours=100)
    data = await _collector_with([]).collect_window("v8", started_at=started)

    assert data["ready"] is False
    assert data["timed_out"] is True


@pytest.mark.asyncio
async def test_timed_out_window_is_rejected_not_promoted(monkeypatch):
    """Forcing a decision must not become a free promotion."""
    from src.shadow.promotion_gate import PromotionGate

    monkeypatch.setattr(settings, "ab_min_requests", 1000)
    eval_result = SimpleNamespace(
        safety_score=1.0, faithfulness=0.9, answer_relevancy=0.9, context_recall=0.9
    )
    ab_data = {"ready": False, "timed_out": True, "n_requests": 3, "elapsed_hours": 99.0}

    decision = PromotionGate().evaluate(ab_data, eval_result, {})

    assert decision.promote is False
    assert "timed_out" in decision.reason


# ── 3. The quality ratchet actually ratchets ─────────────────────────────────
def test_promoted_scores_become_the_next_incumbent():
    from src.graph.graph import _promoted_scores

    state = {
        "eval_result": {
            "faithfulness": 0.91,
            "answer_relevancy": 0.88,
            "context_recall": 0.85,
        }
    }
    assert _promoted_scores(state) == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.88,
        "context_recall": 0.85,
    }


def test_partial_scores_do_not_lower_the_bar():
    """A missing metric must leave the incumbent untouched, not zero it."""
    from src.graph.graph import _promoted_scores

    assert _promoted_scores({"eval_result": {"faithfulness": 0.9}}) is None
    assert _promoted_scores({"eval_result": {}}) is None
    assert _promoted_scores({}) is None


@pytest.mark.asyncio
async def test_gate4_prefers_incumbent_rescored_on_snapshot(monkeypatch):
    """Pre-fix the snapshot re-eval was computed then thrown away."""
    from src.graph.nodes.promotion_decider import promotion_decider_node

    captured = {}

    class _Gate:
        def evaluate(self, ab_data, eval_result, incumbent_scores):
            captured["incumbent"] = incumbent_scores
            return SimpleNamespace(promote=False, reason="stub", metrics={})

    monkeypatch.setattr("src.graph.nodes.promotion_decider._gate", _Gate())

    await promotion_decider_node({
        "version_tag": "v9",
        "eval_passed": True,
        "ab_data": {"ready": True},
        "incumbent_scores": {"faithfulness": 0.70},   # stale seed
        "eval_result": {
            "faithfulness": 0.9,
            "incumbent_scores_on_snapshot": {"faithfulness": 0.86},
        },
    })

    assert captured["incumbent"] == {"faithfulness": 0.86}


# ── 4. A rolled-back version tag is never reissued ───────────────────────────
@pytest.mark.asyncio
async def test_next_version_tag_skips_rolled_back_versions():
    """Production stays at v7 after v8 rolls back; the next tag must be v9."""
    from src.db.repositories.model_versions import ModelRepository

    db = MagicMock()
    db.execute = AsyncMock(return_value=_FakeResult([("v7",), ("v8",)]))

    assert await ModelRepository(db).next_version_tag() == "v9"


@pytest.mark.asyncio
async def test_next_version_tag_ignores_non_numeric_tags():
    from src.db.repositories.model_versions import ModelRepository

    db = MagicMock()
    db.execute = AsyncMock(
        return_value=_FakeResult([("v7",), ("baseline",), (None,), ("v12",)])
    )

    assert await ModelRepository(db).next_version_tag() == "v13"


# ── 5. Maintenance jobs are scheduled off a counter that actually moves ──────
def test_due_uses_runner_local_cycle_count():
    from src.graph.runner import PipelineRunner

    runner = PipelineRunner.__new__(PipelineRunner)  # skip graph compilation
    runner._cycle_count = 0

    assert runner._due(10) is False          # nothing is due at startup
    runner._cycle_count = 10
    assert runner._due(10) is True           # pre-fix this never happened
    runner._cycle_count = 11
    assert runner._due(10) is False
    runner._cycle_count = 20
    assert runner._due(10) is True
    assert runner._due(0) is False           # disabled interval


def test_cycle_count_is_independent_of_promotion_state():
    """The old counter only advanced on promotion, so intervals never elapsed."""
    from src.graph.runner import PipelineRunner

    runner = PipelineRunner.__new__(PipelineRunner)
    runner._cycle_count = 0
    runner._state = {"cycles_completed": 0}   # never promoted

    for _ in range(20):
        runner._cycle_count += 1

    assert runner._due(20) is True


# ── 6. Logs are handed to the detectors once ─────────────────────────────────
class _FakeRedis:
    def __init__(self):
        self.members: set[str] = set()
        self.expired_with = None

    async def smismember(self, key, ids):
        return [1 if i in self.members else 0 for i in ids]

    async def sadd(self, key, *ids):
        self.members.update(ids)

    async def expire(self, key, ttl):
        self.expired_with = ttl


@pytest.mark.asyncio
async def test_already_processed_logs_are_skipped(monkeypatch):
    from src.graph.nodes import log_monitor

    fake = _FakeRedis()
    monkeypatch.setattr(log_monitor, "_get_factory_redis", lambda: fake)
    monkeypatch.setattr(settings, "log_dedup_enabled", True)

    first = await log_monitor._filter_unprocessed(["a", "b", "c"])
    assert first == ["a", "b", "c"]

    second = await log_monitor._filter_unprocessed(["a", "b", "c", "d"])
    assert second == ["d"]          # pre-fix: all four, every cycle
    assert fake.expired_with == settings.log_dedup_ttl_seconds


@pytest.mark.asyncio
async def test_dedup_fails_open_on_redis_error(monkeypatch):
    """Losing Redis must never drop logs — reprocessing is the safe failure."""
    from src.graph.nodes import log_monitor

    class _Broken:
        async def smismember(self, *_):
            raise RuntimeError("redis down")

    monkeypatch.setattr(log_monitor, "_get_factory_redis", lambda: _Broken())
    monkeypatch.setattr(settings, "log_dedup_enabled", True)

    assert await log_monitor._filter_unprocessed(["a", "b"]) == ["a", "b"]


@pytest.mark.asyncio
async def test_dedup_can_be_disabled(monkeypatch):
    from src.graph.nodes import log_monitor

    monkeypatch.setattr(settings, "log_dedup_enabled", False)
    assert await log_monitor._filter_unprocessed(["a", "a"]) == ["a", "a"]


# ── 7. The audit_logger node is reachable ────────────────────────────────────
def test_failure_batch_is_audited_before_curation():
    from src.graph.edges import after_failure_detector

    assert after_failure_detector({"has_failures": True}) == "audit_logger"
    assert after_failure_detector({"has_failures": False}) == "data_validator"


def test_audit_logger_is_wired_into_the_graph():
    """Pre-fix the node was registered but no edge ever reached it."""
    from src.graph.graph import build_graph

    graph = build_graph()
    targets = {edge[1] for edge in graph.edges}
    branch_targets = set()
    for branches in getattr(graph, "branches", {}).values():
        for branch in branches.values():
            branch_targets.update((branch.ends or {}).values())

    assert "audit_logger" in (targets | branch_targets)
    assert ("audit_logger", "example_curator") in graph.edges


def test_terminal_nodes_clear_promotion_decision():
    """A sticky decision would mislabel the next cycle's detection audit entry."""
    import inspect
    from src.graph import graph as graph_mod
    from src.graph.nodes import rollback_node as rollback_mod

    assert '"promotion_decision": None' in inspect.getsource(
        graph_mod.promote_model_node
    )
    assert '"promotion_decision": None' in inspect.getsource(
        rollback_mod.rollback_node
    )


# ── 1. The shadow hook is reachable from the ingestion path ──────────────────
@pytest.mark.asyncio
async def test_observe_is_a_noop_when_no_challenger(monkeypatch):
    from src.shadow import service

    service.reset_for_tests()

    class _Router:
        async def get_challenger_version(self):
            return None

    monkeypatch.setattr(service, "_get_router", lambda: _Router())
    monkeypatch.setattr(settings, "shadow_observation_enabled", True)

    assert await service.observe_production_call("p", "o") is None


@pytest.mark.asyncio
async def test_observe_routes_to_shadow_when_challenger_active(monkeypatch):
    from src.shadow import service

    service.reset_for_tests()
    seen = {}

    class _Router:
        async def get_challenger_version(self):
            return "v8"

        async def maybe_shadow(self, prompt, production_output, invoke_fn):
            seen["prompt"] = prompt
            seen["challenger_output"] = await invoke_fn(prompt)
            return 0.0

    monkeypatch.setattr(service, "_get_router", lambda: _Router())
    monkeypatch.setattr(settings, "shadow_observation_enabled", True)
    monkeypatch.setattr(settings, "eval_real_inference", False)
    monkeypatch.setattr(settings, "shadow_allow_stub_challenger", True)

    delta = await service.observe_production_call("why is the sky blue?", "because")

    assert delta == 0.0
    assert seen["prompt"] == "why is the sky blue?"
    assert seen["challenger_output"] == "because"   # stub echoes production


@pytest.mark.asyncio
async def test_observe_never_raises_into_the_caller(monkeypatch):
    """Shadowing must never break ingestion or serving."""
    from src.shadow import service

    service.reset_for_tests()

    def _boom():
        raise RuntimeError("redis exploded")

    monkeypatch.setattr(service, "_get_router", _boom)
    monkeypatch.setattr(settings, "shadow_observation_enabled", True)

    assert await service.observe_production_call("p", "o") is None


@pytest.mark.asyncio
async def test_observe_disabled_by_kill_switch(monkeypatch):
    from src.shadow import service

    service.reset_for_tests()
    monkeypatch.setattr(settings, "shadow_observation_enabled", False)

    assert await service.observe_production_call("p", "o") is None


def test_simulator_emit_path_calls_the_shadow_hook():
    """The simulator is the app layer, so it must feed the shadow window."""
    import inspect
    from sim import emit

    source = inspect.getsource(emit._emit_db)
    assert "observe_production_call" in source


def test_interceptor_calls_the_shadow_hook():
    import inspect
    from src.middleware.llm_interceptor import LLMInterceptorMiddleware

    source = inspect.getsource(LLMInterceptorMiddleware._emit_event)
    assert "observe_production_call" in source
