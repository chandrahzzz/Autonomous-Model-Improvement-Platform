"""
Unit tests for the synthetic traffic simulator (sim/ package).

No external services: scenario planning + call generation are pure/seeded, and
the emit layer is exercised with mocked DB / producer. Failure injectors are
asserted against the REAL detector contracts (e.g. the actual refusal regex)
so the simulator can't silently drift out of sync with the detectors.
"""

import json
import random

import pytest

from sim.config import SimConfig, FAILURE_TYPES
from sim.answers import generate_call, Call, REFUSAL_COMPLETIONS, DRIFT_COMPLETIONS
from sim.scenarios import build_call_plan
from sim.prompts import ACME_FACTS, BANKS
from sim.emit import call_to_row, call_to_event


def _rng():
    return random.Random(42)


# ── Scenario planning ─────────────────────────────────────────────────────────
def test_healthy_scenario_has_no_failures():
    cfg = SimConfig(scenario="healthy", refusal_rate=0.5)  # rate ignored by healthy
    plan = build_call_plan(cfg, 200)
    assert all(spec.failure_type is None for spec in plan)


def test_plan_is_deterministic_for_seed():
    cfg = SimConfig(scenario="mixed", refusal_rate=0.3, drift_rate=0.2, seed=7)
    a = build_call_plan(cfg, 300)
    b = build_call_plan(cfg, 300)
    assert [s.failure_type for s in a] == [s.failure_type for s in b]
    assert [s.domain for s in a] == [s.domain for s in b]


def test_mixed_scenario_produces_configured_failures():
    cfg = SimConfig(scenario="mixed", hallucination_rate=0.3, refusal_rate=0.3,
                    format_break_rate=0.3, drift_rate=0.3)
    plan = build_call_plan(cfg, 800)
    seen = {s.failure_type for s in plan}
    for f in FAILURE_TYPES:
        assert f in seen, f"expected some {f} in a high-rate mixed run"
    assert None in seen  # still some healthy


def test_degrade_warmup_is_clean():
    cfg = SimConfig(scenario="degrade", refusal_rate=0.9, drift_rate=0.9)
    n = 1000
    plan = build_call_plan(cfg, n)
    warmup = plan[: int(0.30 * n)]
    assert all(s.failure_type is None for s in warmup), "first 30% must seed a clean baseline"
    tail = plan[int(0.30 * n):]
    assert any(s.failure_type is not None for s in tail), "failures must appear after warmup"


def test_burst_is_localised_to_the_middle():
    cfg = SimConfig(scenario="burst", refusal_rate=0.5)
    n = 1000
    plan = build_call_plan(cfg, n)
    outside = plan[: int(0.4 * n)] + plan[int(0.6 * n):]
    middle = plan[int(0.4 * n): int(0.6 * n)]
    assert all(s.failure_type is None for s in outside)
    assert sum(s.failure_type is not None for s in middle) > 0.5 * len(middle)


def test_rag_heavy_is_all_business_policy():
    cfg = SimConfig(scenario="rag_heavy", hallucination_rate=0.3)
    plan = build_call_plan(cfg, 300)
    assert all(s.domain == "business_policy" for s in plan)


# ── Failure injectors match the real detector contracts ───────────────────────
def test_refusal_matches_real_refusal_regex():
    from src.detection.refusal import REFUSAL_PATTERN
    rng = _rng()
    for _ in range(30):
        call = generate_call("factual", "refusal", rng)
        assert call.failure_type == "refusal"
        assert REFUSAL_PATTERN.search(call.completion), \
            f"refusal completion must match the detector regex: {call.completion!r}"


def test_all_static_refusal_completions_match_regex():
    from src.detection.refusal import REFUSAL_PATTERN
    for text in REFUSAL_COMPLETIONS:
        assert REFUSAL_PATTERN.search(text), f"{text!r} would not be detected as a refusal"


def test_hallucination_is_grounded_and_contradictory():
    rng = _rng()
    for _ in range(20):
        call = generate_call("business_policy", "hallucination", rng)
        assert call.is_rag is True
        assert call.retrieved_context, "hallucination must carry retrieved_context"
        assert call.retrieved_context in ACME_FACTS.values()
        # The wrong answer must differ from the grounding fact.
        assert call.completion.strip() != call.retrieved_context.strip()


def test_format_break_is_not_valid_json():
    rng = _rng()
    broke = 0
    for _ in range(20):
        call = generate_call("json_format", "format_break", rng)
        assert call.expect_json is True
        try:
            json.loads(call.completion)
        except (json.JSONDecodeError, ValueError):
            broke += 1
    assert broke >= 15, "most format_break completions should fail JSON parsing"


def test_drift_completions_are_off_distribution():
    rng = _rng()
    for _ in range(10):
        call = generate_call("factual", "drift", rng)
        assert call.completion in DRIFT_COMPLETIONS


def test_healthy_answers_render_cleanly():
    rng = _rng()
    for domain in BANKS:
        for _ in range(10):
            call = generate_call(domain, None, rng)
            assert call.failure_type is None
            assert call.prompt and call.completion
            assert "{" not in call.prompt or domain == "json_format"


# ── Schema mapping ────────────────────────────────────────────────────────────
_LLMLOG_ATTRS = {
    "session_id", "user_cohort", "model_version", "prompt", "completion",
    "retrieved_context", "prompt_tokens", "completion_tokens", "latency_ms",
    "finish_reason", "cost_usd", "embedding_hash", "metadata_", "created_at",
}


def test_call_to_row_keys_are_valid_llmlog_attrs():
    cfg = SimConfig()
    call = generate_call("business_policy", "hallucination", _rng())
    row = call_to_row(cfg, call, _rng(), 0, 10, {})
    assert set(row) <= _LLMLOG_ATTRS, f"unexpected keys: {set(row) - _LLMLOG_ATTRS}"
    # Constructing the ORM object must not raise (validates against the model).
    from src.db.models import LLMLog
    LLMLog(**row)


def test_call_to_row_puts_is_rag_in_metadata():
    cfg = SimConfig()
    rag_call = generate_call("business_policy", "hallucination", _rng())
    row = call_to_row(cfg, rag_call, _rng(), 0, 10, {})
    assert row["metadata_"]["is_rag"] is True
    assert row["metadata_"]["sim"] is True

    plain = generate_call("factual", None, _rng())
    row2 = call_to_row(cfg, plain, _rng(), 0, 10, {})
    assert row2["metadata_"]["is_rag"] is False


def test_healthy_row_satisfies_known_good_replay_filter():
    """Replay filter = finish_reason='stop' AND latency_ms < 3000."""
    cfg = SimConfig()
    rng = random.Random(3)
    for i in range(50):
        call = generate_call("factual", None, rng)
        row = call_to_row(cfg, call, rng, i, 50, {})
        assert row["finish_reason"] == "stop"
        assert row["latency_ms"] < 3000


def test_call_to_event_builds_valid_llmevent():
    cfg = SimConfig()
    call = generate_call("business_policy", "hallucination", _rng())
    payload = call_to_event(cfg, call, _rng(), 0, 10, {})
    assert payload["is_rag"] is True
    assert payload["retrieved_context"]
    assert payload["completion"] == call.completion
    assert "event_id" in payload and "timestamp" in payload


# ── Emit dispatch ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_emit_unknown_mode_raises():
    from sim.emit import emit_calls
    cfg = SimConfig(mode="bogus")
    with pytest.raises(ValueError):
        await emit_calls([], cfg)


@pytest.mark.asyncio
async def test_emit_db_inserts_all_rows(monkeypatch):
    """db mode must insert one row per call via the repository."""
    import sim.emit as emit_mod
    from contextlib import asynccontextmanager
    from unittest.mock import AsyncMock, MagicMock

    inserted = []

    fake_repo = MagicMock()
    fake_repo.insert = AsyncMock(side_effect=lambda data: inserted.append(data))

    @asynccontextmanager
    async def fake_get_db():
        yield MagicMock()

    import src.db.connection as conn_mod
    import src.db.repositories.llm_logs as repo_mod
    monkeypatch.setattr(conn_mod, "get_db", fake_get_db)
    monkeypatch.setattr(repo_mod, "LLMLogRepository", lambda db: fake_repo)

    cfg = SimConfig(mode="db", scenario="mixed", refusal_rate=0.3, db_batch_size=64)
    _, calls = _build(cfg, 150)
    stats = await emit_mod.emit_calls(calls, cfg)
    assert stats["inserted"] == 150
    assert len(inserted) == 150


def _build(cfg: SimConfig, n: int):
    plan = build_call_plan(cfg, n)
    rng = random.Random(cfg.seed + 99)
    calls = [generate_call(s.domain, s.failure_type, rng) for s in plan]
    return plan, calls
