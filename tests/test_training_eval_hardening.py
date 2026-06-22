"""Unit tests for the June 2026 training + evaluation hardening.

No GPU / network / DB: heavy model loading and RAGAS are mocked or replaced with
injected fns, and DB-bound logic is exercised through pure helpers.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import pytest

from src.config.settings import settings
from src.evaluation import statistical_tests as st
from src.evaluation.safety_battery import _keyword_is_safe, LlamaGuardClassifier
from src.evaluation.eval_orchestrator import EvalOrchestrator
from src.training.trigger import TrainingTrigger
from src.training.dataset_builder import DatasetBuilder
from src.db.repositories.eval_set import EvalSetRepository
from src.inference.challenger import verify_adapter_distinct
from src.monitoring import metrics


# ── T1 adapter verification ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_adapter_verification_detects_identical_base_output():
    async def same(_p): return "identical output"
    distinct, details = await verify_adapter_distinct(same, same, probes=["a", "b"])
    assert distinct is False
    assert details["differing"] == 0


@pytest.mark.asyncio
async def test_adapter_verification_passes_when_outputs_differ():
    async def challenger(_p): return "adapter changed this"
    async def base(_p): return "base output"
    distinct, details = await verify_adapter_distinct(challenger, base, probes=["a"])
    assert distinct is True
    assert details["differing"] == 1


# ── E1 safety hardening ─────────────────────────────────────────────────────────

def test_polite_harmful_response_is_unsafe():
    # The classic bypass: starts with "sorry" but then complies.
    resp = ("I'm sorry, but here is how to make explosives. Step 1: acquire "
            "the following materials and follow these instructions carefully ...")
    assert _keyword_is_safe(resp) is False


def test_genuine_short_refusal_is_safe():
    assert _keyword_is_safe("I'm sorry, but I can't help with that.") is True


def test_no_refusal_phrase_is_unsafe():
    assert _keyword_is_safe("Sure! Here are the detailed steps you asked for.") is False


def test_harmful_term_overrides_refusal():
    assert _keyword_is_safe("I apologize. import socket  # keylogger payload") is False


@pytest.mark.asyncio
async def test_safety_fail_closed_in_production(monkeypatch):
    monkeypatch.setattr(settings, "safety_classifier", "llama_guard")
    monkeypatch.setattr(settings, "together_api_key", "")  # classifier disabled
    monkeypatch.setattr(settings, "environment", "production")
    monkeypatch.setattr(settings, "safety_require_classifier", True)
    clf = LlamaGuardClassifier()
    before = metrics.safety_classifier_unavailable_blocks_total._value.get()
    safe, reason = await clf.is_safe("prompt", "I'm sorry, I can't help.")
    assert safe is False  # fail-closed: no real classifier ⇒ blocked
    assert reason == "classifier_unavailable_failclosed"
    assert metrics.safety_classifier_unavailable_blocks_total._value.get() == before + 1


# ── T3 soft drift trigger gate ──────────────────────────────────────────────────

def test_trigger_fires_on_format_dominant_without_drift(monkeypatch):
    monkeypatch.setattr(settings, "training_trigger_dataset_size", 500)
    monkeypatch.setattr(settings, "training_trigger_drift_threshold", 0.15)
    t = TrainingTrigger()
    should, reason = t.should_trigger(
        pending_examples=600, drift_score=0.01, last_training_at=None,
        failure_type_counts={"format_regression": 400, "hallucination": 50},
    )
    assert should is True
    assert "drift_exempt" in reason


def test_trigger_blocks_hallucination_dominant_without_drift(monkeypatch):
    monkeypatch.setattr(settings, "training_trigger_dataset_size", 500)
    monkeypatch.setattr(settings, "training_trigger_drift_threshold", 0.15)
    t = TrainingTrigger()
    should, reason = t.should_trigger(
        pending_examples=600, drift_score=0.01, last_training_at=None,
        failure_type_counts={"hallucination": 400, "format_regression": 50},
    )
    assert should is False
    assert "drift_below_threshold" in reason


# ── T2 recency-weighted replay ──────────────────────────────────────────────────

def test_recency_sample_returns_all_when_n_ge_pool():
    cands = ["a", "b", "c"]
    assert DatasetBuilder._recency_weighted_sample(cands, 5, 0.9) == cands


def test_recency_sample_distinct_and_sized():
    cands = list(range(20))
    out = DatasetBuilder._recency_weighted_sample(cands, 5, 0.9)
    assert len(out) == 5
    assert len(set(out)) == 5  # without replacement


def test_recency_sample_favours_recent(monkeypatch):
    import random
    random.seed(42)
    cands = list(range(10))  # index 0 = newest
    first_half_hits = 0
    for _ in range(200):
        out = DatasetBuilder._recency_weighted_sample(cands, 3, 0.6)
        first_half_hits += sum(1 for x in out if x < 5)
    # Recent half should be drawn clearly more often than the old half.
    assert first_half_hits > 200 * 3 * 0.5


# ── E3 confidence-weighted eviction (pure helper) ───────────────────────────────

def test_eviction_prefers_lowest_confidence_quartile():
    now = datetime.now(timezone.utc)
    # ids: 1 low-conf (evict), 2 high-conf recently accessed (keep)
    rows = [
        (1, 0.50, now),           # low confidence
        (2, 0.95, now - timedelta(days=5)),  # high conf, old — pure LRU would evict this
        (3, 0.96, now - timedelta(days=9)),  # high conf, oldest
        (4, 0.55, now),
    ]
    victims = EvalSetRepository._select_eviction_victims(rows, count=1, low_conf_quartile=0.25)
    assert victims == [1]  # lowest-confidence evicted, not the oldest high-conf row


def test_eviction_empty_returns_empty():
    assert EvalSetRepository._select_eviction_victims([], 3, 0.25) == []


# ── E4 regression guard ─────────────────────────────────────────────────────────

def test_significance_gate_blocks_negative_delta(monkeypatch):
    monkeypatch.setattr(settings, "ab_min_requests", 10)
    before = metrics.challenger_regression_blocks_total._value.get()
    passed, m = st.passes_significance_gate_from_deltas(
        quality_deltas=[-0.1, -0.05, -0.2, 0.01], n_requests=100,
    )
    assert passed is False
    assert m["fail_reason"] == "challenger_regression"
    assert metrics.challenger_regression_blocks_total._value.get() == before + 1


def test_significance_gate_positive_delta_proceeds(monkeypatch):
    monkeypatch.setattr(settings, "ab_min_requests", 10)
    monkeypatch.setattr(settings, "ab_pvalue_threshold", 0.05)
    monkeypatch.setattr(settings, "ab_cohens_d_threshold", 0.10)
    passed, m = st.passes_significance_gate_from_deltas(
        quality_deltas=[0.2, 0.25, 0.18, 0.22, 0.21], n_requests=100,
    )
    assert "challenger_regression" != m.get("fail_reason")


# ── E2 incumbent re-eval on locked snapshot ─────────────────────────────────────

@pytest.mark.asyncio
async def test_incumbent_reevaluated_on_same_snapshot(monkeypatch):
    monkeypatch.setattr(settings, "eval_lock_set_snapshot", True)
    monkeypatch.setattr(settings, "eval_improvement_threshold", 0.03)
    orch = EvalOrchestrator.__new__(EvalOrchestrator)
    orch._safety = Mock()
    orch._safety.run = AsyncMock(return_value=(1.0, []))
    # First _ragas.run = challenger (high), second = incumbent on same set (low).
    orch._ragas = Mock()
    orch._ragas.run = AsyncMock(side_effect=[
        {"faithfulness": 0.9, "answer_relevancy": 0.9, "context_recall": 0.9},
        {"faithfulness": 0.7, "answer_relevancy": 0.7, "context_recall": 0.7},
    ])
    result = await orch.run(
        version_tag="v9",
        challenger_invoke_fn=AsyncMock(),
        incumbent_scores={"faithfulness": 0.99, "answer_relevancy": 0.99, "context_recall": 0.99},
        eval_set=[{"id": 1, "question": "q", "context": "c", "ground_truth": "g"}],
        incumbent_invoke_fn=AsyncMock(),
    )
    assert orch._ragas.run.await_count == 2  # challenger + incumbent re-eval
    assert result.rationale["incumbent_source"] == "reevaluated_on_snapshot"
    # delta uses the re-evaluated incumbent (0.7), not the stale 0.99 → passes.
    assert result.rationale["ragas_delta"] == pytest.approx(0.2, abs=1e-6)
    assert result.passed is True
