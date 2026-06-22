"""Unit tests for the June 2026 curation-layer hardening.

No models or network: the teacher's embedder is injected, OpenAI errors are
simulated via a patched retryable-error tuple, and the deduplicator/clusterer
helpers are exercised directly.
"""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from src.config.settings import settings
from src.curation import teacher as teacher_mod
from src.curation.teacher import TeacherModel, GroundingResult, _mean_pairwise_cosine
from src.curation.curator import CurationPipeline
from src.curation.deduplicator import Deduplicator
from src.curation.clustering import _effective_min_cluster_size, FailureClusterer
from src.detection.failure_classifier import FailureEvent
from src.monitoring import metrics


def _failure(prompt="q", completion="bad", log_id="unknown"):
    return FailureEvent(llm_log_id=log_id, prompt=prompt, completion=completion,
                        failure_type="hallucination", score=0.8)


# ── #1 semantic consistency ────────────────────────────────────────────────────

def test_mean_pairwise_cosine_identical_is_one():
    embs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    assert _mean_pairwise_cosine(embs) == pytest.approx(1.0)


def test_mean_pairwise_cosine_orthogonal_is_zero():
    embs = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert _mean_pairwise_cosine(embs) == pytest.approx(0.0, abs=1e-6)


def test_paraphrases_pass_but_divergent_fail():
    t = TeacherModel.__new__(TeacherModel)
    # Paraphrases: near-parallel vectors → high cosine, above 0.80 threshold.
    para = np.array([[1.0, 0.05], [0.98, 0.1], [1.0, 0.08]])
    assert t._compute_consistency_score(["a", "b", "c"], para) > 0.9
    # Divergent meanings: spread-out vectors → low cosine.
    diverge = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert t._compute_consistency_score(["a", "b", "c"], diverge) < 0.5


def test_pick_best_returns_centroid_nearest():
    t = TeacherModel.__new__(TeacherModel)
    corrections = ["outlier", "central1", "central2"]
    embs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])  # last two cluster
    assert t._pick_best(corrections, embs) in ("central1", "central2")


def test_consistency_falls_back_to_rouge_without_embeddings():
    t = TeacherModel.__new__(TeacherModel)
    # No embeddings → ROUGE-L path; identical strings → 1.0.
    assert t._compute_consistency_score(["same text", "same text"], None) == pytest.approx(1.0)


# ── #3 rate-limit backoff ───────────────────────────────────────────────────────

class _FakeRateLimit(Exception):
    pass


def _backoff_teacher(monkeypatch):
    monkeypatch.setattr(teacher_mod, "_RETRYABLE_ERRORS", (_FakeRateLimit,))
    monkeypatch.setattr(settings, "teacher_max_retries", 3)
    monkeypatch.setattr(settings, "teacher_retry_base_delay_seconds", 0.0)
    monkeypatch.setattr(settings, "teacher_retry_max_delay_seconds", 0.0)
    t = TeacherModel.__new__(TeacherModel)
    import asyncio
    t._semaphore = asyncio.Semaphore(1)
    t._total_cost_usd = 0.0
    return t


@pytest.mark.asyncio
async def test_backoff_retries_then_succeeds(monkeypatch):
    t = _backoff_teacher(monkeypatch)
    before = metrics.teacher_rate_limit_retries_total._value.get()
    calls = {"n": 0}

    async def flaky(_messages):
        calls["n"] += 1
        if calls["n"] < 3:
            raise _FakeRateLimit("429")
        return Mock(content="recovered answer")

    t._llm = Mock(ainvoke=flaky)
    out = await t._single_correction("prompt", use_sampler=False)
    assert out == "recovered answer"
    assert calls["n"] == 3
    assert metrics.teacher_rate_limit_retries_total._value.get() == before + 2


@pytest.mark.asyncio
async def test_backoff_exhausts_and_drops(monkeypatch):
    t = _backoff_teacher(monkeypatch)
    before = metrics.teacher_dropped_rate_limited_total._value.get()

    async def always_429(_messages):
        raise _FakeRateLimit("429")

    t._llm = Mock(ainvoke=always_429)
    out = await t._single_correction("prompt", use_sampler=False)
    assert out is None
    assert metrics.teacher_dropped_rate_limited_total._value.get() == before + 1


@pytest.mark.asyncio
async def test_non_retryable_error_returns_none_immediately(monkeypatch):
    t = _backoff_teacher(monkeypatch)

    async def boom(_messages):
        raise ValueError("not retryable")

    t._llm = Mock(ainvoke=boom)
    assert await t._single_correction("prompt") is None


# ── #2 PII scrubbed BEFORE the teacher ─────────────────────────────────────────

def _pipeline_with_mocks():
    pipe = CurationPipeline.__new__(CurationPipeline)
    pipe._pii = Mock()
    pipe._teacher = Mock()
    pipe._dedup = Mock()
    pipe._dedup.compute_hash = Mock(return_value="h1")
    pipe._dedup.is_duplicate = Mock(return_value=False)
    pipe._quality = Mock()
    pipe._quality.passes = Mock(return_value=(True, 0.7, "ok"))
    return pipe


@pytest.mark.asyncio
async def test_teacher_receives_scrubbed_inputs_not_raw_pii():
    pipe = _pipeline_with_mocks()
    # PII scrubber replaces the SSN in prompt and the name in the bad completion.
    pipe._pii.scrub_example = Mock(return_value=("ssn <US_SSN>", "I'm <PERSON>", True))
    pipe._pii.scrub = Mock(return_value=("clean correction", True))
    seen = {}

    async def capture(failure):
        seen["prompt"] = failure.prompt
        seen["completion"] = failure.completion
        return GroundingResult(correction="clean correction", confidence=0.9,
                               grounding_score=None, grounding_sources=[])

    pipe._teacher.generate_correction = capture
    row = await pipe._process_failure(_failure(prompt="ssn 123-45-6789",
                                               completion="I'm John Doe"))
    # Teacher must have been handed the SCRUBBED text, never the raw PII.
    assert seen["prompt"] == "ssn <US_SSN>"
    assert seen["completion"] == "I'm <PERSON>"
    assert "123-45-6789" not in seen["prompt"]
    assert row["bad_completion"] == "I'm <PERSON>"  # scrubbed stored, not raw


@pytest.mark.asyncio
async def test_failed_prescrub_drops_before_teacher_call():
    pipe = _pipeline_with_mocks()
    pipe._pii.scrub_example = Mock(return_value=("", "", False))  # fail-closed
    pipe._teacher.generate_correction = AsyncMock()
    before = metrics.examples_dropped_pre_scrub_total._value.get()
    row = await pipe._process_failure(_failure())
    assert row is None
    pipe._teacher.generate_correction.assert_not_called()  # never reached the API
    assert metrics.examples_dropped_pre_scrub_total._value.get() == before + 1


# ── #4 dynamic clustering ───────────────────────────────────────────────────────

def test_effective_min_cluster_size_scales():
    assert _effective_min_cluster_size(2) == 2
    assert _effective_min_cluster_size(3) == 2
    assert _effective_min_cluster_size(6) == 2
    assert _effective_min_cluster_size(9) == 3
    assert _effective_min_cluster_size(100) == 5  # capped


def test_cluster_single_failure_bypasses():
    clf = FailureClusterer.__new__(FailureClusterer)  # skip model load
    before = metrics.clustering_bypassed_total._value.get()
    failures = [_failure()]
    out = clf.cluster(failures)
    assert out[0].metadata["cluster_id"] == -1
    assert metrics.clustering_bypassed_total._value.get() == before + 1


# ── #5 dedup rehydration ────────────────────────────────────────────────────────

def test_preload_then_exact_duplicate_caught():
    d = Deduplicator()
    n = d.preload([("what is the refund window", "30 days from delivery")])
    assert n == 1
    assert d.is_duplicate("what is the refund window", "30 days from delivery") is True


def test_preload_then_near_duplicate_caught():
    d = Deduplicator()
    d.preload([("how long is the refund window for orders",
                "Refunds are available within thirty days of delivery date")])
    # >85% word overlap — a near-duplicate that the DB exact hash would miss.
    near = d.is_duplicate("how long is the refund window for orders",
                          "Refunds are available within thirty days of the delivery date")
    assert near is True


def test_preload_skips_none_and_is_idempotent():
    d = Deduplicator()
    d.preload([("p", "c"), (None, "c"), ("p2", None)])
    assert d.size == 1
    d.preload([("p", "c")])  # already seen
    assert d.size == 1
