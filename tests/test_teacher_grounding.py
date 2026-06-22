"""Unit tests for RAG-grounded teacher corrections. NLI + LLM are mocked."""

from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from src.curation.teacher import TeacherModel, GroundingResult
from src.curation.curator import CurationPipeline
from src.detection.failure_classifier import FailureEvent
from src.monitoring import metrics

CONTEXT = (
    "[doc_id: policy-42] Refunds are available within 30 days of delivery, "
    "provided the item is unused and in its original packaging."
)


def _failure(context: str | None = None, log_id: str = "unknown") -> FailureEvent:
    meta = {"retrieved_context": context} if context else {}
    return FailureEvent(
        llm_log_id=log_id,
        prompt="What is the refund window?",
        completion="Refunds are available for 90 days.",
        failure_type="hallucination",
        score=0.8,
        metadata=meta,
    )


def _teacher(correction: str, hallucination_prob: float | None) -> TeacherModel:
    t = TeacherModel()
    # Bypass real OpenAI calls — every vote returns the same correction.
    t._single_correction = AsyncMock(return_value=correction)
    # Avoid loading MiniLM in unit tests: identical votes → identical embeddings
    # → cosine consistency 1.0 (above threshold), matching the old behaviour.
    t._embed = lambda texts: np.ones((len(texts), 8))
    if hallucination_prob is not None:
        det = Mock()
        det.score_batch = AsyncMock(return_value=[hallucination_prob])
        t._hall_detector = det
    return t


@pytest.mark.asyncio
async def test_grounded_correction_accepted():
    t = _teacher("Refunds are available within 30 days.", hallucination_prob=0.1)
    result = await t.generate_correction(_failure(CONTEXT))
    assert isinstance(result, GroundingResult)
    assert result.grounding_score == pytest.approx(0.9, abs=1e-6)
    assert result.is_grounded is True
    assert result.grounding_sources  # at least one source id extracted


@pytest.mark.asyncio
async def test_ungrounded_correction_dropped():
    before = metrics.teacher_corrections_rejected_grounding_total._value.get()
    t = _teacher("Refunds are available for 90 days.", hallucination_prob=0.8)
    result = await t.generate_correction(_failure(CONTEXT))
    assert result is None
    after = metrics.teacher_corrections_rejected_grounding_total._value.get()
    assert after == before + 1


@pytest.mark.asyncio
async def test_no_context_is_valid_without_grounding():
    t = _teacher("Refunds are available within 30 days.", hallucination_prob=None)
    # Pin the retrieval fallback to "no match" so this unit test is deterministic
    # regardless of what's in the dev knowledge base.
    t._retrieve_context = AsyncMock(return_value=None)
    result = await t.generate_correction(_failure(context=None))
    assert isinstance(result, GroundingResult)
    assert result.grounding_score is None
    assert result.grounding_sources == []
    assert result.is_grounded is True


@pytest.mark.asyncio
async def test_insufficient_context_dropped():
    t = _teacher("INSUFFICIENT_CONTEXT", hallucination_prob=0.1)
    result = await t.generate_correction(_failure(CONTEXT))
    assert result is None


def test_extract_source_ids_structured_and_fallback():
    t = TeacherModel()
    ids = t._extract_source_ids("[doc_id: D1] source: kb-7 chunk_id: c99")
    assert "D1" in ids and "kb-7" in ids and "c99" in ids

    fallback = t._extract_source_ids("just some unstructured text with no ids")
    assert len(fallback) == 1
    assert fallback[0].startswith("context_hash:")


@pytest.mark.asyncio
async def test_grounding_fields_flow_to_insert_dict():
    # Prove the curator carries grounding metadata into the row it upserts,
    # without standing up the DB or heavy curation models.
    pipe = CurationPipeline.__new__(CurationPipeline)
    pipe._teacher = Mock()
    pipe._teacher.generate_correction = AsyncMock(return_value=GroundingResult(
        correction="Refunds within 30 days.", confidence=0.88,
        grounding_score=0.91, grounding_sources=["policy-42"], is_grounded=True,
    ))
    pipe._pii = Mock()
    pipe._pii.scrub_example = Mock(return_value=("q", "Refunds within 30 days.", True))
    pipe._pii.scrub = Mock(return_value=("Refunds within 30 days.", False))
    pipe._dedup = Mock()
    pipe._dedup.compute_hash = Mock(return_value="hash1")
    pipe._dedup.is_duplicate = Mock(return_value=False)
    pipe._quality = Mock()
    pipe._quality.passes = Mock(return_value=(True, 0.7, "ok"))

    row = await pipe._process_failure(_failure(CONTEXT, log_id="unknown"))
    assert row is not None
    assert row["grounding_score"] == 0.91
    assert row["grounding_sources"] == ["policy-42"]
