"""Unit tests for the NLI-based hallucination detector.

These load the NLI model, so they are comparatively slow. Bounds are lenient to
stay robust across model versions — the key property is grounded << hallucinated.
"""

import pytest

from src.detection.hallucination import HallucinationDetector


@pytest.fixture(scope="module")
def detector():
    return HallucinationDetector()


@pytest.mark.asyncio
async def test_grounded_completion_scores_low(detector):
    ctx = "Paris is the capital of France."
    _, score = await detector.is_hallucination(ctx, "The capital of France is Paris.")
    assert score < 0.5


@pytest.mark.asyncio
async def test_contradicted_completion_scores_high(detector):
    ctx = "Paris is the capital of France."
    _, score = await detector.is_hallucination(ctx, "London is the capital of France.")
    assert score > 0.5


@pytest.mark.asyncio
async def test_grounded_below_contradicted(detector):
    ctx = "Paris is the capital of France."
    _, grounded = await detector.is_hallucination(ctx, "The capital of France is Paris.")
    _, halluc = await detector.is_hallucination(ctx, "London is the capital of France.")
    assert grounded < halluc
