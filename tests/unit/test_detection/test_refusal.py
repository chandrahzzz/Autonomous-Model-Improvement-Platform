"""Unit tests for RefusalDetector."""

import pytest
from src.detection.refusal import RefusalDetector


@pytest.fixture
def detector():
    return RefusalDetector()


def test_keyword_refusal_detected(detector):
    text = "I'm sorry, I can't help with that request."
    is_ref, score = detector.is_refusal(text)
    assert is_ref is True
    assert score == 1.0


def test_normal_response_not_refusal(detector):
    text = "The capital of France is Paris, which has a population of about 2 million."
    is_ref, score = detector.is_refusal(text)
    assert is_ref is False


def test_refusal_rate_tracking(detector):
    refusals = ["I can't help with that.", "I won't do that.", "I am unable to assist."]
    normals = ["Paris is in France.", "The answer is 42.", "Water is H2O."]

    for text in refusals:
        detector.is_refusal(text)
    for text in normals:
        detector.is_refusal(text)

    assert detector.current_refusal_rate == pytest.approx(0.5, abs=0.1)


def test_i_cannot_pattern(detector):
    is_ref, _ = detector.is_refusal("I cannot provide that information.")
    assert is_ref is True


def test_as_an_ai_pattern(detector):
    is_ref, _ = detector.is_refusal("As an AI, I don't have the ability to do that.")
    assert is_ref is True


@pytest.mark.asyncio
async def test_classify_batch(detector):
    events = [
        {"completion": "I can't help with that."},
        {"completion": "The answer is Paris."},
        {"completion": "I won't provide that."},
    ]
    results = await detector.classify_batch(events)
    assert len(results) == 3
    types = [r[0] for r in results]
    assert types[0] == "refusal_creep"
    assert types[1] == ""
    assert types[2] == "refusal_creep"
