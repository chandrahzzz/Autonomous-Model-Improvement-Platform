"""Unit tests for FormatValidator."""

import pytest
from src.detection.format_validator import FormatValidator


@pytest.fixture
def validator():
    v = FormatValidator()
    return v


def test_json_validation_pass(validator):
    validator.set_expect_json(True)
    is_reg, score = validator.score('{"key": "value", "num": 42}')
    assert is_reg is False


def test_json_validation_fail(validator):
    validator.set_expect_json(True)
    is_reg, score = validator.score("This is plain text, not JSON.")
    assert is_reg is True
    assert score == 1.0


def test_no_json_expectation(validator):
    validator.set_expect_json(False)
    is_reg, _ = validator.score("This is plain text.")
    assert is_reg is False


def test_length_baseline(validator):
    baseline_lengths = [50 + i % 20 for i in range(500)]
    validator.set_baseline_lengths(baseline_lengths)
    is_reg, kl = validator.score("A medium length response with about ten words here.")
    assert isinstance(kl, float)


@pytest.mark.asyncio
async def test_validate_batch(validator):
    validator.set_expect_json(True)
    events = [
        {"completion": '{"result": "ok"}'},
        {"completion": "not json at all"},
    ]
    results = await validator.validate_batch(events)
    assert len(results) == 2
    assert results[0][0] == ""
    assert results[1][0] == "format_regression"
