"""Unit tests for TrainingTrigger."""

import pytest
from datetime import datetime, timedelta
from src.training.trigger import TrainingTrigger


@pytest.fixture
def trigger():
    return TrainingTrigger()


def test_fires_when_all_conditions_met(trigger):
    should, reason = trigger.should_trigger(
        pending_examples=600,
        drift_score=0.20,
        last_training_at=datetime.utcnow() - timedelta(hours=10),
    )
    assert should is True
    assert "triggered" in reason


def test_no_fire_insufficient_examples(trigger):
    should, reason = trigger.should_trigger(
        pending_examples=100,
        drift_score=0.20,
        last_training_at=None,
    )
    assert should is False
    assert "insufficient_data" in reason


def test_no_fire_low_drift(trigger):
    should, reason = trigger.should_trigger(
        pending_examples=600,
        drift_score=0.05,
        last_training_at=None,
    )
    assert should is False
    assert "drift_below_threshold" in reason


def test_no_fire_in_cooldown(trigger):
    should, reason = trigger.should_trigger(
        pending_examples=600,
        drift_score=0.20,
        last_training_at=datetime.utcnow() - timedelta(hours=1),
    )
    assert should is False
    assert "cooldown" in reason


def test_fires_when_no_previous_run(trigger):
    should, _ = trigger.should_trigger(
        pending_examples=600,
        drift_score=0.20,
        last_training_at=None,
    )
    assert should is True
