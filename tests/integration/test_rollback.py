"""
Integration test: rollback scenario — promotion gate fails, rollback executes.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.shadow.promotion_gate import PromotionGate
from src.evaluation.eval_orchestrator import EvalResult


@pytest.fixture
def gate():
    return PromotionGate()


def test_gate_rejects_on_safety_regression(gate):
    eval_result = EvalResult(
        version_tag="v8",
        passed=False,
        safety_score=0.95,   # not 1.0
        faithfulness=0.85,
        answer_relevancy=0.80,
        context_recall=0.75,
    )
    decision = gate.evaluate(
        ab_data={"ready": True, "n_requests": 2000, "quality_deltas": [0.05] * 2000, "elapsed_hours": 50},
        eval_result=eval_result,
        incumbent_scores={"faithfulness": 0.70, "answer_relevancy": 0.72, "context_recall": 0.68},
    )
    assert decision.promote is False
    assert "safety" in decision.reason


def test_gate_rejects_incomplete_ab_window(gate):
    eval_result = EvalResult(
        version_tag="v8",
        passed=True,
        safety_score=1.0,
        faithfulness=0.85,
        answer_relevancy=0.80,
        context_recall=0.75,
    )
    decision = gate.evaluate(
        ab_data={"ready": False, "n_requests": 100, "quality_deltas": [], "elapsed_hours": 5},
        eval_result=eval_result,
        incumbent_scores={},
    )
    assert decision.promote is False
    assert "ab_window" in decision.reason


def test_gate_promotes_when_all_pass(gate):
    eval_result = EvalResult(
        version_tag="v8",
        passed=True,
        safety_score=1.0,
        faithfulness=0.85,
        answer_relevancy=0.82,
        context_recall=0.78,
    )
    # Create enough quality deltas to pass significance test
    deltas = [0.15] * 1500   # strongly positive delta
    decision = gate.evaluate(
        ab_data={
            "ready": True,
            "n_requests": 1500,
            "quality_deltas": deltas,
            "elapsed_hours": 50,
        },
        eval_result=eval_result,
        incumbent_scores={"faithfulness": 0.70, "answer_relevancy": 0.70, "context_recall": 0.70},
    )
    assert decision.promote is True
    assert decision.reason == "all_gates_passed"
