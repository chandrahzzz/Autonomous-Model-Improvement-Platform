"""Unit tests for statistical significance tests."""

import pytest
from src.evaluation.statistical_tests import welch_t_test, cohens_d, passes_significance_gate


def test_welch_t_test_significant():
    prod = [0.5, 0.52, 0.48, 0.51, 0.49] * 20
    chal = [0.7, 0.72, 0.68, 0.71, 0.69] * 20
    _, p = welch_t_test(prod, chal)
    assert p < 0.05


def test_welch_t_test_not_significant():
    prod = [0.5, 0.5, 0.5, 0.5, 0.5]
    chal = [0.5, 0.5, 0.5, 0.5, 0.5]
    _, p = welch_t_test(prod, chal)
    assert p >= 0.05 or p == 1.0


def test_cohens_d_positive():
    prod = [0.5] * 50
    chal = [0.7] * 50
    d = cohens_d(prod, chal)
    assert d > 0


def test_cohens_d_zero_variance():
    prod = [0.5] * 10
    chal = [0.5] * 10
    d = cohens_d(prod, chal)
    assert d == 0.0


def test_passes_gate_insufficient_requests():
    passes, metrics = passes_significance_gate([0.5] * 10, [0.7] * 10, n_requests=100)
    assert passes is False
    assert "insufficient_requests" in metrics["fail_reason"]


def test_passes_gate_strong_improvement():
    prod = [0.5] * 500
    chal = [0.8] * 500
    passes, metrics = passes_significance_gate(prod, chal, n_requests=1500)
    assert passes is True
    assert metrics["cohens_d"] > 0


def test_fails_gate_no_improvement():
    prod = [0.7] * 500
    chal = [0.5] * 500   # challenger is worse
    passes, _ = passes_significance_gate(prod, chal, n_requests=1500)
    assert passes is False
