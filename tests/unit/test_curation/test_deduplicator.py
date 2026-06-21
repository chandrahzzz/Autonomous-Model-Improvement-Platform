"""Unit tests for Deduplicator."""

import pytest
from src.curation.deduplicator import Deduplicator


@pytest.fixture
def dedup():
    return Deduplicator()


def test_first_occurrence_not_duplicate(dedup):
    assert dedup.is_duplicate("What is AI?", "AI stands for Artificial Intelligence.") is False


def test_exact_duplicate_detected(dedup):
    p = "What is AI?"
    c = "AI stands for Artificial Intelligence."
    dedup.is_duplicate(p, c)   # register first
    assert dedup.is_duplicate(p, c) is True


def test_different_pairs_not_duplicate(dedup):
    assert dedup.is_duplicate("Q1", "A1") is False
    assert dedup.is_duplicate("Q2", "A2") is False


def test_hash_deterministic(dedup):
    h1 = dedup.compute_hash("prompt", "completion")
    h2 = dedup.compute_hash("prompt", "completion")
    assert h1 == h2


def test_hash_different_for_different_inputs(dedup):
    h1 = dedup.compute_hash("prompt A", "completion A")
    h2 = dedup.compute_hash("prompt B", "completion B")
    assert h1 != h2
