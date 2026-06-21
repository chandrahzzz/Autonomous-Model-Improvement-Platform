"""Unit tests for failure attribution (RFC-003). No real I/O or models."""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.attribution.influence import EmbeddingInfluenceBackend, InfluenceBackend
from src.attribution.attributor import FailureAttributor


def make_mock_backend(scores: list[float]) -> MagicMock:
    backend = MagicMock(spec=InfluenceBackend)
    backend.score.return_value = scores
    backend.backend_name = "mock_backend"
    return backend


def make_failure_event(log_id="00000000-0000-0000-0000-000000000001"):
    f = MagicMock()
    f.log_id = log_id
    f.prompt = "Can I carry lithium batteries?"
    f.completion = "Yes, unlimited batteries are allowed."
    f.failure_type = "hallucination"
    f.classification_id = None
    return f


def make_candidate(i: int, quality_score: float = 0.8) -> MagicMock:
    c = MagicMock()
    c.id = f"cand-{i:04d}"
    c.prompt = f"Candidate prompt {i}"
    c.corrected_completion = f"Correct answer {i}"
    c.failure_type = "hallucination"
    c.quality_score = quality_score
    return c


def _patches(model_repo, te_repo, attr_repo):
    return (
        patch("src.attribution.attributor.ModelRepository", return_value=model_repo),
        patch("src.attribution.attributor.TrainingExampleRepository", return_value=te_repo),
        patch("src.attribution.attributor.FailureAttributionRepository", return_value=attr_repo),
    )


# ── EmbeddingInfluenceBackend ────────────────────────────────────────────────

def test_embedding_backend_score_length():
    backend = EmbeddingInfluenceBackend.__new__(EmbeddingInfluenceBackend)
    enc = MagicMock()
    embs = np.random.rand(6, 384).astype(np.float32)
    embs = embs / np.linalg.norm(embs, axis=1, keepdims=True)
    enc.encode.return_value = embs
    backend._encoder = enc
    assert len(backend.score("failure text", ["cand"] * 5)) == 5


def test_embedding_backend_score_ordering():
    backend = EmbeddingInfluenceBackend.__new__(EmbeddingInfluenceBackend)
    enc = MagicMock()

    def norm(v):
        return v / np.linalg.norm(v)

    embs = np.vstack([
        norm(np.array([1.0, 0.0, 0.0])),
        norm(np.array([0.99, 0.14, 0.0])),
        norm(np.array([0.0, 1.0, 0.0])),
    ])
    enc.encode.return_value = embs
    backend._encoder = enc
    scores = backend.score("failure", ["close", "far"])
    assert scores[0] > scores[1]


def test_embedding_backend_name():
    backend = EmbeddingInfluenceBackend.__new__(EmbeddingInfluenceBackend)
    backend._encoder = MagicMock()
    assert backend.backend_name == "embedding_cosine"


def test_embedding_backend_implements_protocol():
    backend = EmbeddingInfluenceBackend.__new__(EmbeddingInfluenceBackend)
    backend._encoder = MagicMock()
    assert isinstance(backend, InfluenceBackend)


# ── FailureAttributor ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_attributor_no_candidates_returns_false():
    attributor = FailureAttributor(backend=make_mock_backend([]))
    model_repo = AsyncMock()
    model_repo.get_production_version.return_value = MagicMock(version_tag="v8")
    model_repo.get_training_run_for_version.return_value = 1
    te_repo = AsyncMock(); te_repo.get_used_for_run.return_value = []
    attr_repo = AsyncMock()
    p1, p2, p3 = _patches(model_repo, te_repo, attr_repo)
    with p1, p2, p3:
        result = await attributor.attribute(make_failure_event(), db=AsyncMock())
    assert result is False
    attr_repo.insert.assert_not_called()


@pytest.mark.asyncio
async def test_attributor_no_production_model_returns_false():
    attributor = FailureAttributor(backend=make_mock_backend([]))
    model_repo = AsyncMock(); model_repo.get_production_version.return_value = None
    with patch("src.attribution.attributor.ModelRepository", return_value=model_repo):
        result = await attributor.attribute(make_failure_event(), db=AsyncMock())
    assert result is False


@pytest.mark.asyncio
async def test_attributor_top_k_sorted():
    scores = [0.5, 0.9, 0.3, 0.7, 0.8]
    attributor = FailureAttributor(backend=make_mock_backend(scores))
    candidates = [make_candidate(i) for i in range(5)]
    model_repo = AsyncMock()
    model_repo.get_production_version.return_value = MagicMock(version_tag="v8")
    model_repo.get_training_run_for_version.return_value = 1
    te_repo = AsyncMock(); te_repo.get_used_for_run.return_value = candidates
    attr_repo = AsyncMock()
    p1, p2, p3 = _patches(model_repo, te_repo, attr_repo)
    with p1, p2, p3, patch("src.attribution.attributor.settings") as s:
        s.attribution_top_k = 3
        s.attribution_max_candidates = 500
        result = await attributor.attribute(make_failure_event(), db=AsyncMock())
    assert result is True
    top_k = attr_repo.insert.call_args[0][0]["top_k_examples"]
    assert len(top_k) == 3
    influence = [t["influence_score"] for t in top_k]
    assert influence == sorted(influence, reverse=True)
    assert influence[0] == pytest.approx(0.9, abs=0.01)


@pytest.mark.asyncio
async def test_attributor_backend_name_stored():
    backend = make_mock_backend([0.8] * 3); backend.backend_name = "test_backend"
    attributor = FailureAttributor(backend=backend)
    candidates = [make_candidate(i) for i in range(3)]
    model_repo = AsyncMock()
    model_repo.get_production_version.return_value = MagicMock(version_tag="v8")
    model_repo.get_training_run_for_version.return_value = 1
    te_repo = AsyncMock(); te_repo.get_used_for_run.return_value = candidates
    attr_repo = AsyncMock()
    p1, p2, p3 = _patches(model_repo, te_repo, attr_repo)
    with p1, p2, p3, patch("src.attribution.attributor.settings") as s:
        s.attribution_top_k = 10
        s.attribution_max_candidates = 500
        await attributor.attribute(make_failure_event(), db=AsyncMock())
    assert attr_repo.insert.call_args[0][0]["backend_used"] == "test_backend"


@pytest.mark.asyncio
async def test_attributor_all_zero_scores_returns_false():
    attributor = FailureAttributor(backend=make_mock_backend([0.0, 0.0, 0.0]))
    candidates = [make_candidate(i) for i in range(3)]
    model_repo = AsyncMock()
    model_repo.get_production_version.return_value = MagicMock(version_tag="v8")
    model_repo.get_training_run_for_version.return_value = 1
    te_repo = AsyncMock(); te_repo.get_used_for_run.return_value = candidates
    attr_repo = AsyncMock()
    p1, p2, p3 = _patches(model_repo, te_repo, attr_repo)
    with p1, p2, p3, patch("src.attribution.attributor.settings") as s:
        s.attribution_top_k = 10
        s.attribution_max_candidates = 500
        result = await attributor.attribute(make_failure_event(), db=AsyncMock())
    assert result is False
    attr_repo.insert.assert_not_called()


@pytest.mark.asyncio
async def test_attributor_backend_exception_returns_false():
    backend = MagicMock(spec=InfluenceBackend)
    backend.score.side_effect = RuntimeError("GPU OOM")
    backend.backend_name = "failing_backend"
    attributor = FailureAttributor(backend=backend)
    candidates = [make_candidate(0)]
    model_repo = AsyncMock()
    model_repo.get_production_version.return_value = MagicMock(version_tag="v8")
    model_repo.get_training_run_for_version.return_value = 1
    te_repo = AsyncMock(); te_repo.get_used_for_run.return_value = candidates
    attr_repo = AsyncMock()
    p1, p2, p3 = _patches(model_repo, te_repo, attr_repo)
    with p1, p2, p3, patch("src.attribution.attributor.settings") as s:
        s.attribution_top_k = 10
        s.attribution_max_candidates = 500
        result = await attributor.attribute(make_failure_event(), db=AsyncMock())
    assert isinstance(result, bool)
    assert result is False


@pytest.mark.asyncio
async def test_attributor_no_training_run_returns_false():
    attributor = FailureAttributor(backend=make_mock_backend([0.9]))
    model_repo = AsyncMock()
    model_repo.get_production_version.return_value = MagicMock(version_tag="v7")
    model_repo.get_training_run_for_version.return_value = None
    with patch("src.attribution.attributor.ModelRepository", return_value=model_repo):
        result = await attributor.attribute(make_failure_event(), db=AsyncMock())
    assert result is False
