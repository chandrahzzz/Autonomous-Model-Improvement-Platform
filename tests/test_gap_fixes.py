"""
Tests for two gaps found by running the pipeline against real data:

  Gap 1 — failure_classifications were never persisted, so bad logs counted as
          "known-good" in the replay filter. Now the detector writes them and the
          curator links training_examples.failure_id.
  Gap 2 — the drift baseline covariance was ill-conditioned on low-diversity text
          (near-singular → Mahalanobis exploded → drift fired on everything). Now
          the covariance is shrunk/floored so distances stay sane.
"""

from uuid import uuid4
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from scipy.spatial.distance import mahalanobis

from src.detection.drift import _condition_covariance
from src.db.repositories.failure_classifications import FailureClassificationRepository
from src.graph.nodes.failure_detector import _looks_like_uuid


# ── Gap 2: drift covariance conditioning ──────────────────────────────────────
def _near_singular_data(n=200, d=20, rank=2, seed=0):
    """Rank-`rank` data in d dims → near-singular covariance (like templated text
    whose embeddings live in a tiny subspace)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, rank)) @ rng.normal(size=(rank, d))


def test_condition_covariance_floors_the_diagonal():
    X = _near_singular_data()
    cov = np.cov(X.T)
    conditioned = _condition_covariance(cov, shrinkage=0.1)
    mean_var = np.trace(cov) / cov.shape[0]
    assert np.min(np.diag(conditioned)) >= min(mean_var * 1e-3, 1e-6) - 1e-12
    assert conditioned.shape == cov.shape


def test_conditioning_prevents_mahalanobis_explosion():
    # A rank-2 baseline (like templated text living in a tiny subspace). The
    # explosion happens for an OFF-distribution point that has components in the
    # near-zero-variance directions — exactly what a drifted completion looks like.
    rng = np.random.default_rng(1)
    X = _near_singular_data(seed=1)
    cov = np.cov(X.T)
    centroid = X.mean(axis=0)
    d = cov.shape[0]
    off_point = centroid + rng.normal(size=d)  # nonzero in the null directions

    # Old behaviour: tiny 1e-6 regularization → near-zero-variance dims invert to
    # ~1e6 → an off-subspace point's distance explodes.
    old_inv = np.linalg.inv(cov + np.eye(d) * 1e-6)
    old_dist = mahalanobis(off_point, centroid, old_inv)

    # New behaviour: shrink toward scaled identity + floor + pinv.
    new_dist = mahalanobis(off_point, centroid, np.linalg.pinv(_condition_covariance(cov, 0.1)))

    assert old_dist > 50, "sanity: the old path really does explode off-distribution"
    assert np.isfinite(new_dist)
    assert new_dist < old_dist / 10, "conditioning must dramatically tame the distance"


def test_shrinkage_zero_is_still_floored():
    # Even with no shrinkage, the variance floor alone keeps the matrix invertible.
    X = _near_singular_data()
    cov = np.cov(X.T)
    conditioned = _condition_covariance(cov, shrinkage=0.0)
    inv = np.linalg.pinv(conditioned)
    assert np.all(np.isfinite(inv))


# ── Gap 1: failure classification persistence ─────────────────────────────────
def test_looks_like_uuid():
    assert _looks_like_uuid(str(uuid4()))
    assert not _looks_like_uuid("unknown")
    assert not _looks_like_uuid("unknown_3")
    assert not _looks_like_uuid(None)
    assert not _looks_like_uuid("")


@pytest.mark.asyncio
async def test_existing_for_logs_empty_returns_empty():
    repo = FailureClassificationRepository(MagicMock())
    assert await repo.existing_for_logs([]) == {}


@pytest.mark.asyncio
async def test_existing_for_logs_maps_ids():
    lid1, cid1, lid2, cid2 = uuid4(), uuid4(), uuid4(), uuid4()
    db = MagicMock()
    result = MagicMock()
    result.fetchall = lambda: [(lid1, cid1), (lid2, cid2)]
    db.execute = AsyncMock(return_value=result)
    repo = FailureClassificationRepository(db)
    out = await repo.existing_for_logs([str(lid1), str(lid2)])
    assert out == {str(lid1): str(cid1), str(lid2): str(cid2)}


@pytest.mark.asyncio
async def test_insert_adds_failure_row():
    db = MagicMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    repo = FailureClassificationRepository(db)
    await repo.insert({
        "llm_log_id": str(uuid4()),
        "failure_type": "refusal_creep",
        "score": 0.92,
        "cluster_id": None,
        "cluster_label": None,
        "metadata_": {"is_correlated": False},
    })
    assert db.add.call_count == 1
    added = db.add.call_args[0][0]
    assert added.failure_type == "refusal_creep"
    assert added.score == 0.92
    db.flush.assert_awaited_once()


@pytest.mark.asyncio
async def test_curator_process_failure_links_failure_id():
    """The real curator._process_failure must copy the persisted classification
    id into failure_id and stamp the current teacher model (not hardcoded gpt-4o),
    so lineage (training_example → failure → log) links up."""
    from src.curation.curator import CurationPipeline
    from src.curation.teacher import GroundingResult
    from src.detection.failure_classifier import FailureEvent
    from src.config.settings import settings

    c = CurationPipeline.__new__(CurationPipeline)  # skip heavy __init__
    c._pii = MagicMock()
    c._pii.scrub_example = MagicMock(return_value=("clean q", "clean bad", True))
    c._pii.scrub = MagicMock(return_value=("clean corrected", True))
    c._teacher = MagicMock()
    c._teacher.generate_correction = AsyncMock(return_value=GroundingResult(
        correction="clean corrected", confidence=0.9,
        grounding_score=None, grounding_sources=[],
    ))
    c._dedup = MagicMock()
    c._dedup.compute_hash = MagicMock(return_value="hash-1")
    c._dedup.is_duplicate = MagicMock(return_value=False)
    c._quality = MagicMock()
    c._quality.passes = MagicMock(return_value=(True, 0.88, "ok"))

    fe = FailureEvent(
        llm_log_id=str(uuid4()), prompt="q", completion="bad",
        failure_type="hallucination", score=0.8,
        metadata={"failure_classification_id": "abc-123", "cluster_id": 2},
    )
    result = await c._process_failure(fe)
    assert result is not None
    assert result["failure_id"] == "abc-123"
    assert result["cluster_id"] == 2
    assert result["teacher_model"] == settings.teacher_model
    assert result["teacher_model"] != "gpt-4o"
