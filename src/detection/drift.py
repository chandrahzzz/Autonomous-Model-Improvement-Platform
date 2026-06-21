"""
Semantic drift detection via Mahalanobis distance in embedding space.

Mahalanobis accounts for correlations between embedding dimensions and
gives a statistically rigorous notion of "unusualness" vs the baseline
distribution — more accurate than cosine distance which ignores covariance.

Baseline: centroid + inverse covariance matrix from 10k known-good outputs.
Trigger: rolling mean over last 1000 outputs exceeds threshold.
"""

import json
from collections import deque
from typing import Any

import numpy as np
import structlog
from scipy.spatial.distance import mahalanobis
from sentence_transformers import SentenceTransformer

from src.config.settings import settings

log = structlog.get_logger()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_SIZE = 1000


class DriftDetector:
    def __init__(self) -> None:
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        self._centroid: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._loaded = False

    async def load_baseline(self, db: Any) -> None:
        from src.db.repositories.model_versions import ModelRepository
        repo = ModelRepository(db)
        baseline = await repo.get_active_baseline()
        if baseline:
            self._centroid = np.array(json.loads(baseline["centroid"]))
            self._cov_inv = np.array(json.loads(baseline["covariance_inv"]))
            self._loaded = True
            log.info("drift_baseline_loaded", sample_size=baseline["sample_size"])
        else:
            log.warning("no_drift_baseline_found")

    async def compute_baseline(self, texts: list[str]) -> dict[str, Any]:
        """
        Compute centroid + inverse covariance from sample texts.
        Call once at setup with ~10k known-good production outputs.
        """
        log.info("computing_drift_baseline", n=len(texts))
        embeddings = self._encoder.encode(texts, batch_size=64, show_progress_bar=True)
        centroid = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings.T)
        cov += np.eye(cov.shape[0]) * 1e-6   # regularize for numerical stability
        cov_inv = np.linalg.inv(cov)
        return {
            "centroid": centroid.tolist(),
            "covariance_inv": cov_inv.tolist(),
            "sample_size": len(texts),
        }

    async def refresh_baseline(
        self, db: Any, model_version: str, min_samples: int = 500
    ) -> bool:
        """Recompute and persist the drift baseline from recent production logs.

        Call after a successful promotion so drift is measured against the NEW
        production model's output distribution rather than the original seed
        (which otherwise grows stale and causes false alarms or missed drift).

        Returns True if a new baseline was saved, False if there were too few
        logs to compute one.
        """
        from src.db.repositories.llm_logs import LLMLogRepository
        from src.db.repositories.model_versions import ModelRepository

        log_repo = LLMLogRepository(db)
        texts = await log_repo.get_recent_completions(limit=10000, model_version=model_version)
        if len(texts) < min_samples:
            # New model may have no traffic yet — fall back to most-recent overall.
            texts = await log_repo.get_recent_completions(limit=10000)
        if len(texts) < min_samples:
            log.warning(
                "drift_baseline_refresh_insufficient_samples",
                found=len(texts), required=min_samples,
            )
            return False

        baseline = await self.compute_baseline(texts)
        await ModelRepository(db).save_baseline(model_version, baseline)

        # Adopt the new baseline in-process and reset the rolling window.
        self._centroid = np.array(baseline["centroid"])
        self._cov_inv = np.array(baseline["covariance_inv"])
        self._window.clear()
        self._loaded = True
        log.info("drift_baseline_refreshed", model_version=model_version, n=len(texts))
        return True

    def score(self, text: str) -> float:
        """
        Returns Mahalanobis distance for one output. 0.0 if baseline not loaded.
        Appends to rolling window for trend detection.
        """
        if not self._loaded or self._centroid is None or self._cov_inv is None:
            return 0.0

        embedding = self._encoder.encode([text])[0]
        try:
            dist = float(mahalanobis(embedding, self._centroid, self._cov_inv))
        except Exception:
            dist = 0.0

        self._window.append(dist)
        return dist

    @property
    def rolling_drift_score(self) -> float:
        if not self._window:
            return 0.0
        return float(np.mean(self._window))

    def is_drifting(self) -> bool:
        return self.rolling_drift_score > settings.drift_mahalanobis_threshold
