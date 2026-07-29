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
from datetime import datetime, timezone
from typing import Any

import numpy as np
import structlog
from scipy.spatial.distance import mahalanobis
from sentence_transformers import SentenceTransformer

from src.config.settings import settings

log = structlog.get_logger()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WINDOW_SIZE = 1000


def _condition_covariance(cov: np.ndarray, shrinkage: float) -> np.ndarray:
    """Condition a (possibly near-singular) covariance so its inverse is stable.

    Ledoit-Wolf-style shrinkage toward a scaled identity target:
        Σ' = (1 - λ)·Σ + λ·mean_var·I
    plus a variance floor on the diagonal. Low-diversity baselines (e.g. templated
    text) have near-zero variance in many embedding dimensions, making Σ singular
    and Mahalanobis distances explode; shrinking toward the average-variance
    identity keeps distances sane without discarding the correlation structure.
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    d = cov.shape[0]
    mean_var = float(np.trace(cov) / d) if d else 1.0
    if mean_var <= 0:
        mean_var = 1.0
    lam = float(min(max(shrinkage, 0.0), 1.0))
    if lam > 0:
        cov = (1.0 - lam) * cov + lam * mean_var * np.eye(d)
    # Floor the diagonal so no single dimension has ~zero variance.
    floor = max(mean_var * 1e-3, 1e-6)
    diag = np.diag(cov).copy()
    np.fill_diagonal(cov, np.maximum(diag, floor))
    return cov


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class DriftDetector:
    def __init__(self) -> None:
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        self._centroid: np.ndarray | None = None
        self._cov_inv: np.ndarray | None = None
        self._window: deque[float] = deque(maxlen=WINDOW_SIZE)
        self._loaded = False
        self._computed_at: datetime | None = None  # when the active baseline was built

    async def load_baseline(self, db: Any) -> None:
        from src.db.repositories.model_versions import ModelRepository
        repo = ModelRepository(db)
        baseline = await repo.get_active_baseline()
        if baseline:
            self._centroid = np.array(json.loads(baseline["centroid"]))
            self._cov_inv = np.array(json.loads(baseline["covariance_inv"]))
            self._computed_at = _parse_dt(baseline.get("computed_at"))
            self._loaded = True
            log.info("drift_baseline_loaded", sample_size=baseline["sample_size"])
        else:
            log.warning("no_drift_baseline_found")

    @property
    def baseline_age_hours(self) -> float | None:
        """Hours since the active baseline was computed; None if not loaded."""
        if self._computed_at is None:
            return None
        delta = datetime.now(timezone.utc) - self._computed_at
        return delta.total_seconds() / 3600.0

    def is_baseline_stale(self) -> bool:
        age = self.baseline_age_hours
        return age is not None and age > settings.drift_baseline_max_age_hours

    async def compute_baseline(self, texts: list[str]) -> dict[str, Any]:
        """
        Compute centroid + inverse covariance from sample texts.
        Call once at setup with ~10k known-good production outputs.
        """
        log.info("computing_drift_baseline", n=len(texts))
        embeddings = self._encoder.encode(texts, batch_size=64, show_progress_bar=True)
        centroid = np.mean(embeddings, axis=0)
        cov = np.cov(embeddings.T)
        cov = _condition_covariance(cov, settings.drift_covariance_shrinkage)
        # pinv (not inv) is stable even if the conditioned matrix is still close to
        # singular — it never raises and degrades gracefully.
        cov_inv = np.linalg.pinv(cov)
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
        self._computed_at = datetime.now(timezone.utc)
        self._window.clear()
        self._loaded = True
        log.info("drift_baseline_refreshed", model_version=model_version, n=len(texts))
        return True

    def score(self, text: str) -> float:
        """
        Returns a DIMENSION-NORMALIZED Mahalanobis distance for one output, or 0.0
        if the baseline isn't loaded. Appends to the rolling window.

        Raw Mahalanobis distance in a d-dim embedding space is ~sqrt(d) for any
        in-distribution point (mahalanobis² ~ chi-squared with d dof), so for
        MiniLM's 384 dims a normal point scores ~20 — which made the old absolute
        threshold (0.15) meaningless and fired drift on everything. Dividing by
        sqrt(d) makes the score dimension-independent: an in-distribution point
        scores ~1.0 and a drifted one scores higher, so the threshold is a simple
        ratio-of-expected (e.g. 1.5 = 50% beyond a typical in-distribution point).
        """
        if not self._loaded or self._centroid is None or self._cov_inv is None:
            return 0.0

        embedding = self._encoder.encode([text])[0]
        try:
            raw = float(mahalanobis(embedding, self._centroid, self._cov_inv))
            d = len(self._centroid)
            dist = raw / np.sqrt(d) if d > 0 else raw
        except Exception:
            dist = 0.0

        self._window.append(dist)
        return dist

    @property
    def rolling_drift_score(self) -> float:
        if not self._window:
            return 0.0
        return float(np.mean(self._window))

    @property
    def window_size(self) -> int:
        return len(self._window)

    def is_drifting(self) -> bool:
        """Fire only when the rolling mean exceeds the threshold AND the window
        holds enough samples to be trustworthy. On quiet traffic a handful of
        outliers must not be enough to trigger a (costly) training run (#3)."""
        if len(self._window) < settings.drift_min_window:
            from src.monitoring.metrics import detector_insufficient_data_total
            detector_insufficient_data_total.labels(detector="drift").inc()
            return False
        return self.rolling_drift_score > settings.drift_mahalanobis_threshold

    # ── Rolling-window persistence (#4) ────────────────────────────────────────
    # The window is in-memory only; a process restart would reset the drift trend
    # to empty and silence detection for the first ~1000 requests. Persist it to
    # Redis each cycle and rehydrate on startup.
    async def save_window_state(self, redis: Any) -> None:
        if not settings.detector_state_persist_enabled:
            return
        key = f"{settings.detector_state_redis_prefix}:drift_window"
        try:
            await redis.set(key, json.dumps(list(self._window)))
        except Exception:
            log.warning("drift_window_persist_failed")

    async def load_window_state(self, redis: Any) -> None:
        if not settings.detector_state_persist_enabled:
            return
        key = f"{settings.detector_state_redis_prefix}:drift_window"
        try:
            raw = await redis.get(key)
        except Exception:
            log.warning("drift_window_rehydrate_failed")
            return
        if not raw:
            return
        try:
            values = json.loads(raw)
            self._window = deque((float(v) for v in values), maxlen=WINDOW_SIZE)
            log.info("drift_window_rehydrated", n=len(self._window))
        except (ValueError, TypeError):
            log.warning("drift_window_rehydrate_parse_failed")
