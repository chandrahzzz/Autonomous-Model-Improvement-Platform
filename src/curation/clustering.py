"""
HDBSCAN clustering of failure embeddings.

Groups similar failures so the curation pipeline can select diverse
examples across all failure modes rather than over-sampling one cluster.
HDBSCAN requires no cluster count — it discovers structure automatically.
"""

import numpy as np
import structlog
import hdbscan
from sentence_transformers import SentenceTransformer

from src.detection.failure_classifier import FailureEvent
from src.monitoring.metrics import clustering_bypassed_total

log = structlog.get_logger()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_MIN_CLUSTER_SIZE = 5  # HDBSCAN min_cluster_size ceiling for large batches


def _effective_min_cluster_size(n: int) -> int:
    """Scale min_cluster_size to the batch so small batches (common early in a
    deployment) still form clusters instead of everything being noise-labelled.
    HDBSCAN requires a minimum of 2."""
    return min(MAX_MIN_CLUSTER_SIZE, max(2, n // 3))


def _mark_all_noise(failures: list[FailureEvent]) -> list[FailureEvent]:
    for f in failures:
        f.metadata["cluster_id"] = -1
        f.metadata["cluster_label"] = "noise"
    return failures


class FailureClusterer:
    def __init__(self) -> None:
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

    def cluster(
        self,
        failures: list[FailureEvent],
        min_cluster_size: int | None = None,
    ) -> list[FailureEvent]:
        """
        Assigns cluster_id to each failure event in-place.
        Returns the same list with cluster_id populated.
        Cluster -1 means noise (no cluster).

        ``min_cluster_size`` defaults to a value scaled to the batch size so a
        handful of failures still cluster rather than all being labelled noise.
        """
        n = len(failures)
        if n < 2:
            # Too few to cluster at all — process uniformly (count the bypass).
            clustering_bypassed_total.inc()
            return _mark_all_noise(failures)

        effective = min_cluster_size or _effective_min_cluster_size(n)

        texts = [f"{f.prompt} {f.completion}" for f in failures]
        embeddings = self._encoder.encode(texts, batch_size=64, show_progress_bar=False)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=effective,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)

        # When HDBSCAN finds no structure (everything noise), don't pretend a
        # single "-1 cluster" exists — flag the bypass so downstream diversity
        # logic treats them uniformly.
        if all(label == -1 for label in labels):
            clustering_bypassed_total.inc()
            log.info("clustering_all_noise_bypassed", n_failures=n, min_cluster_size=effective)
            return _mark_all_noise(failures)

        for failure, label in zip(failures, labels):
            failure.metadata["cluster_id"] = int(label)
            failure.metadata["cluster_label"] = f"cluster_{label}" if label >= 0 else "noise"

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        log.info(
            "failure_clustering_complete",
            n_failures=n,
            n_clusters=n_clusters,
            min_cluster_size=effective,
        )
        return failures
