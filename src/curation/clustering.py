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

log = structlog.get_logger()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class FailureClusterer:
    def __init__(self) -> None:
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

    def cluster(
        self,
        failures: list[FailureEvent],
        min_cluster_size: int = 5,
    ) -> list[FailureEvent]:
        """
        Assigns cluster_id to each failure event in-place.
        Returns the same list with cluster_id populated.
        Cluster -1 means noise (no cluster).
        """
        if len(failures) < min_cluster_size:
            for f in failures:
                f.metadata["cluster_id"] = -1
                f.metadata["cluster_label"] = "noise"
            return failures

        texts = [f"{f.prompt} {f.completion}" for f in failures]
        embeddings = self._encoder.encode(texts, batch_size=64, show_progress_bar=False)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)

        for failure, label in zip(failures, labels):
            failure.metadata["cluster_id"] = int(label)
            failure.metadata["cluster_label"] = f"cluster_{label}" if label >= 0 else "noise"

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        log.info(
            "failure_clustering_complete",
            n_failures=len(failures),
            n_clusters=n_clusters,
        )
        return failures
