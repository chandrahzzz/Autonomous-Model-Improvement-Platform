"""
Influence estimation backends (RFC-003).

Embedding-based influence: cosine similarity between a failed output and a
training example in the shared MiniLM space is a production-appropriate proxy
for influence (LoRA layers processing similar representations are trained most
strongly on similar inputs). The `InfluenceBackend` protocol lets a gradient-based
estimator (TracIn/TRAK) be swapped in later with zero changes to the attributor,
repository, or API.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class InfluenceBackend(Protocol):
    def score(self, failure_text: str, candidate_texts: list[str]) -> list[float]:
        """One influence score per candidate (higher = more influential)."""
        ...

    @property
    def backend_name(self) -> str:
        ...


class EmbeddingInfluenceBackend:
    """Cosine-similarity influence in sentence-embedding space."""

    backend_name = "embedding_cosine"

    def __init__(self) -> None:
        # Reuse the same MiniLM model used elsewhere. Loaded here (only via
        # get_default_backend at runtime); unit tests bypass __init__ with __new__.
        from sentence_transformers import SentenceTransformer
        from src.curation.clustering import EMBEDDING_MODEL
        self._encoder = SentenceTransformer(EMBEDDING_MODEL)

    def score(self, failure_text: str, candidate_texts: list[str]) -> list[float]:
        """SYNCHRONOUS (encode is sync) — caller must run_in_executor from async."""
        if not candidate_texts:
            return []
        all_texts = [failure_text] + candidate_texts
        embeddings = self._encoder.encode(
            all_texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        embeddings = np.asarray(embeddings, dtype=float)
        failure_emb = embeddings[0]
        candidate_embs = embeddings[1:]
        scores = candidate_embs @ failure_emb  # normalized → dot == cosine
        return scores.tolist()


_backend_instance: EmbeddingInfluenceBackend | None = None


def get_default_backend() -> EmbeddingInfluenceBackend:
    global _backend_instance
    if _backend_instance is None:
        _backend_instance = EmbeddingInfluenceBackend()
    return _backend_instance
