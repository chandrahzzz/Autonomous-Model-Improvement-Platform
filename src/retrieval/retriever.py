"""
Document retriever (MiniLM + numpy cosine, no pgvector — consistent with the
rest of the codebase's similarity scans).

Encodes a query, scans stored document embeddings, and returns the top-K most
similar documents (above a similarity floor) as a single context string with
`[doc_id: ...]` markers so the grounded teacher's source extraction can record
which documents grounded a correction.
"""

import numpy as np
import structlog

from src.config.settings import settings
from src.db.repositories.knowledge import KnowledgeDocumentRepository
from src.monitoring.metrics import retrieval_queries_total

log = structlog.get_logger()

_shared_encoder = None


def _get_encoder():
    global _shared_encoder
    if _shared_encoder is None:
        from sentence_transformers import SentenceTransformer
        from src.curation.clustering import EMBEDDING_MODEL
        _shared_encoder = SentenceTransformer(EMBEDDING_MODEL)
    return _shared_encoder


class DocumentRetriever:
    def embed(self, text: str) -> list[float]:
        vec = _get_encoder().encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vec, dtype=float).tolist()

    @staticmethod
    def _rank(
        query_embedding: list[float],
        docs: list[tuple[str, str, list[float]]],
        top_k: int,
        min_similarity: float,
    ) -> list[tuple[str, str, float]]:
        """Pure ranking — (source_id, content, similarity) for the top_k docs at
        or above min_similarity, highest first. Easily unit-tested."""
        q = np.asarray(query_embedding, dtype=float)
        qn = np.linalg.norm(q)
        if qn == 0 or not docs:
            return []
        q = q / qn
        scored: list[tuple[str, str, float]] = []
        for source_id, content, emb in docs:
            if not emb:
                continue
            v = np.asarray(emb, dtype=float)
            vn = np.linalg.norm(v)
            if vn == 0:
                continue
            sim = float(v @ q / vn)
            if sim >= min_similarity:
                scored.append((source_id, content, sim))
        scored.sort(key=lambda t: t[2], reverse=True)
        return scored[:top_k]

    async def retrieve(self, query: str, db) -> tuple[str, list[str]] | None:
        """Returns (context_string, [source_ids]) or None if nothing relevant.
        Fetches documents first so an empty knowledge base short-circuits without
        loading the embedding model."""
        docs = await KnowledgeDocumentRepository(db).all_with_embeddings()
        if not docs:
            retrieval_queries_total.labels(result="miss").inc()
            return None

        ranked = self._rank(
            self.embed(query), docs,
            top_k=settings.retrieval_top_k,
            min_similarity=settings.retrieval_min_similarity,
        )
        if not ranked:
            retrieval_queries_total.labels(result="miss").inc()
            return None

        parts = [f"[doc_id: {sid}] {content}" for sid, content, _ in ranked]
        sources = [sid for sid, _, _ in ranked]
        context = "\n\n".join(parts)[: settings.retrieval_max_context_chars]
        retrieval_queries_total.labels(result="hit").inc()
        log.info("retrieval_hit", n=len(ranked), top_sim=round(ranked[0][2], 3))
        return context, sources


_retriever_instance: DocumentRetriever | None = None


def get_retriever() -> DocumentRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DocumentRetriever()
    return _retriever_instance
