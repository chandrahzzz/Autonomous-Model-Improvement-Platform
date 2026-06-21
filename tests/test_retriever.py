"""Unit tests for the document retriever (RAG grounding fallback). No DB/model."""

from unittest.mock import AsyncMock, patch

import pytest

from src.retrieval.retriever import DocumentRetriever


# ── pure ranking (_rank) ─────────────────────────────────────────────────────

def test_rank_orders_by_similarity_desc():
    q = [1.0, 0.0, 0.0]
    docs = [
        ("far", "far content", [0.0, 1.0, 0.0]),     # orthogonal → sim 0
        ("near", "near content", [0.99, 0.14, 0.0]),  # high sim
        ("mid", "mid content", [0.7, 0.7, 0.0]),      # ~0.7
    ]
    ranked = DocumentRetriever._rank(q, docs, top_k=3, min_similarity=0.0)
    assert [r[0] for r in ranked] == ["near", "mid", "far"]
    assert ranked[0][2] >= ranked[1][2] >= ranked[2][2]


def test_rank_filters_below_threshold():
    q = [1.0, 0.0, 0.0]
    docs = [
        ("near", "c", [1.0, 0.0, 0.0]),   # sim 1.0
        ("far", "c", [0.0, 1.0, 0.0]),    # sim 0.0
    ]
    ranked = DocumentRetriever._rank(q, docs, top_k=5, min_similarity=0.45)
    assert [r[0] for r in ranked] == ["near"]


def test_rank_respects_top_k():
    q = [1.0, 0.0]
    docs = [(f"d{i}", "c", [1.0, 0.0]) for i in range(10)]
    ranked = DocumentRetriever._rank(q, docs, top_k=3, min_similarity=0.0)
    assert len(ranked) == 3


def test_rank_empty_docs_returns_empty():
    assert DocumentRetriever._rank([1.0, 0.0], [], top_k=3, min_similarity=0.0) == []


def test_rank_skips_zero_and_missing_embeddings():
    q = [1.0, 0.0]
    docs = [
        ("zero", "c", [0.0, 0.0]),   # zero vector → skipped
        ("none", "c", None),         # missing → skipped
        ("ok", "c", [1.0, 0.0]),
    ]
    ranked = DocumentRetriever._rank(q, docs, top_k=5, min_similarity=0.0)
    assert [r[0] for r in ranked] == ["ok"]


# ── retrieve() — DB/encoder mocked ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_retrieve_empty_kb_returns_none_without_embedding():
    r = DocumentRetriever()
    r.embed = lambda text: (_ for _ in ()).throw(AssertionError("embed must not run on empty KB"))
    repo = AsyncMock()
    repo.all_with_embeddings.return_value = []
    with patch("src.retrieval.retriever.KnowledgeDocumentRepository", return_value=repo):
        result = await r.retrieve("any query", db=AsyncMock())
    assert result is None  # short-circuits before loading the model


@pytest.mark.asyncio
async def test_retrieve_returns_context_and_sources():
    r = DocumentRetriever()
    r.embed = lambda text: [1.0, 0.0, 0.0]
    repo = AsyncMock()
    repo.all_with_embeddings.return_value = [
        ("policy-refunds", "Refunds within 30 days.", [1.0, 0.0, 0.0]),
        ("policy-shipping", "Ships in 3-5 days.", [0.0, 1.0, 0.0]),  # orthogonal → filtered
    ]
    with patch("src.retrieval.retriever.KnowledgeDocumentRepository", return_value=repo), \
         patch("src.retrieval.retriever.settings") as s:
        s.retrieval_top_k = 3
        s.retrieval_min_similarity = 0.45
        s.retrieval_max_context_chars = 4000
        result = await r.retrieve("refund window?", db=AsyncMock())
    assert result is not None
    context, sources = result
    assert sources == ["policy-refunds"]
    assert "[doc_id: policy-refunds]" in context
    assert "Refunds within 30 days." in context


@pytest.mark.asyncio
async def test_retrieve_no_match_returns_none():
    r = DocumentRetriever()
    r.embed = lambda text: [1.0, 0.0, 0.0]
    repo = AsyncMock()
    repo.all_with_embeddings.return_value = [("d", "c", [0.0, 1.0, 0.0])]  # orthogonal
    with patch("src.retrieval.retriever.KnowledgeDocumentRepository", return_value=repo), \
         patch("src.retrieval.retriever.settings") as s:
        s.retrieval_top_k = 3
        s.retrieval_min_similarity = 0.45
        s.retrieval_max_context_chars = 4000
        result = await r.retrieve("query", db=AsyncMock())
    assert result is None
