"""Domain knowledge base endpoints (retrieval grounding)."""

import structlog
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.config.settings import settings
from src.db.repositories.knowledge import KnowledgeDocumentRepository
from src.monitoring.metrics import knowledge_base_documents, knowledge_base_size_warnings_total
from src.retrieval.retriever import get_retriever

log = structlog.get_logger()
router = APIRouter()


class DocumentIn(BaseModel):
    source_id: str
    content: str
    metadata: dict | None = None


@router.post("/documents")
async def add_document(doc: DocumentIn, db: AsyncSession = Depends(get_db_session)) -> dict:
    repo = KnowledgeDocumentRepository(db)
    embedding = get_retriever().embed(doc.content)
    row = await repo.insert(
        source_id=doc.source_id, content=doc.content, embedding=embedding, metadata=doc.metadata
    )
    # Size guard (#I3): retrieval is a numpy O(N) full scan that runs on every
    # teacher correction, so warn when the KB grows past the point where latency
    # starts to matter. The documented migration path is pgvector + IVFFlat.
    count = await repo.count()
    knowledge_base_documents.set(count)
    warning = None
    if count > settings.knowledge_base_size_warn_threshold:
        knowledge_base_size_warnings_total.inc()
        warning = (
            f"knowledge_base has {count} docs (> {settings.knowledge_base_size_warn_threshold}); "
            "retrieval is an O(N) numpy scan — consider migrating to pgvector + IVFFlat "
            "for sub-linear ANN search (see docs/RETRIEVAL_SCALING.md)."
        )
        log.warning("knowledge_base_size_warning", count=count)
    resp = {"id": str(row.id), "source_id": row.source_id, "document_count": count}
    if warning:
        resp["warning"] = warning
    return resp


@router.get("/count")
async def document_count(db: AsyncSession = Depends(get_db_session)) -> dict:
    return {"documents": await KnowledgeDocumentRepository(db).count()}


@router.get("/search")
async def search(q: str, db: AsyncSession = Depends(get_db_session)) -> dict:
    """Debug retrieval: returns the context + source IDs a query would ground on."""
    result = await get_retriever().retrieve(q, db)
    if result is None:
        return {"query": q, "hit": False, "sources": [], "context": None}
    context, sources = result
    return {"query": q, "hit": True, "sources": sources, "context": context}
