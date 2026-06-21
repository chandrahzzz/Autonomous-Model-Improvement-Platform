"""Domain knowledge base endpoints (retrieval grounding)."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import get_db_session
from src.db.repositories.knowledge import KnowledgeDocumentRepository
from src.retrieval.retriever import get_retriever

router = APIRouter()


class DocumentIn(BaseModel):
    source_id: str
    content: str
    metadata: dict | None = None


@router.post("/documents")
async def add_document(doc: DocumentIn, db: AsyncSession = Depends(get_db_session)) -> dict:
    embedding = get_retriever().embed(doc.content)
    row = await KnowledgeDocumentRepository(db).insert(
        source_id=doc.source_id, content=doc.content, embedding=embedding, metadata=doc.metadata
    )
    return {"id": str(row.id), "source_id": row.source_id}


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
