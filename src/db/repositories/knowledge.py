from typing import Any

from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import KnowledgeDocument


class KnowledgeDocumentRepository:
    """Thin data layer for the domain knowledge base (retrieval)."""

    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def insert(
        self, source_id: str, content: str, embedding: list[float], metadata: dict | None = None
    ) -> KnowledgeDocument:
        doc = KnowledgeDocument(
            source_id=source_id, content=content, embedding=embedding, metadata_=metadata or {}
        )
        self._db.add(doc)
        await self._db.flush()
        return doc

    async def all_with_embeddings(self) -> list[tuple[str, str, list[float]]]:
        """Returns (source_id, content, embedding) for every embedded document."""
        result = await self._db.execute(
            select(KnowledgeDocument.source_id, KnowledgeDocument.content, KnowledgeDocument.embedding)
            .where(KnowledgeDocument.embedding.isnot(None))
        )
        return [(r.source_id, r.content, r.embedding) for r in result.fetchall()]

    async def count(self) -> int:
        result = await self._db.execute(select(func.count()).select_from(KnowledgeDocument))
        return int(result.scalar() or 0)

    async def delete_by_source(self, source_id: str) -> int:
        result = await self._db.execute(
            delete(KnowledgeDocument).where(KnowledgeDocument.source_id == source_id)
        )
        return result.rowcount or 0
