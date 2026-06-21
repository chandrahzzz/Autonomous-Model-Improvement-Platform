"""Seed the domain knowledge base from tests/fixtures/knowledge_base.json.

Embeds each document with MiniLM and stores it for retrieval grounding.
Run once at bootstrap (or whenever the docs change):
    uv run python scripts/seed_knowledge_base.py
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_db
from src.db.repositories.knowledge import KnowledgeDocumentRepository
from src.retrieval.retriever import get_retriever

FIXTURE = Path(__file__).parent.parent / "tests" / "fixtures" / "knowledge_base.json"


async def seed() -> None:
    docs = json.loads(FIXTURE.read_text(encoding="utf-8"))
    retriever = get_retriever()
    async with get_db() as db:
        repo = KnowledgeDocumentRepository(db)
        for d in docs:
            # Replace any existing doc with the same source id (idempotent re-seed).
            await repo.delete_by_source(d["source_id"])
            embedding = retriever.embed(d["content"])
            await repo.insert(source_id=d["source_id"], content=d["content"], embedding=embedding)
        total = await repo.count()
    print(f"Seeded {len(docs)} documents. Knowledge base now has {total} documents.")


if __name__ == "__main__":
    asyncio.run(seed())
