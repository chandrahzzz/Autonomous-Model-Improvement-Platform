"""
INSERT-only audit trail repository.
No update or delete methods exist — by design.
"""

from typing import Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import AuditTrail


class AuditRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def insert(self, data: dict[str, Any]) -> AuditTrail:
        """Only public write method. No update or delete."""
        entry = AuditTrail(**data)
        self._db.add(entry)
        await self._db.flush()
        return entry

    async def get_recent(self, limit: int = 100) -> list[AuditTrail]:
        result = await self._db.execute(
            select(AuditTrail).order_by(AuditTrail.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())

    async def get_by_id(self, row_id: int) -> AuditTrail | None:
        result = await self._db.execute(
            select(AuditTrail).where(AuditTrail.id == row_id)
        )
        return result.scalar_one_or_none()

    async def get_chain(self, limit: int = 1000) -> list[AuditTrail]:
        """Return entries in insertion order for HMAC chain verification."""
        result = await self._db.execute(
            select(AuditTrail).order_by(AuditTrail.id.asc()).limit(limit)
        )
        return list(result.scalars().all())
