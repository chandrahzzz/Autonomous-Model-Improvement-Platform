"""
Immutable, HMAC-signed audit logger.

CRITICAL: Every audit entry is written BEFORE the action executes.
This ensures the audit trail is always ahead of actual system state.
Write-then-act, never act-then-write.

The audit_trail table has row-level security enforced at the DB level —
no role can UPDATE or DELETE rows.
"""

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.audit.schemas import AuditEvent
from src.audit.hmac_signer import HMACSigner
from src.db.repositories.audit_trail import AuditRepository

log = structlog.get_logger()

_signer = HMACSigner()


class AuditLogger:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._repo = AuditRepository(db)

    async def log(self, event: AuditEvent) -> int:
        """
        Write an audit entry. Returns the row ID.

        Must be called BEFORE the action it records.
        """
        payload = {
            "event_type": event.event_type,
            "decision": event.decision,
            "rationale": event.rationale,
            "state_snapshot": event.state_snapshot,
            "operator": event.operator,
        }
        signature = _signer.sign(payload)

        entry = await self._repo.insert({
            "event_type": event.event_type,
            "decision": event.decision,
            "rationale": event.rationale,
            "state_snapshot": event.state_snapshot,
            "model_version_before": event.model_version_before,
            "model_version_after": event.model_version_after,
            "operator": event.operator,
            "hmac_sha256": signature,
        })

        log.info(
            "audit_entry_written",
            row_id=entry.id,
            event_type=event.event_type,
            decision=event.decision,
        )
        return entry.id
