"""
Verify the HMAC integrity of every row in the audit_trail table.
Prints a pass/fail for each row and exits with code 1 if any row is invalid.
Run with: make verify-audit
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def verify() -> None:
    from src.db.connection import AsyncSessionLocal
    from src.db.repositories.audit_trail import AuditRepository
    from src.audit.hmac_signer import HMACSigner

    signer = HMACSigner()
    failures = []

    async with AsyncSessionLocal() as db:
        repo = AuditRepository(db)
        entries = await repo.get_chain(limit=10000)

    print(f"Verifying {len(entries)} audit entries...")
    for entry in entries:
        payload = {
            "event_type": entry.event_type,
            "decision": entry.decision,
            "rationale": entry.rationale,
            "state_snapshot": entry.state_snapshot,
            "operator": entry.operator,
        }
        valid = signer.verify(payload, entry.hmac_sha256)
        status = "✓" if valid else "✗ TAMPERED"
        print(f"  [{entry.id}] {entry.event_type}: {status}")
        if not valid:
            failures.append(entry.id)

    if failures:
        print(f"\nFAILED: {len(failures)} entries have invalid signatures: {failures}")
        sys.exit(1)
    else:
        print(f"\nAll {len(entries)} entries verified successfully.")


if __name__ == "__main__":
    asyncio.run(verify())
