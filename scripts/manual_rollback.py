"""
Emergency manual rollback script.
Rolls back the current production model and writes a manual audit entry.

Usage: python scripts/manual_rollback.py --version v7 --reason "safety_regression"
"""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def rollback(version_tag: str, reason: str) -> None:
    from src.db.connection import AsyncSessionLocal
    from src.db.repositories.model_versions import ModelRepository
    from src.audit.logger import AuditLogger
    from src.audit.schemas import AuditEvent

    async with AsyncSessionLocal() as db:
        repo = ModelRepository(db)
        prod = await repo.get_production_version()
        current = prod.version_tag if prod else "unknown"

        print(f"Current production: {current}")
        print(f"Rolling back to: {version_tag}")
        print(f"Reason: {reason}")

        confirm = input("Confirm rollback? [y/N]: ")
        if confirm.lower() != "y":
            print("Rollback cancelled.")
            return

        # Write audit BEFORE executing rollback
        audit = AuditLogger(db)
        event = AuditEvent(
            event_type="model_rolled_back",
            decision=f"manual rollback to {version_tag}",
            rationale={"reason": reason, "operator": "human"},
            state_snapshot={"current_production": current, "rolling_back_to": version_tag},
            model_version_before=current,
            model_version_after=version_tag,
            operator="human_operator",
        )
        await audit.log(event)

        # Execute rollback
        await repo.rollback(current)
        await repo.promote(version_tag)
        await db.commit()

    print(f"Rollback complete. Production is now: {version_tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Emergency manual model rollback")
    parser.add_argument("--version", required=True, help="Target version tag to roll back to")
    parser.add_argument("--reason", required=True, help="Reason for rollback")
    args = parser.parse_args()
    asyncio.run(rollback(args.version, args.reason))
