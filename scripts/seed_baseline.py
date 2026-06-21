"""
Compute and store the drift detection baseline from recent production outputs.
Requires at least 500 records in llm_logs.
Run once at bootstrap or after a major model change: make seed-baseline
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def seed() -> None:
    from src.db.connection import AsyncSessionLocal
    from src.db.repositories.llm_logs import LLMLogRepository
    from src.db.repositories.model_versions import ModelRepository
    from src.detection.drift import DriftDetector

    async with AsyncSessionLocal() as db:
        log_repo = LLMLogRepository(db)
        model_repo = ModelRepository(db)

        recent = await log_repo.get_recent(limit=10000, hours=168)   # last 7 days
        if len(recent) < 50:
            print(f"Only {len(recent)} logs found. Need at least 50 for baseline. Exiting.")
            return

        texts = [r.completion for r in recent]
        print(f"Computing baseline from {len(texts)} outputs...")

        detector = DriftDetector()
        baseline = await detector.compute_baseline(texts)

        prod = await model_repo.get_production_version()
        model_version = prod.version_tag if prod else "v7"
        await model_repo.save_baseline(model_version, baseline)
        await db.commit()

    print(f"Baseline saved for model {model_version} ({len(texts)} samples).")


if __name__ == "__main__":
    asyncio.run(seed())
