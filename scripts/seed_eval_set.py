"""
Seed the held-out evaluation set from tests/fixtures/eval_set.json.
Run once at bootstrap: make seed-eval
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import AsyncSessionLocal
from sqlalchemy import text


FIXTURE_PATH = Path(__file__).parent.parent / "tests" / "fixtures" / "eval_set.json"


async def seed() -> None:
    with open(FIXTURE_PATH) as f:
        examples = json.load(f)

    async with AsyncSessionLocal() as db:
        for ex in examples:
            await db.execute(
                text("""
                    INSERT INTO eval_set (version, question, context, ground_truth, domain)
                    VALUES (:version, :question, :context, :ground_truth, :domain)
                    ON CONFLICT DO NOTHING
                """),
                {
                    "version": "v1",
                    "question": ex["question"],
                    "context": ex["context"],
                    "ground_truth": ex["ground_truth"],
                    "domain": ex.get("domain", "general"),
                },
            )
        await db.commit()

    categories: dict[str, int] = {}
    for ex in examples:
        domain = ex.get("domain", "general")
        categories[domain] = categories.get(domain, 0) + 1
    breakdown = ", ".join(f"{k}={v}" for k, v in sorted(categories.items()))
    print(
        f"Seeded {len(examples)} examples across {len(categories)} categories "
        f"({breakdown}). Eval set is now ready."
    )


if __name__ == "__main__":
    asyncio.run(seed())
