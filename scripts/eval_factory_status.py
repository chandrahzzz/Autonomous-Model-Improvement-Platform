"""Show eval-factory status: set size, counter, recent factory examples.

Usage: uv run python scripts/eval_factory_status.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import redis.asyncio as aioredis
from sqlalchemy import text

from src.config.settings import settings
from src.db.connection import get_db
from src.db.repositories.eval_set import EvalSetRepository


async def main() -> None:
    async with get_db() as db:
        repo = EvalSetRepository(db)
        counts = await repo.count_active_by_source()
        recent = await repo.list_active(source="factory", limit=5)
        runs = (await db.execute(text(
            "SELECT COUNT(*) FROM audit_trail WHERE event_type = 'eval_set_updated'"
        ))).scalar()

    try:
        r = aioredis.from_url(settings.redis_url, decode_responses=True)
        counter = await r.get(settings.eval_factory_request_counter_key)
        await r.aclose()
    except Exception:
        counter = None

    total = sum(counts.values())
    print("Eval Set Status")
    print("-" * 31)
    print(f"Total active:     {total}")
    print(f"  seed:           {counts.get('seed', 0)}")
    print(f"  factory:        {counts.get('factory', 0)}")
    print()
    print(f"Request counter:  {counter or 0} / {settings.eval_factory_trigger_every_n_requests}")
    print(f"Factory runs:     {runs}  (all time)")
    print()
    print("Newest factory examples:")
    for i, ex in enumerate(recent, 1):
        conf = f"{ex.factory_confidence:.2f}" if ex.factory_confidence is not None else "n/a"
        print(f'  {i}. "{ex.question[:50]}" (conf={conf}, cluster={ex.cluster_id})')


if __name__ == "__main__":
    asyncio.run(main())
