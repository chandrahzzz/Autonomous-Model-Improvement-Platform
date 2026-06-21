"""Show the training examples most responsible for a model's failures (RFC-003).

Usage:
  uv run python scripts/attribution_report.py --version v8 --top-n 20 --min-score 0.70
  uv run python scripts/attribution_report.py --version v8 --top-n 20 --retract --confirm
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from src.db.connection import get_db
from src.db.repositories.attribution import FailureAttributionRepository


async def _report(version: str, top_n: int, min_score: float) -> list[dict]:
    async with get_db() as db:
        rows = await FailureAttributionRepository(db).get_top_influential_examples(version, min_score=min_score)
    return rows[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--min-score", type=float, default=0.70)
    ap.add_argument("--retract", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--api", default="http://localhost:8000")
    args = ap.parse_args()

    rows = asyncio.run(_report(args.version, args.top_n, args.min_score))

    print(f"Attribution Report — Model {args.version}")
    print("-" * 70)
    print(f"{'Rank':>4} | {'Appears':>7} | {'AvgScore':>8} | {'FailureType':<14} | Prompt Preview")
    for i, r in enumerate(rows, 1):
        print(f"{i:>4} | {r['appearances']:>7} | {r['mean_influence_score']:>8.3f} | "
              f"{(r['failure_type'] or ''):<14} | {(r['prompt_preview'] or '')[:40]!r}")
    print(f"\nFound {len(rows)} examples. Pass --retract --confirm to mark them for removal.")

    if args.retract:
        if not args.confirm:
            print("Refusing to retract without --confirm.")
            return
        ids = [r["example_id"] for r in rows]
        resp = httpx.post(
            f"{args.api}/attribution/retract",
            json={"example_ids": ids, "reason": "attribution_report_cli"},
            timeout=30.0,
        )
        print("Retract response:", resp.status_code, resp.text)


if __name__ == "__main__":
    main()
