"""
Fetch the exact training dataset used for a training run, for debugging/audit.

Usage:
    python scripts/reproduce_dataset.py <run_id> [--out ./dataset.jsonl]

Reads the run's dataset_uri (modal://finetuning-artifacts/datasets/<tag>.jsonl)
and downloads it from the Modal Volume via the Modal CLI.
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.connection import get_db
from sqlalchemy import text


async def _lookup_uri(run_id: int) -> str | None:
    async with get_db() as db:
        row = (await db.execute(
            text("SELECT dataset_uri FROM training_runs WHERE id = :r"), {"r": run_id}
        )).fetchone()
    return row.dataset_uri if row else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", type=int)
    ap.add_argument("--out", default="./dataset.jsonl")
    args = ap.parse_args()

    uri = asyncio.run(_lookup_uri(args.run_id))
    if not uri:
        print(f"No dataset_uri recorded for run {args.run_id}")
        sys.exit(1)

    # uri form: modal://<volume>/<path>
    rest = uri.split("modal://", 1)[1]
    volume, _, vol_path = rest.partition("/")
    print(f"Fetching {vol_path} from Modal volume {volume} -> {args.out}")
    subprocess.run(["modal", "volume", "get", volume, vol_path, args.out], check=True)
    print("Done.")


if __name__ == "__main__":
    main()
