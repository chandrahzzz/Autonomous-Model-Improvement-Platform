"""
Synthetic traffic simulator — feed the pipeline realistic, controllable LLM
traffic so the detect → curate → train → eval → promote loop has something to
observe. Additive, test-only; never runs in production.

Examples:
    # 1. Healthy traffic to seed the drift baseline + populate replay rows
    python scripts/traffic_simulator.py --mode db --scenario healthy --count 1500
    python scripts/seed_baseline.py

    # 2. Degrade: detectors fire, examples accumulate, training can trigger
    python scripts/traffic_simulator.py --mode db --scenario degrade --count 3000 \
        --hallucination-rate 0.25 --refusal-rate 0.20 --format-break-rate 0.15 --drift-rate 0.15

    # 3. Watch it move
    curl localhost:8000/pipeline/status
    curl localhost:8000/metrics | grep -E 'failures_detected|examples_curated|pending_examples'

See docs/TRAFFIC_SIMULATOR.md for the full runbook + .env demo overrides.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from collections import Counter
from pathlib import Path

# Make both `sim` and `src` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim import SimConfig, build_call_plan, generate_call, emit_calls  # noqa: E402
from sim.scenarios import SCENARIOS  # noqa: E402
from sim.config import FAILURE_TYPES  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthetic LLM traffic simulator for the fine-tuning pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario", choices=SCENARIOS, default="mixed")
    p.add_argument("--mode", choices=("db", "kafka", "http"), default="db")

    size = p.add_mutually_exclusive_group()
    size.add_argument("--count", type=int, default=500, help="total calls to emit")
    size.add_argument("--duration", type=float, default=None,
                      help="run for N seconds at --rate (overrides --count)")
    p.add_argument("--rate", type=float, default=50.0,
                   help="calls/sec pacing (kafka/http modes; 0 = unthrottled)")

    p.add_argument("--hallucination-rate", type=float, default=0.0)
    p.add_argument("--refusal-rate", type=float, default=0.0)
    p.add_argument("--format-break-rate", type=float, default=0.0)
    p.add_argument("--drift-rate", type=float, default=0.0)

    p.add_argument("--model-version", default="v7")
    p.add_argument("--base-url", default="http://localhost:8000", help="http mode")
    p.add_argument("--http-path", default="/sim/llm-call", help="http mode endpoint")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--backfill-minutes", type=float, default=0.0,
                   help="spread created_at over the past N minutes (db mode)")
    p.add_argument("--use-groq", action="store_true",
                   help="(reserved) call Groq for healthy answers; default templated")
    p.add_argument("--dry-run", action="store_true",
                   help="build + print the plan and sample calls; emit nothing")
    return p.parse_args(argv)


def _config_from_args(a: argparse.Namespace) -> SimConfig:
    return SimConfig(
        scenario=a.scenario, mode=a.mode, count=a.count,
        duration_seconds=a.duration, rate=a.rate,
        hallucination_rate=a.hallucination_rate, refusal_rate=a.refusal_rate,
        format_break_rate=a.format_break_rate, drift_rate=a.drift_rate,
        model_version=a.model_version, base_url=a.base_url, http_path=a.http_path,
        seed=a.seed, backfill_minutes=a.backfill_minutes, use_groq=a.use_groq,
    )


def _resolve_count(cfg: SimConfig) -> int:
    if cfg.duration_seconds is not None:
        return max(1, int(cfg.duration_seconds * max(cfg.rate, 1.0)))
    return cfg.count


def build_calls(cfg: SimConfig, n: int):
    """Deterministic (seeded) plan → concrete Calls. Separated from emit so it's
    unit-testable without any external service."""
    plan = build_call_plan(cfg, n)
    rng = random.Random(cfg.seed + 99)
    calls = [generate_call(spec.domain, spec.failure_type, rng) for spec in plan]
    return plan, calls


async def _run(cfg: SimConfig, dry_run: bool) -> None:
    n = _resolve_count(cfg)
    plan, calls = build_calls(cfg, n)

    intended = Counter(spec.failure_type or "healthy" for spec in plan)
    print(f"\n  scenario={cfg.scenario}  mode={cfg.mode}  calls={n}  seed={cfg.seed}")
    print("  intended mix:")
    for label in ("healthy", *FAILURE_TYPES):
        if intended.get(label):
            pct = 100 * intended[label] / n
            print(f"    {label:<16} {intended[label]:>6}  ({pct:4.1f}%)")

    if dry_run:
        print("\n  --dry-run: sample calls (first 3 of each intended type):")
        seen: Counter = Counter()
        for spec, call in zip(plan, calls):
            key = call.failure_type or "healthy"
            if seen[key] < 3:
                seen[key] += 1
                ctx = " [RAG]" if call.is_rag else ""
                print(f"    [{key}]{ctx} Q: {call.prompt[:70]}")
                print(f"          A: {call.completion[:70]}")
        print("\n  emitted nothing (dry run).")
        return

    print(f"\n  emitting via {cfg.mode} ...")
    stats = await emit_calls(calls, cfg)
    print(f"  done: {stats}")
    if cfg.mode == "db":
        print("\n  next: run `python scripts/seed_baseline.py` if you emitted a "
              "healthy batch, then start the runner and watch /pipeline/status.")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    asyncio.run(_run(cfg, args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
