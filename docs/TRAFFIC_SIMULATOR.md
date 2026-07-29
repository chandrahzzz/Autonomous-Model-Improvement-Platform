# Synthetic Traffic Simulator

The pipeline observes a production app's LLM calls. Without an app feeding it,
the LangGraph loop cycles and finds nothing. This simulator fabricates realistic,
controllable traffic so you can drive the whole
**detect → curate → train → eval → promote** loop end-to-end.

It is **additive and test-only** — it never runs in production and never modifies
pipeline logic. It writes events through the same interfaces a real app would.

- Entry point: `scripts/traffic_simulator.py`
- Package: `sim/` (`config`, `prompts`, `answers`, `scenarios`, `emit`)
- Tests: `tests/test_traffic_simulator.py`

## How it works

1. A **scenario** decides, per call, whether it is healthy or a specific failure.
2. An **injector** turns that decision into a prompt/completion pair engineered to
   trip the *real* detector for that failure type.
3. An **emit** adapter lands the call in `llm_logs` (DB / Kafka / HTTP).

Failure injectors are written against the actual detector code and are
regression-tested against it (e.g. refusal completions are checked against the
real `REFUSAL_PATTERN`), so the simulator can't silently drift out of sync.

| Failure | How it's synthesized | Detector it trips |
|---|---|---|
| `refusal` | completion contains a real refusal phrase | `RefusalDetector` (regex + semantic) |
| `hallucination` | RAG call whose completion contradicts its Acme `retrieved_context` | `HallucinationDetector` (NLI entailment) |
| `drift` | completion from an off-distribution jargon set | `DriftDetector` (Mahalanobis) |
| `format_break` | prose / broken JSON where JSON was requested | `FormatValidator` (JSON + length KL) |

## Ingestion modes (`--mode`)

- **`db` (recommended)** — inserts rows directly via `LLMLogRepository`. Works on a
  laptop with no Kafka consumer. Use this for demos.
- **`kafka`** — produces `LLMEvent`s to `llm.production.events` via the real
  producer. **Only reaches `llm_logs` if a Kafka→DB consumer is running** — the
  simulator warns loudly if you use this mode; prefer `db` unless you've wired a
  consumer.
- **`http`** — POSTs through `LLMInterceptorMiddleware`. Requires
  `SIM_TRAFFIC_ENABLED=true` (mounts a no-op `POST /sim/llm-call`). Same consumer
  caveat as kafka.

## Scenarios (`--scenario`)

- `healthy` — all clean traffic. Use it to seed the drift baseline and populate
  known-good replay rows.
- `degrade` — 30% healthy warm-up, then ramps failures up. The main "watch it
  train" scenario.
- `mixed` — steady realistic blend for a long soak.
- `burst` — a spike of one failure type in the middle (exercises the fast-path).
- `rag_heavy` — mostly grounded Acme RAG traffic; a fraction hallucinate.

## Two detector caveats (important)

Both are properties of the pipeline, not the simulator:

1. **Drift needs a seeded baseline.** `DriftDetector.is_drifting()` compares
   against the `drift_baselines` row created by `scripts/seed_baseline.py`, and
   needs ≥ `DRIFT_MIN_WINDOW` (50) samples. Run a `healthy` batch → `seed_baseline.py`
   *before* a `degrade` run, or drift never fires.
2. **Format-KL needs a length baseline.** `FormatValidator`'s length-distribution
   signal only fires once a baseline length distribution exists, which the pipeline
   sets after the first promotion. Until then, format regressions are caught only
   when JSON validation is explicitly enabled. The simulator still emits broken
   JSON so the signal is available once a baseline exists.

## Demo runbook (cold pipeline → visible full loop)

Assumes infra up (`docker compose up -d postgres redis redpanda`), migrations
applied, and `scripts/seed_eval_set.py` + `scripts/seed_knowledge_base.py` run.

```bash
# 1. Healthy traffic → seed the drift baseline + replay rows
python scripts/traffic_simulator.py --mode db --scenario healthy --count 1500
python scripts/seed_baseline.py

# 2. Degrade → detectors fire, examples accumulate, training can trigger
python scripts/traffic_simulator.py --mode db --scenario degrade --count 3000 \
    --hallucination-rate 0.25 --refusal-rate 0.20 \
    --format-break-rate 0.15 --drift-rate 0.15

# 3. Start the runner (separate terminal) and watch it move
python -m src.graph.runner
curl localhost:8000/pipeline/status
curl localhost:8000/metrics | grep -E 'failures_detected|examples_curated|pending_examples'
```

Preview a plan without touching the DB:

```bash
python scripts/traffic_simulator.py --scenario degrade --count 400 \
    --refusal-rate 0.2 --hallucination-rate 0.2 --dry-run
```

## `.env` demo overrides (so you don't wait 48h)

The production gates require 500 examples and a 48-hour A/B window. For a demo,
lower them — then **restore the production values afterward**:

```dotenv
# DEMO ONLY — revert before real use
TRAINING_TRIGGER_DATASET_SIZE=50     # prod: 500
TRAINING_MIN_INTERVAL_HOURS=0        # prod: 6
AB_MIN_REQUESTS=50                   # prod: 1000
AB_MIN_HOURS=0.05                    # prod: 48
DRIFT_MIN_WINDOW=50                  # keep; below this drift won't fire
```

To exercise `--mode http` as well:

```dotenv
SIM_TRAFFIC_ENABLED=true             # mounts POST /sim/llm-call; never in prod
```

## CLI reference

```
--scenario {healthy,degrade,mixed,burst,rag_heavy}
--mode {db,kafka,http}
--count N | --duration SECONDS      (--duration emits N = duration*rate calls)
--rate CALLS_PER_SEC                 (kafka/http pacing; 0 = unthrottled)
--hallucination-rate / --refusal-rate / --format-break-rate / --drift-rate  [0..1]
--model-version v7
--base-url / --http-path            (http mode)
--backfill-minutes N                 (spread created_at into the past; db mode)
--seed N
--dry-run                            (print the plan + sample calls; emit nothing)
```
