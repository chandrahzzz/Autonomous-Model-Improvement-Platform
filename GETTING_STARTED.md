# Getting Started — Continuous Fine-Tuning Pipeline ($0 stack)

One-stop guide to run the whole thing. This is the **handoff doc**: what's done,
how to run it end-to-end, and the one part only you can do (the Colab GPU step).

## What this project is

An autonomous LLM fine-tuning pipeline: it watches an app's LLM calls, detects
when the model degrades (hallucination / drift / refusal / format), gets a
teacher model to write corrections, trains a LoRA adapter, evaluates it, shadow
A/B tests it, and promotes it only if it's better — forever, no human in the loop.
Runs on a **permanent $0 stack**: Groq (teacher), local HF models (safety/eval),
Colab T4 (GPU), HuggingFace Hub (adapters), Redpanda (event bus).

## Status — what's done and proven

- **Code complete**, **222 tests green**.
- **$0 migration done** (no OpenAI/Together/paid deps on the hot path).
- **Live loop proven on this machine**: detect → curate → store produced real
  training examples via Groq at **$0 cost**; `failure_classifications` persist;
  drift scores read sane; audit trail writes and is immutable.
- **Traffic simulator** (`scripts/traffic_simulator.py`) feeds realistic,
  controllable traffic — the pipeline has something to watch.
- **Colab-primary training backend** built (no Modal needed).

### Loop-completion audit (fixed)

An earlier version of this doc claimed the Colab step was the *only* thing left.
That was wrong. A code audit found seven defects that made the loop unable to
finish regardless of Colab; all are fixed and covered by
`tests/test_loop_completion_fixes.py`:

| Defect | Effect before the fix |
|---|---|
| `ShadowRouter` had no callers | `shadow_logs` stayed empty → the graph looped `ab_test_node → END` forever; it could never promote or roll back |
| `elapsed_hours` was `now - (now - ab_min_hours)` | A constant, so the "48h window" gate always passed |
| No shadow-window ceiling | A challenger that never hit `AB_MIN_REQUESTS` stalled the pipeline permanently |
| `incumbent_scores` never updated after a promotion | Gate 4 compared every challenger to frozen seed constants; a regression could pass |
| Next version tag derived from *production* | After a rollback the tag was reissued → UNIQUE violation on `version_tag` |
| `cycles_completed` only moved on promotion | Baseline refresh, DLQ replay, calibration and `shadow_logs` pruning never ran |
| `log_monitor` had no dedup | Every row re-classified up to ~60×/hour; drift window fed duplicates |
| `audit_logger` node registered but unreachable | Detection events never reached the audit trail |

Remaining known gaps (not blockers for the loop, but real): the API has **no
authentication** on 12 mutating endpoints, the `tests/integration/` suite is
fully mocked so nothing has run against a real Postgres/Redis/Kafka, there is no
`Dockerfile` despite `k8s/` referencing an image, and there is no CI.

With those fixed, the Colab GPU step below is what remains for a hands-off
train→promote loop — your part, because it needs your Google/ngrok accounts.

## Prerequisites (all free, you already have most)

| Thing | Where | In `.env` |
|---|---|---|
| Groq API key | console.groq.com | `GROQ_API_KEY` ✅ |
| HuggingFace token (Write) | huggingface.co | `HF_TOKEN` ✅ |
| HF repo prefix | your HF username | `HF_HUB_REPO_PREFIX` ✅ |
| Docker Desktop | docker.com | (runs infra) |
| ngrok account | ngrok.com | for the Colab step |
| A Google account | for Colab (T4 GPU) | for the Colab step |

Also accept the Llama-3 license once: huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

## Run it end-to-end (local, ~15 min)

```bash
# 1. Infrastructure
docker compose up -d postgres redis redpanda

# 2. DB schema
uv run alembic -c alembic/alembic.ini upgrade head

# 3. Seed eval set + knowledge base
uv run python scripts/seed_eval_set.py
uv run python scripts/seed_knowledge_base.py

# 4. Healthy traffic → drift baseline
uv run python scripts/traffic_simulator.py --mode db --scenario healthy --count 1500
uv run python scripts/seed_baseline.py

# 5. Start the API and the runner (two terminals)
uv run uvicorn src.api.main:app --port 8000        # terminal A
uv run python -m src.graph.runner                  # terminal B

# 6. Degrade traffic → detectors fire, examples accumulate
uv run python scripts/traffic_simulator.py --mode db --scenario degrade --count 400 \
    --hallucination-rate 0.35 --refusal-rate 0.30 --format-break-rate 0.20 --drift-rate 0.10

# 7. Watch it move
curl localhost:8000/pipeline/status
curl localhost:8000/metrics | grep -E "failures_detected|examples_curated|pending_examples"
```

You'll see failures detected → Groq curating corrections → `training_examples`
growing. When they cross the trigger (50 in demo config), training fires.

## Your part — the Colab GPU step (the last leg)

The pipeline is `TRAINING_BACKEND=colab`. When training triggers it inlines the
dataset into a webhook, Colab trains on a free T4 and pushes the adapter to your
HF Hub, and the pipeline polls the Hub for it. To wire it up (~10 min):

1. **ngrok** → sign up, copy your authtoken.
2. Open a **Colab notebook**, set runtime to **T4 GPU**, and run the 3 cells in
   [docs/TRAINING_BACKENDS.md](docs/TRAINING_BACKENDS.md) (paste your `HF_TOKEN`
   and ngrok token).
3. It prints an ngrok URL → put it in `COLAB_WEBHOOK_URL` in `.env`.
4. Restart the runner and keep the Colab tab open.

Now the full loop runs itself: trigger → Colab trains → adapter on Hub → pipeline
polls → eval → shadow A/B → promote/rollback, all audited.

I can't do steps 1–2 for you (your accounts, browser, T4 runtime). Everything
else is done and running.

## Demo vs Production config

`.env` currently holds **demo** values so the loop completes in minutes. Each is
marked inline with its production value. Before any real use, restore them (see
also `.env.demo`):

| Setting | Demo | Production |
|---|---|---|
| `TRAINING_TRIGGER_DATASET_SIZE` | 50 | 500 |
| `TRAINING_MIN_INTERVAL_HOURS` | 0 | 6 |
| `AB_MIN_REQUESTS` | 50 | 1000 |
| `AB_MIN_HOURS` | 0.05 | 48 |
| `RETRIEVAL_ENABLED` | false | true (see note) |
| `GROQ_REQUESTS_PER_MINUTE` | 60 | match your Groq tier |
| `SIM_TRAFFIC_ENABLED` | true | false |
| `EVAL_REAL_INFERENCE` | false | true (needs a GPU for eval) |

## Honest limitations (things to know)

- **Detection is English-centric** (MiniLM / DeBERTa-NLI / toxic-bert / Presidio).
  Non-English or code-switched traffic scores less accurately — swap in
  multilingual models for that.
- **Retrieval (RAG-fallback) is off**: it grounds non-RAG failures against 3
  concatenated KB docs, which strict NLI rejects. Real hallucinations still ground
  against their own attached context. A future fix: ground against the single best doc.
- **Curation drop rate is high** on synthetic data (a quality gate doing its job);
  real varied traffic yields more.
- **`EVAL_REAL_INFERENCE=false`** on a laptop — eval uses a stub. Real challenger
  eval needs a GPU + the gated base weights (do it on Colab or a GPU box).
- Cost breaker caps Groq spend (`MONTHLY_BUDGET_USD`); Modal is unused on the
  colab backend.

## Your checklist

- [ ] Run steps 1–7 above → see the loop detect + curate (no more code needed)
- [ ] ngrok signup → authtoken
- [ ] Colab notebook (T4) → paste HF + ngrok tokens → Run All → copy URL
- [ ] `COLAB_WEBHOOK_URL=<url>` in `.env` → restart runner → keep tab open
- [ ] Run more traffic until training triggers → watch it train + promote
- [ ] Before real use: restore production `.env` values; rotate your Groq/HF tokens
