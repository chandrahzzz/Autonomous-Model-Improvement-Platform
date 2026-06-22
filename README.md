# Continuous Fine-Tuning Pipeline

> **Production-grade, autonomous LLM fine-tuning with zero human intervention.**

An end-to-end system that monitors every LLM call, detects quality failures, curates training data, fine-tunes a LoRA adapter, evaluates it against a held-out set, runs a 48h shadow A/B test, and autonomously promotes or rolls back — 24/7, forever.

---

## Architecture Overview

```
Production LLM Calls
        │
        ▼
┌─────────────────┐
│ LLM Interceptor │  ← FastAPI middleware, <5ms overhead
│  (Kafka emit)   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                   LangGraph State Machine                │
│                                                         │
│  log_monitor → failure_detector → example_curator       │
│       ↑              │                    │             │
│       │              ▼                    ▼             │
│       │       data_validator → fine_tune_trigger        │
│       │                              │                  │
│       │                              ▼                  │
│       │                      lora_trainer               │
│       │                     (Modal A100)                │
│       │                              │                  │
│       │                              ▼                  │
│       │                    training_poller              │
│       │                              │                  │
│       │                              ▼                  │
│       │                       eval_runner               │
│       │                    (RAGAS + Safety)             │
│       │                              │                  │
│       │                              ▼                  │
│       │                       ab_test_node              │
│       │                    (48h shadow test)            │
│       │                              │                  │
│       │                              ▼                  │
│       │                   promotion_decider             │
│       │                    /              \             │
│       │               promote          rollback         │
│       │                    \              /             │
│       └──────────────────── log_monitor ←──────────────┘
└─────────────────────────────────────────────────────────┘
```

> Each cycle runs the graph to `END`, then the runner re-invokes it. A
> **conditional entry point** resumes an in-flight `training_poller`,
> `ab_test_node`, or `canary_node` instead of always restarting at
> `log_monitor`. When `CANARY_ENABLED`, a `canary_node` sits between
> `promotion_decider` and the final promote. The runner paces itself
> (`CYCLE_INTERVALS`) and skips cycles when over the cost budget.

---

## Quick Start

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker + Docker Compose
- A PostgreSQL 16 instance (via Docker)

### 1. Clone and install

```bash
git clone <repo-url>
cd continuous-finetuning-pipeline
cp .env.example .env
# Edit .env with your API keys
uv sync
```

### 2. Start infrastructure

```bash
make infra-up
```

This starts PostgreSQL, Redis, Kafka, Prometheus, and Grafana.

### 3. Run migrations

```bash
make migrate
```

### 4. Create Kafka topics

```bash
make topics
```

### 5. Seed evaluation set + drift baseline

```bash
make seed-eval
# Add some LLM logs first, then:
make seed-baseline
```

### 6. Start the pipeline

```bash
# Terminal 1: FastAPI
make dev

# Terminal 2: LangGraph runner
make dev-pipeline
```

Or bootstrap everything at once:

```bash
make bootstrap
```

---

## Configuration

All configuration is in `.env`. Key variables:

| Variable | Default | Description |
|---|---|---|
| `TRAINING_TRIGGER_DATASET_SIZE` | 500 | Minimum examples before training fires |
| `TRAINING_TRIGGER_DRIFT_THRESHOLD` | 0.15 | Mahalanobis drift threshold |
| `TRAINING_MIN_INTERVAL_HOURS` | 6 | Cooldown between training runs |
| `EVAL_IMPROVEMENT_THRESHOLD` | 0.03 | Minimum RAGAS improvement to promote |
| `AB_MIN_REQUESTS` | 1000 | Minimum shadow requests for promotion |
| `AB_MIN_HOURS` | 48 | Minimum shadow test duration |
| `HALLUCINATION_THRESHOLD` | 0.50 | NLI non-entailment cutoff (was `CLAP_HALLUCINATION_THRESHOLD`) |
| `MIN_EVAL_EXAMPLES` | 50 | Floor before a promotion decision is trusted |
| `SHADOW_SCORING_STRATEGY` | `llm_judge` | Shadow quality metric (`llm_judge` \| `reference_rouge`) |
| `SAFETY_CLASSIFIER` | `llama_guard` | Safety classifier (needs `TOGETHER_API_KEY`; else keyword fallback) |
| `VAULT_REQUIRED` | true | Fail loudly in prod if Vault is unreachable |
| `CANARY_ENABLED` | false | Live canary gate before full promotion (needs serving hook) |
| `MONTHLY_BUDGET_USD` | 500 | Cost circuit breaker; runner skips cycles when exceeded |
| `DRIFT_PREDICTION_ENABLED` | true | RFC-001 predictive drift early warning |
| `EVAL_FACTORY_ENABLED` | true | RFC-002 living eval-set generation from traffic |
| `EVAL_FACTORY_TRIGGER_EVERY_N_REQUESTS` | 1000 | Requests between factory runs |
| `ATTRIBUTION_ENABLED` | true | RFC-003 failure→training-example influence attribution |
| `RETRIEVAL_ENABLED` | true | Domain retriever — grounds the teacher when no context is attached |
| `RETRIEVAL_MIN_SIMILARITY` | 0.30 | Cosine floor for a retrieved doc to count as relevant |
| `DRIFT_BASELINE_MAX_AGE_HOURS` | 24.0 | Auto-refresh the drift baseline once it ages past this (not just on promotion) |
| `FORMAT_BASELINE_MAX_AGE_HOURS` | 24.0 | Same age-out for the format length baseline |
| `DRIFT_MIN_WINDOW` / `REFUSAL_MIN_SAMPLES` | 50 | Min samples before drift/refusal aggregate signals may fire |
| `DETECTOR_STATE_PERSIST_ENABLED` | true | Persist detector rolling windows to Redis so restarts don't reset them |
| `CORRELATE_FAILURES_ENABLED` | true | Collapse multi-detector hits on one log into a single failure event |
| `TEACHER_SEMANTIC_CONSISTENCY_THRESHOLD` | 0.80 | MiniLM-cosine agreement needed across the 3 teacher votes |
| `TEACHER_MAX_RETRIES` | 5 | Exponential-backoff retries on OpenAI 429 / timeout before dropping |
| `DEDUP_REHYDRATE_ENABLED` | true | Rebuild the MinHash near-dup index from the DB on startup |
| `EVAL_REAL_INFERENCE` | false | Load base+LoRA (merge_and_unload) for eval; must be on in prod |
| `EVAL_REQUIRE_ADAPTER_VERIFICATION` | true | Block promotion if the challenger ≡ base model (adapter no-op) |
| `EVAL_LOCK_SET_SNAPSHOT` | true | Re-evaluate the incumbent on the same eval-set snapshot for a fair delta |
| `SAFETY_REQUIRE_CLASSIFIER` | true | Fail-closed in prod when Llama Guard is unavailable (no keyword fallback) |
| `TRIGGER_DRIFT_EXEMPT_FAILURE_TYPES` | format_regression, refusal_creep | Failure types that can trigger training without crossing the drift threshold |
| `REPLAY_RECENCY_DECAY` | 0.9 | Exponential decay favouring recent known-good logs in the replay buffer |

### Detection-layer hardening (June 2026)
Six production gaps in the failure-detection layer were closed (plus a real
double-scoring bug). All additive and kill-switched:
- **Grounding visibility** — hallucination events carry `premise_source`/`grounded`;
  RAG calls (`is_rag`) missing their context are counted (`hallucination_premise_missing_total`) and surfaced on `/health` instead of being silently graded against the prompt.
- **Self-healing baselines** — drift *and* format baselines age out and auto-refresh
  independently of promotions, so a 48h shadow window can't leave them stale (`drift_baseline_age_hours` / `format_baseline_age_hours` gauges, shown on `/health`).
- **Quiet-traffic guards** — drift/refusal/format aggregate signals return
  `insufficient_data` below a min-sample floor rather than firing false positives.
- **Restart-safe windows** — detector rolling windows persist to Redis and rehydrate
  on startup, so refusal rate / drift trend don't reset to a misleading clean slate after a deploy.
- **Failure correlation** — when several detectors fire on the same log, only the
  highest-severity event is emitted (others recorded in `metadata.all_failure_types`), so the curator makes one correction, not duplicates.

### Curation-layer hardening (June 2026)
Five production gaps in the curation pipeline were closed. All additive and kill-switched:
- **PII never leaves for OpenAI** — Presidio scrubs the prompt *and* bad completion
  **before** the teacher API call (fail-closed drop on scrub failure); the teacher output is scrubbed again, and the stored `bad_completion` is the scrubbed version.
- **Semantic self-consistency** — the teacher's 3-vote confidence uses MiniLM cosine
  (meaning) instead of ROUGE-L (surface overlap), so paraphrase agreement counts as confidence and divergent meanings don't.
- **Rate-limit resilience** — teacher calls retry OpenAI 429/timeout/connection errors
  with exponential backoff + jitter (`teacher_rate_limit_retries_total`) instead of silently dropping the example.
- **Batch-aware clustering** — HDBSCAN `min_cluster_size` scales to the batch so small
  early-deployment batches still cluster; all-noise batches bypass cleanly (`clustering_bypassed_total`).
- **Restart-safe dedup** — the MinHash near-duplicate index rehydrates from the DB on
  startup so near-dupes (that the exact-hash DB index can't catch) don't re-enter after a restart.

### Training + evaluation hardening (June 2026)
Eight production gaps in the training/eval systems were closed. All additive and kill-switched:
- **Eval runs on the real merged model** — the eval node loads base + LoRA
  (`merge_and_unload`) and **verifies the challenger differs from the base** before trusting any score; a no-op adapter blocks promotion (`src/inference/challenger.py`).
- **Apples-to-apples deltas** — the eval-set snapshot is locked per run and the
  incumbent is re-evaluated on the *same* set, so a RAGAS delta reflects the model, not eval-set drift.
- **Safety gate fails closed** — in production an unavailable Llama Guard blocks
  promotion instead of using brittle keywords; the keyword fallback no longer passes "I'm sorry, but here's how to…".
- **Trigger catches structural regressions** — format/refusal-dominant backlogs can
  trigger training on example count alone (drift is a soft gate for those types).
- **Recency-weighted replay** — known-good replay examples favour recent production
  logs (exponential decay) and the version distribution is recorded per run.
- **Durable dataset artifact** — the dataset is uploaded to the Modal Volume and
  confirmed resolvable *before* submit, so `dataset_uri` never dangles on a failed run.
- **Confidence-weighted eval eviction** — the eval factory evicts the lowest-confidence
  quartile first (LRU within), not pure LRU, so high-value rare examples survive.
- **Regression guard** — a challenger with a negative mean quality delta is blocked
  immediately, before the one-sided significance test.

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health (DB, Redis, Kafka) |
| `/metrics` | GET | Prometheus metrics |
| `/pipeline/status` | GET | Current pipeline state |
| `/pipeline/pause` | POST | Pause the pipeline |
| `/pipeline/resume` | POST | Resume the pipeline |
| `/models/current` | GET | Current production model version |
| `/models/history` | GET | Model version history |
| `/models/rollback/{version}` | POST | Emergency manual rollback |
| `/audit/trail` | GET | Recent audit entries |
| `/audit/verify/{id}` | GET | Verify HMAC signature of audit row |
| `/audit/lineage/{version}` | GET | Full lineage: logs → examples → run → model + dataset URI |
| `/shadow/status` | GET | Shadow A/B test status |
| `/shadow/abort` | POST | Abort current shadow test |
| `/shadow/canary/status` | GET | Canary rollout state |
| `/shadow/canary/abort` | POST | Abort current canary rollout |
| `/metrics/cost` | GET | Current-month spend vs. budget |
| `/training/examples/by-source` | GET | Training examples grounded on a source ID |
| `/training/examples/by-source` | DELETE | Retract examples grounded on a source ID |
| `/drift/trend` · `/trend/history` · `/trend/alarms` | GET | RFC-001 predicted drift trend / history / alarms |
| `/eval/set/summary` · `/set/examples` | GET | RFC-002 eval-set summary / examples |
| `/eval/set/factory/trigger` | POST | RFC-002 manually run the eval factory |
| `/attribution/log/{id}` · `/model/{v}` · `/influential` | GET | RFC-003 failure attributions / most-blamed examples |
| `/attribution/retract` | POST | RFC-003 retract training examples (audit-before-act) |
| `/knowledge/documents` | POST | Add a domain document to the retrieval knowledge base |
| `/knowledge/count` · `/knowledge/search` | GET | KB size / debug what a query grounds on |

---

## Failure Detection

Four detectors run concurrently on every LLM log batch:

| Detector | Method | Threshold |
|---|---|---|
| **Hallucination** | NLI entailment (nli-deberta-v3-base) grounded on `retrieved_context` | Non-entailment > 0.50 |
| **Semantic Drift** | Mahalanobis distance (rolling window 1000); baseline auto-refreshes on promotion | Mean > 0.15 |
| **Refusal Creep** | Keyword + semantic similarity | Rate > 2× baseline |
| **Format Regression** | JSON validation + KL divergence | KL > 0.5 |

---

## Curation Pipeline

Failed examples flow through:

1. **HDBSCAN clustering** — groups similar failures for diverse sampling
2. **GPT-4o teacher correction (RAG-grounded)** — 3-way self-consistency vote at a >0 sampling temperature, bounded by a concurrency semaphore + per-run cost breaker. Context is resolved in three tiers: context attached to the failure → `llm_logs.retrieved_context` (what production used) → **retrieved from the domain knowledge base** (`DocumentRetriever`) when neither is present. When context is found, the teacher is constrained to answer **only** from it and the correction is NLI-verified — corrections not entailed by their context (grounding score < 0.50) are dropped, and the grounding score + source IDs are stored for later retraction.
3. **Presidio PII scrubbing** — fail-closed; drops example on any failure
4. **MinHash LSH deduplication** — Jaccard threshold 0.85
5. **Quality filter** — ROUGE-L + confidence + poison detection

Training data is formatted with the base model's **native chat template** (not Alpaca) and mixes in a ~25% **replay buffer** of known-good examples to counter catastrophic forgetting.

---

## Promotion Gates

A challenger model is only promoted when ALL of the following pass:

1. **Safety battery**: 100 adversarial prompts — Llama Guard 3 must clear all (100%)
2. **RAGAS improvement**: ≥3% absolute improvement over incumbent (eval on the seeded set, ≥`MIN_EVAL_EXAMPLES`)
3. **A/B window**: ≥1000 shadow requests over ≥48 hours, scored by a signed quality delta (LLM-judge / reference ROUGE)
4. **Statistical significance**: paired one-sample test — p < 0.05 AND Cohen's d ≥ 0.10

If any gate fails → automatic rollback. When `CANARY_ENABLED`, a small **live canary** rollout (error-rate + safety over a window) gates full promotion after these gates pass.

---

## Audit Trail

Every pipeline decision is written to `audit_trail` BEFORE execution:

- HMAC-SHA256 signed (key from HashiCorp Vault)
- INSERT-only (PostgreSQL row-level security enforces no UPDATE/DELETE)
- Verify chain integrity: `make verify-audit`

---

## Monitoring

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin / `$GRAFANA_PASSWORD`)
  - Pipeline Overview dashboard
  - Model Quality dashboard
- **PagerDuty**: configured via `PAGERDUTY_API_KEY` + `PAGERDUTY_SERVICE_ID`

---

## Testing

```bash
# All tests
make test

# Unit tests only (no external services needed)
make test-unit

# Integration tests
make test-integration

# Coverage report
make test-cov
```

**Suite status:** 154 tests collected. 148 pass locally with no external services.
The 6 `tests/integration/test_audit_integrity.py` + `test_pipeline_cycle.py` HMAC
tests require a running Vault — they **fail loudly by design** when
`VAULT_REQUIRED=true` and Vault is unreachable (this is the fail-closed audit
behaviour, not a regression). Set `VAULT_REQUIRED=false` or start Vault to run them.

---

## Production Deployment (Kubernetes)

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml      # requires External Secrets Operator + Vault
kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/pipeline-deployment.yaml
kubectl apply -f k8s/api-service.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/ingress.yaml
```

Training jobs run on Modal Labs (serverless A100) — no GPU nodes needed in your cluster.

---

## Project Structure

```
src/
├── config/          # Settings, Vault client, structured logging
├── db/              # SQLAlchemy models, async repositories, migrations
├── kafka/           # Producer, consumer, event schemas, topic config
├── middleware/       # LLM interceptor (<5ms overhead)
├── detection/        # Hallucination, drift, refusal, format detectors
├── curation/         # Clustering, teacher model, PII, dedup, quality filter
├── training/         # Trigger logic, Modal GPU worker, LoRA config, dataset builder
├── evaluation/       # RAGAS runner, safety battery, statistical tests
├── shadow/           # A/B router, metrics collector, promotion gate
├── audit/            # HMAC-signed immutable audit logger
├── graph/            # LangGraph state machine (nodes, edges, graph, runner)
├── api/              # FastAPI app and all REST endpoints
└── monitoring/       # Prometheus metrics, Grafana dashboards, PagerDuty alerts
```

---

## Production Hardening Notes

Recent fixes to the measurement and reliability layers:

- **Eval set**: the promotion gate now reads the seeded `eval_set` table (≥50 examples across 5 categories) instead of a 2-example mock, and refuses to run on too few examples outside development (`MIN_EVAL_EXAMPLES`).
- **Hallucination detection**: switched from a passage-retrieval cross-encoder to NLI entailment (`cross-encoder/nli-deberta-v3-base`), grounded on `llm_logs.retrieved_context` when present (falls back to the prompt for non-RAG calls).
- **Shadow quality metric**: replaced the circular self-comparison (always ≤ 0) with a signed LLM-judge delta (or eval-set reference ROUGE), and wired the previously no-op shadow logging so the A/B window actually accumulates samples.
- **Safety**: Llama Guard 3 (Together API) replaces keyword matching, with a loud keyword fallback when `TOGETHER_API_KEY` is unset.
- **Curation cost**: bounded teacher concurrency (`MAX_CONCURRENT_TEACHER_CALLS`), a per-run budget breaker, and a >0 sampling temperature so self-consistency voting is meaningful.
- **Training data**: native chat-template formatting (with a Llama-3 fallback) and a ~25% replay buffer of known-good examples to counter catastrophic forgetting.
- **Drift baseline** auto-refreshes after each promotion; **Vault** failures now fail loudly in production (`VAULT_REQUIRED`) and surface a degraded flag on `/health`.
- **Adaptive cycle pacing** (`CYCLE_INTERVALS`) backs off polling during the 48h A/B phase.
- **Lineage** (`GET /audit/lineage/{version}`) and **cost** (`GET /metrics/cost`) endpoints added.

### Maturity features (now implemented)

- **Hallucination grounding** — `llm_logs.retrieved_context` (from the LLM event) is now the NLI premise, giving true factual-consistency checks for RAG traffic (falls back to the prompt for non-RAG calls).
- **Durable artifacts + dataset versioning** — LoRA adapters and the exact training dataset are written to a persistent **Modal Volume** (`finetuning-artifacts`); `training_runs.dataset_uri` records the dataset, retrievable via `scripts/reproduce_dataset.py`.
- **Confidence calibration** — `ThresholdCalibrator` derives a false-positive proxy from the curation drop-rate and writes threshold suggestions to `calibration_history` (suggest-only unless `ALLOW_AUTO_CALIBRATION`). Runs every `CALIBRATION_INTERVAL_CYCLES`.
- **Canary deployment** — `CanaryController` + a graph phase gate full promotion behind a small live rollout. **Off by default** (`CANARY_ENABLED`) because it requires the serving layer to call `record_result()`/`should_route_to_canary()`; when enabled it gates promotion on error rate + safety over the window.
- **Cost circuit breaker** — tracks teacher, Modal GPU, and judge spend in Redis; the runner **skips cycles** once `MONTHLY_BUDGET_USD` is exceeded. See `GET /metrics/cost`.
- **Predictive drift early warning (RFC-001)** — fits a regression on the drift rolling window and predicts *hours until* the threshold is crossed, alerting proactively (`drift_trend_history`, `GET /drift/trend`). Off via `DRIFT_PREDICTION_ENABLED`.
- **Continuous eval factory (RFC-002)** — every N production requests, clusters recent prompts, generates GPT-4o ground-truth answers (confidence-gated, deduped), and grows the `eval_set` with `source='factory'` (LRU-evicting old factory rows; seed rows never evicted). The benchmark tracks live traffic instead of going stale. Off via `EVAL_FACTORY_ENABLED`.
- **Failure attribution (RFC-003)** — after a failure, scores which training examples most influenced it (embedding cosine; pluggable `InfluenceBackend`), stores the top-K (`failure_attributions`), and exposes a "most-blamed examples" hit-list (`GET /attribution/influential`) plus audited surgical retraction (`POST /attribution/retract`). Off via `ATTRIBUTION_ENABLED`.
- **Domain retriever (real RAG)** — a `knowledge_documents` store (MiniLM embeddings, numpy cosine — no pgvector) + `DocumentRetriever`. It's the **third tier** of the grounded teacher's context resolution (attached context → `llm_logs.retrieved_context` → **retrieved domain context**), so a domain-specific failure gets grounded against your docs even when upstream attached nothing — instead of falling back to GPT-4o's open knowledge. Seed with `scripts/seed_knowledge_base.py`; ingest/search via `/knowledge/*`. Off via `RETRIEVAL_ENABLED`.
- **RAG-grounded teacher + source tracking** — for failures whose production answer used retrieved context, the teacher must answer strictly from that context and the correction is NLI-verified against it (ungrounded corrections dropped, counter `teacher_corrections_rejected_grounding_total`). Each grounded example stores its `grounding_score` + `grounding_sources`, so if a source document changes you can find/retract affected examples via `/training/examples/by-source` (retracted examples are excluded from the next training run).

### Still intentionally out of scope

- **Multi-tenancy** — omitted; single-tenant deployment (see `src/db/models.py`).
- **S3/GCS object storage** — artifacts use a Modal Volume instead; swap in a cloud bucket if you need cross-cloud access.
