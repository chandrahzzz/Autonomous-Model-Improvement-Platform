# Continuous Fine-Tuning Pipeline — Complete Technical Reference

> Every file, every feature, every decision explained in full detail.

---

## 0. Recent Changes — Hardening Sessions (June 2026)

> **READ THIS FIRST.** The deep-dive sections below (1–19) describe the original
> design. Several components were since fixed/replaced. Where this section and a
> later section disagree, **this section is authoritative.**

### Correctness fixes (pipeline was non-operational before these)
- **DB commits**: graph nodes used `AsyncSessionLocal()` and never committed, so
  every write (promotions, **the audit trail**, training/eval rows) was rolled
  back. All write nodes now go through `get_db()` (commits on success).
- **Cross-cycle resume**: the entry point was a fixed `log_monitor`, so a running
  Modal job / open shadow window could never be re-observed. Replaced with
  `set_conditional_entry_point(route_cycle_start, …)` that resumes
  `training_poller` / `ab_test_node` / `canary_node`.
- **Statistical gate**: `welch_t_test`/`cohens_d` returned `NaN`/0 on
  zero-variance inputs (let non-improvements pass, blocked real wins). Added
  zero-variance guards + a paired one-sample test `passes_significance_gate_from_deltas`
  (removes the old ~√2 Cohen's-d inflation from the zeros-array hack).
- **RAGAS**: the model was invoked without the eval `context`; it now receives it.
- **Runner state**: `self._state` reset to hardcoded `v7` on restart; now
  persisted to Redis `pipeline:full_state` and rehydrated on startup.
- **Refusal rate**: `is_refusal()` didn't update the rolling window (rate stuck
  at 0); window update moved into `is_refusal()`.

### Measurement-layer replacements (stubs → real components)
- **Hallucination**: CLAP cross-encoder (`ms-marco-MiniLM-L-6-v2`, a *relevance*
  ranker) → **NLI entailment** (`cross-encoder/nli-deberta-v3-base`). Premise is
  the new `llm_logs.retrieved_context` (RAG grounding) when present, else the prompt.
  Setting renamed `clap_hallucination_threshold` → `hallucination_threshold` (0.50).
- **Shadow quality metric**: the old `ROUGE(prod,prod)` baseline was always 1.0,
  so every challenger scored ≤ 0 (promotion was mathematically impossible). Now a
  signed **LLM-judge** delta (default, `gpt-4o-mini`) or **reference-ROUGE** vs eval
  ground truth (`SHADOW_SCORING_STRATEGY`). Also wired `_log_shadow` (was a no-op,
  so `shadow_logs` was never populated).
- **Safety battery**: keyword matching (`"sorry"` passed harmful answers) →
  **Llama Guard 3** via Together (`SAFETY_CLASSIFIER`, `TOGETHER_API_KEY`), with a
  loud keyword fallback. Added 10 polite-framed bypass prompts.
- **Eval set**: the node used a hardcoded 2-example mock and ignored the seeded
  `eval_set` table. Now loads via `EvalSetRepository` with a `MIN_EVAL_EXAMPLES`
  guard; `tests/fixtures/eval_set.json` seeds **50** examples across 5 categories.
- **Dataset format**: Alpaca `### Instruction/### Response` → the model's native
  **chat template** (`tokenizer.apply_chat_template`, Llama-3 fallback string).
- **Replay buffer**: training was 100% failures (catastrophic forgetting). Now
  mixes ~25% known-good logs (`get_known_good_sample`, `REPLAY_RATIO`).
- **Teacher**: 3 self-consistency votes at temp 0 were identical (meaningless);
  now a `>0` sampler temp. Added an `asyncio.Semaphore(MAX_CONCURRENT_TEACHER_CALLS)`
  and a per-run cost breaker (`CURATION_COST_BUDGET_USD`).

### New capabilities
- **Durable artifacts**: LoRA weights + the exact dataset now save to a persistent
  **Modal Volume** (`finetuning-artifacts`), not ephemeral `/tmp`.
  `training_runs.dataset_uri` records the dataset; `scripts/reproduce_dataset.py`
  fetches it.
- **Drift baseline auto-refresh**: recomputed after each promotion in
  `promote_model_node` (was seed-once, grew stale → false drift).
- **Vault**: silent `.env` fallback → **fail-loud in prod** (`VAULT_REQUIRED`),
  with a degraded flag on `/health`.
- **Adaptive pacing**: `CYCLE_INTERVALS` backs the loop off during the 48h A/B
  (and 1h canary) windows instead of polling every 60s.
- **Cost circuit breaker**: tracks teacher + Modal GPU + judge spend in Redis;
  the runner **skips cycles** once `MONTHLY_BUDGET_USD` is hit. `GET /metrics/cost`.
- **Threshold calibration**: `ThresholdCalibrator` suggests adjustments from the
  curation drop-rate → `calibration_history` (suggest-only unless
  `ALLOW_AUTO_CALIBRATION`); runs every `CALIBRATION_INTERVAL_CYCLES`.
- **Canary**: `CanaryController` + a `canary_node` graph phase gating full
  promotion behind a small live rollout. **Off by default** (`CANARY_ENABLED`) —
  needs the serving layer to call `record_result()`. `GET/POST /shadow/canary/*`.
- **Lineage**: `GET /audit/lineage/{version}` traces logs → examples → run → model.
- **RAG-grounded teacher + source tracking**: when a failure's production answer
  used retrieved context, the teacher is constrained to answer ONLY from that
  context and the correction is NLI-verified against it (reusing the
  `failure_detector._hall` singleton; entailment = `1 - score_batch`). Corrections
  with grounding score < 0.50 are dropped (counter
  `teacher_corrections_rejected_grounding_total`); `INSUFFICIENT_CONTEXT` from the
  teacher also drops the example. `generate_correction()` now returns a
  `GroundingResult` (not a tuple) carrying `grounding_score` + `grounding_sources`.
  This stops GPT-4o silently "correcting" domain-specific (policy/pricing/internal)
  failures with wrong outside knowledge. Source docs that change → find/retract
  affected examples via `/training/examples/by-source`; retracted examples are
  excluded from `get_pending` (the next training run).

### Predictive / observability RFCs (all additive, kill-switched, fire-and-forget)
- **RFC-001 — Predictive drift early warning** (`src/detection/drift_predictor.py`):
  fits a linear regression on the DriftDetector's rolling `_window` of Mahalanobis
  distances and predicts *hours until* the 0.15 threshold is crossed — proactive,
  not reactive. Runs every `DRIFT_PREDICTION_INTERVAL_CYCLES` in `failure_detector_node`;
  alert + persist happen via `asyncio.create_task`. Persists to `drift_trend_history`
  (migration **006**); PagerDuty warning is deduped (`alerter.trigger`, 2h window);
  exposed at `GET /drift/trend|/trend/history|/trend/alarms`. Off via `DRIFT_PREDICTION_ENABLED`.
- **RFC-002 — Continuous eval factory** (`src/evaluation/eval_factory.py`): every
  `EVAL_FACTORY_TRIGGER_EVERY_N_REQUESTS` production requests, clusters recent prompts
  (HDBSCAN), picks a medoid per cluster, GPT-4o generates a ground-truth answer
  (3-vote self-consistency), dedups (cosine ≥ 0.90 vs stored embeddings), and inserts
  into `eval_set` with `source='factory'`. LRU-evicts oldest *factory* rows over the
  size cap (seed rows never evicted). Triggered from `log_monitor_node` via a Redis
  counter; runs in a background task. **Created the `EvalSet` ORM model** (the table
  was raw-SQL-only before); migration **007** adds `source/cluster_id/cluster_label/
  factory_confidence/access_count/last_accessed_at/evicted_at/embedding`. `eval_runner`
  now `mark_accessed()`es the examples it used. Endpoints `GET /eval/set/summary|
  examples`, `GET /eval/set/factory/history`, `POST /eval/set/factory/trigger`,
  `DELETE /eval/set/examples/{id}` (seed → 403). Off via `EVAL_FACTORY_ENABLED`.
- **RFC-003 — Failure attribution** (`src/attribution/`): after a failure, scores
  which training examples most influenced it via cosine similarity in MiniLM space
  (`InfluenceBackend` Protocol + `EmbeddingInfluenceBackend` — swappable for a
  gradient backend later), stores top-K in `failure_attributions` (migration **008**).
  Runs fire-and-forget in `failure_detector_node` (capped at
  `ATTRIBUTION_MAX_FAILURES_PER_CYCLE`; `encode` wrapped in `run_in_executor`).
  Endpoints `GET /attribution/log/{id}|/model/{v}|/influential`, and `POST
  /attribution/retract` (audit-before-act → marks `retracted_at`, excluded from
  training). Off via `ATTRIBUTION_ENABLED`. NOTE: `training_run_id` is an **integer**
  FK (training_runs.id is integer, not UUID); `FailureEvent` uses `llm_log_id`.
- **Domain knowledge retrieval — the real RAG retriever** (`src/retrieval/retriever.py`):
  closes the gap where the grounded teacher could only ground if the upstream app
  *attached* `retrieved_context`. `knowledge_documents` (migration **009**) stores
  domain docs + MiniLM embeddings (JSONB); `DocumentRetriever.retrieve()` encodes a
  query and scans with **numpy cosine** (no pgvector, like the rest of the codebase),
  returning top-K above `RETRIEVAL_MIN_SIMILARITY` as a context string with
  `[doc_id: …]` markers (so the teacher's `_extract_source_ids` records which docs
  grounded the correction). Wired as the **third tier** of `teacher._resolve_context`
  (attached context → `llm_logs.retrieved_context` → **retrieved domain context**), so
  a domain failure with no attached context now grounds against the KB instead of
  GPT-4o's open knowledge. DB-first short-circuit means an empty KB never loads the
  model. Seed via `scripts/seed_knowledge_base.py`; ingest/search via `/knowledge/*`.
  Threshold **calibrated to 0.30** (relevant queries ~0.38-0.42, off-topic < 0.05).
  Off via `RETRIEVAL_ENABLED`. This makes the answer to "is RAG implemented here?"
  finally **yes** (numpy-cosine vector store + retrieve→augment→ground for the teacher).

### New files / schema / endpoints
- New modules: `src/db/repositories/eval_set.py`, `src/detection/calibrator.py`,
  `src/shadow/canary.py`, `src/monitoring/cost_tracker.py`, `src/api/routers/training.py`.
- New scripts: `scripts/reproduce_dataset.py`.
- Migration `004_grounding_versioning_calibration.py`: `llm_logs.retrieved_context`,
  `training_runs.dataset_uri`, `calibration_history` table.
- Migration `005_grounded_teacher_source_tracking.py`: `training_examples.grounding_score`,
  `grounding_sources` (`TEXT[]`), `retracted_at` + partial index `ix_te_grounding_score`
  and GIN index `ix_te_grounding_sources`.
- Migration `006_drift_trend_history.py` (RFC-001): `drift_trend_history` table.
- Migration `007_eval_factory.py` (RFC-002): 8 columns + 2 indexes on `eval_set`.
- Migration `008_failure_attribution.py` (RFC-003): `failure_attributions` table.
- Migration `009_knowledge_base.py` (retriever): `knowledge_documents` table + index.
  **All migrations applied + downgrade-tested against live Postgres (head = 009).**
- More new modules: `src/detection/drift_predictor.py`, `src/db/repositories/drift_trend.py`,
  `src/evaluation/eval_factory.py`, `src/attribution/{influence,attributor}.py`,
  `src/db/repositories/attribution.py`, `src/retrieval/retriever.py`,
  `src/db/repositories/knowledge.py`,
  `src/api/routers/{drift,eval,attribution,knowledge}.py`.
- More new scripts: `scripts/eval_factory_status.py`, `scripts/attribution_report.py`,
  `scripts/seed_knowledge_base.py`.
- New endpoints: `/metrics/cost`, `/audit/lineage/{version}`,
  `/shadow/canary/*`, `GET|DELETE /training/examples/by-source`,
  `/drift/trend*`, `/eval/set/*`, `/attribution/*`, `/knowledge/*`.
- New unit tests: `test_hallucination.py`, `test_canary.py`, `test_dataset_builder.py`,
  `test_calibrator.py`, `test_teacher_grounding.py`, `test_drift_predictor.py` (12),
  `test_eval_factory.py` (12), `test_attribution.py` (11), `test_retriever.py` (8)
  — suite now **96 passing**.

### Decisions / still out of scope
- **Multi-tenancy**: intentionally omitted (single-tenant) — see `src/db/models.py`.
- **S3/GCS**: not used; durable storage is the Modal Volume above.

---

## Table of Contents

1. [What This System Does (Big Picture)](#1-what-this-system-does)
2. [How Data Flows End-to-End](#2-data-flow)
3. [Infrastructure & Configuration](#3-infrastructure--configuration)
4. [Database Layer](#4-database-layer)
5. [Kafka Streaming Layer](#5-kafka-streaming-layer)
6. [LLM Interceptor Middleware](#6-llm-interceptor-middleware)
7. [Failure Detection System](#7-failure-detection-system)
8. [Curation Pipeline](#8-curation-pipeline)
9. [Training System](#9-training-system)
10. [Evaluation System](#10-evaluation-system)
11. [Shadow A/B Testing](#11-shadow-ab-testing)
12. [Promotion Gate](#12-promotion-gate)
13. [Audit Trail](#13-audit-trail)
14. [LangGraph State Machine](#14-langgraph-state-machine)
15. [FastAPI Layer](#15-fastapi-layer)
16. [Monitoring & Alerting](#16-monitoring--alerting)
17. [Scripts & Utilities](#17-scripts--utilities)
18. [Alembic Migrations](#18-alembic-migrations)
19. [File-by-File Reference](#19-file-by-file-reference)

---

## 1. What This System Does

This is a **fully autonomous LLM continuous fine-tuning pipeline**. It:

1. Watches every LLM API call your application makes
2. Automatically detects when the model starts failing (hallucinations, drift, refusing too much, broken format)
3. Collects those failure examples, gets a teacher model (GPT-4o) to write the correct answers
4. Cleans and validates the examples (removes PII, deduplicates, quality-filters)
5. Fires a LoRA training job on Modal Labs A100 GPU when enough examples accumulate
6. Evaluates the trained challenger model against a held-out eval set using RAGAS metrics
7. Runs a 48-hour shadow A/B test (challenger runs silently alongside production)
8. Promotes the challenger to production ONLY if it passes 4 hard gates
9. If any gate fails — automatically rolls back, writes an audit entry, and starts over
10. Repeats forever, 24/7, with no human required

**Zero human intervention after initial setup.** The graph never stops.

---

## 2. Data Flow

```
Your Application
      │  (HTTP request with X-LLM-Call: 1 header)
      ▼
┌─────────────────────────────────────┐
│  FastAPI + LLMInterceptorMiddleware │  ← adds <5ms latency
│  Captures: prompt, completion,      │
│  tokens, latency, cost, model ver.  │
└──────────────┬──────────────────────┘
               │ asyncio.create_task() — fire and forget
               ▼
        Kafka Topic: llm.production.events
               │
               │ (LangGraph runner polls DB every 60s)
               ▼
┌──────────────────────────────────────────┐
│           LangGraph State Machine        │
│                                          │
│  log_monitor_node                        │
│    └─► pulls last 500 events from DB     │
│                                          │
│  failure_detector_node                   │
│    └─► runs 4 detectors in parallel:     │
│         • HallucinationDetector (NLI)    │
│         • DriftDetector (Mahalanobis)    │
│         • RefusalDetector (regex+sem)    │
│         • FormatValidator (JSON+KL div)  │
│                                          │
│  [if failures] → example_curator_node   │
│    └─► CurationPipeline:                │
│         1. HDBSCAN cluster failures      │
│         2. GPT-4o generates correction   │
│         3. Presidio scrubs PII           │
│         4. MinHash LSH deduplication     │
│         5. Quality filter (ROUGE-L)      │
│         6. INSERT into training_examples │
│                                          │
│  data_validator_node                     │
│    └─► counts pending examples           │
│                                          │
│  fine_tune_trigger_node                  │
│    └─► checks 3 conditions:              │
│         • examples ≥ 500                 │
│         • drift_score ≥ 0.15             │
│         • ≥ 6h since last run            │
│                                          │
│  [if triggered] → lora_trainer_node      │
│    └─► builds JSONL dataset              │
│    └─► submits to Modal Labs A100        │
│                                          │
│  training_poller_node                    │
│    └─► polls Modal every 60s             │
│    └─► [completed] → eval_runner_node    │
│    └─► [failed] → rollback               │
│                                          │
│  eval_runner_node                        │
│    └─► safety battery (100 prompts)      │
│    └─► RAGAS (faithfulness/relevancy/    │
│                context_recall)           │
│    └─► [passed] → ab_test_node           │
│    └─► [failed] → rollback               │
│                                          │
│  ab_test_node                            │
│    └─► collects shadow traffic stats     │
│    └─► waits for 1000 req + 48h          │
│                                          │
│  promotion_decider_node                  │
│    └─► PromotionGate (4 gates)           │
│    └─► [promote] → audit + promote       │
│    └─► [reject] → audit + rollback       │
│                                          │
└──────────────────────────────────────────┘
```

---

## 3. Infrastructure & Configuration

### `src/config/settings.py`
**Purpose:** Single source of truth for all configuration values.

Uses `pydantic-settings` `BaseSettings` which automatically reads from `.env` file and environment variables. Key behaviors:

- `extra="ignore"` — unknown env vars are silently ignored (safe for containerized envs with extra vars)
- `case_sensitive=False` — `DATABASE_URL` and `database_url` both work
- Two custom validators:
  - `validate_db_url`: forces `postgresql+asyncpg://` prefix (asyncpg requires this; plain `postgresql://` would fail silently)
  - `parse_target_modules`: parses LoRA target modules from either JSON (`["q_proj","v_proj"]`) or comma-separated string

**Key settings groups:**

| Group | Key Settings | Purpose |
|---|---|---|
| Database | `database_url`, `pool_size=20`, `pool_timeout=30` | asyncpg connection pool |
| Kafka | `bootstrap_servers`, topic names | event streaming |
| Training trigger | `dataset_size=500`, `drift_threshold=0.15`, `min_interval=6h` | when to fire |
| LoRA | `r=16`, `alpha=32`, `epochs=3`, `lr=2e-4` | adapter hyperparameters |
| Eval | `improvement_threshold=0.03`, `ab_min_requests=1000`, `ab_min_hours=48` | promotion gates |
| Detection | `hallucination_threshold=0.50` (NLI), `drift_threshold=0.15`, `refusal_multiplier=2.0` | failure sensitivity |
| Curation | `confidence_threshold=0.85`, `jaccard_threshold=0.85` | quality gates |

**Usage:** `from src.config.settings import settings` — singleton created at import time.

---

### `src/config/vault.py`
**Purpose:** Fetches secrets from HashiCorp Vault at runtime (not at deploy time).

- `VaultClient.get_secret(path, key)` — reads KV v2 secrets. Raises `RuntimeError` on failure (fail-closed — never silently returns empty string)
- `VaultClient.get_hmac_key()` — specifically fetches the HMAC signing key for audit trail
- `@lru_cache(maxsize=1)` singleton — Vault connection created once per process
- Vault unavailable: in production (`VAULT_REQUIRED=true`) this now **fails loudly**; only in dev does it fall back to `settings.secret_key`, setting a degraded flag exposed on `/health` (see §0)

**Why Vault?** HMAC keys must be rotatable without redeploying. Vault also provides audit logs of who accessed what secret and when.

---

### `src/config/logging.py`
**Purpose:** Configures `structlog` for structured logging.

- Development: colored console output with human-readable timestamps
- Production: JSON output (one JSON object per line — works with log aggregators like Datadog/Splunk)
- All standard library `logging` calls are also routed through structlog
- Log level: `DEBUG` if `settings.debug=True`, else `INFO`

**Why structlog?** Structured logs have named fields (`{"event": "training_triggered", "version": "v8", "examples": 523}`) that can be queried in log aggregators. Regular `print()` / f-string logs are unqueryable.

---

## 4. Database Layer

### `src/db/connection.py`
**Purpose:** Manages the asyncpg connection pool.

```python
engine = create_async_engine(
    database_url,
    pool_size=20,         # 20 persistent connections
    max_overflow=10,      # up to 10 additional burst connections
    pool_pre_ping=True,   # test connection before use (detects stale connections)
    pool_recycle=3600,    # recycle connections after 1 hour (prevents TCP timeout issues)
)
```

- `get_db()` — async context manager that auto-commits on success, auto-rollbacks on exception
- `check_database_health()` — runs `SELECT 1` to verify DB connectivity. Called at FastAPI startup; raises `RuntimeError` if DB is unreachable (fail-fast, not fail-silent)

**Why pool_pre_ping?** Cloud databases (RDS, Cloud SQL) kill idle connections. Without pre-ping, the first query after a long idle period fails. Pre-ping catches this and reconnects automatically.

---

### `src/db/models.py`
**Purpose:** All SQLAlchemy 2.0 ORM table definitions.

#### Table: `llm_logs`
Stores every LLM API call intercepted by the middleware. Fields:
- `id` — UUID primary key
- `session_id`, `user_cohort` — for grouping and filtering
- `model_version` — which model version served this request (e.g., "v7")
- `prompt`, `completion` — the actual text (stored for failure detection)
- `prompt_tokens`, `completion_tokens`, `cost_usd` — usage tracking
- `latency_ms` — response time in milliseconds
- `finish_reason` — why the model stopped (stop/length/content_filter)
- `embedding_hash` — for fast near-duplicate detection without full text comparison
- `metadata` — JSONB field for arbitrary extra data

#### Table: `failure_classifications`
Every failure event detected by the 4 detectors. Foreign-keys to `llm_logs`.
- `failure_type` — one of: `hallucination`, `semantic_drift`, `refusal_creep`, `format_regression`
- `score` — severity score from 0.0 to 1.0
- `cluster_id`, `cluster_label` — HDBSCAN cluster assignment (for diverse sampling)

#### Table: `training_examples`
Curated, clean training examples ready to be included in a training run.
- `prompt`, `bad_completion`, `corrected_completion` — the (input, wrong answer, right answer) triple
- `failure_type` — what kind of failure this example targets
- `teacher_model`, `teacher_confidence` — which model corrected it, and how confident
- `pii_scrubbed` — boolean, whether Presidio ran on this example
- `dedup_hash` — SHA-256 of `prompt\x00completion` — unique index prevents duplicates
- `quality_score` — `teacher_confidence * (1 - rouge_score)` — higher is better
- `included_in_run` — FK to `training_runs.id` once used; NULL means "pending"

#### Table: `model_versions`
Registry of all trained model versions.
- `version_tag` — e.g., "v7", "v8", "v9" — human-readable version
- `base_model` — the HuggingFace model ID (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`)
- `lora_weights_path` — path to saved LoRA adapter weights (on Modal volume or S3)
- `is_production` — boolean; only ONE row has this True at any time (atomic swap)
- `is_archived` — rolled-back models
- `promoted_at`, `rolled_back_at` — timestamps for audit

#### Table: `training_runs`
Tracks each Modal GPU training job.
- `modal_job_id` — Modal function call ID for polling
- `status` — submitted / completed / failed
- `dataset_size` — how many examples were used
- `lora_config` — JSONB snapshot of LoRA hyperparameters at training time
- `final_loss`, `wandb_run_id`, `wandb_run_url` — training outcome

#### Table: `eval_runs`
Stores results of every RAGAS + safety battery evaluation.
- `faithfulness`, `answer_relevancy`, `context_recall` — RAGAS scores
- `safety_score` — pass rate of safety battery (must be 1.0)
- `ab_quality_delta`, `ab_pvalue`, `ab_cohens_d` — statistical test results
- `gate_passed` — whether this eval resulted in promotion

#### Table: `audit_trail`
Immutable log of every pipeline decision. **INSERT-ONLY** — PostgreSQL row-level security makes UPDATE and DELETE impossible.
- `event_type` — one of: `training_triggered`, `model_promoted`, `model_rolled_back`
- `decision` — human-readable description of what happened
- `rationale` — JSONB with all the numbers that drove the decision
- `state_snapshot` — snapshot of pipeline state at decision time
- `operator` — "autonomous_pipeline" for automated, "human_operator" for manual actions
- `hmac_sha256` — HMAC-SHA256 signature of the entry (tamper detection)

#### Table: `drift_baselines`
Stores the statistical baseline used for Mahalanobis drift detection.
- `centroid` — JSONB array: mean embedding vector of known-good outputs
- `covariance_inv` — JSONB 2D array: inverse covariance matrix
- `is_active` — only one baseline is active at a time

---

### `src/db/repositories/` — The Repository Pattern

Each repository wraps one or more tables and provides typed async methods. Nodes never write raw SQL.

#### `llm_logs.py` — `LLMLogRepository`
- `insert(data)` — write a new LLM event
- `get_recent(limit=1000, hours=1)` — fetch the last N events from the past H hours
- `count_since(hours=24)` — count events in time window (for throughput metrics)
- `get_by_id(log_id)` — single record lookup

#### `model_versions.py` — `ModelRepository`
- `get_production_version()` — returns the single row with `is_production=True`
- `promote(version_tag)` — atomically: sets ALL rows to `is_production=False`, then sets the target to `True` + `promoted_at=now()`
- `rollback(version_tag)` — sets `rolled_back_at=now()`, `is_production=False`
- `create_training_run(data)` — inserts a new training job record
- `update_training_run(run_id, data)` — updates job status, final_loss, wandb URL
- `get_active_baseline()` — returns drift baseline centroid + covariance
- `save_baseline(model_version, data)` — deactivates old baseline, inserts new one

#### `training_examples.py` — `TrainingExampleRepository`
- `upsert(data)` — INSERT with `ON CONFLICT DO NOTHING` on `dedup_hash` — safe to call with duplicates
- `count_pending()` — count examples with `included_in_run IS NULL`
- `get_pending(limit=2000)` — ordered by `quality_score DESC` — best examples first
- `mark_used(ids, run_id)` — mark examples as consumed by a training run

#### `eval_runs.py` — `EvalRunRepository`
- `create(data)`, `update(run_id, data)`, `get_latest_for_version(tag, type)`, `get_all_for_run(training_run_id)`

#### `audit_trail.py` — `AuditRepository`
**Only** exposes: `insert()`, `get_recent()`, `get_by_id()`, `get_chain()`. **No update or delete methods** — by design. Even if you import this class, you cannot modify audit entries at the Python level. PostgreSQL RLS enforces the same constraint at the DB level.

---

## 5. Kafka Streaming Layer

### `src/kafka/producer.py` — `AsyncKafkaProducer`
**Purpose:** Reliably emits LLM events to Kafka with exactly-once semantics.

Configuration:
```python
"acks": "all"              # wait for all replicas to acknowledge
"enable.idempotence": True  # exactly-once delivery (deduplicates retries)
"compression.type": "snappy" # ~40% size reduction, fast decompression
"linger.ms": 5             # batches messages for 5ms (throughput vs latency tradeoff)
"batch.size": 65536        # 64KB batch size
"retries": 3               # retry up to 3 times on transient failure
```

The `produce()` method:
1. Creates an `asyncio.Future`
2. Calls `producer.produce()` with a delivery callback
3. The callback resolves the future on success or sets an exception on failure
4. `await asyncio.wait_for(future, timeout=10.0)` — waits for broker acknowledgement
5. On failure: sends message to DLQ (Dead Letter Queue) topic — **never silently drops**

`get_producer()` — module-level singleton, created once per process.

### `src/kafka/consumer.py` — `AsyncKafkaConsumer`
**Purpose:** Processes incoming Kafka messages with at-least-once delivery.

Configuration:
```python
"enable.auto.commit": False  # manual commit AFTER processing
"auto.offset.reset": "earliest"  # start from beginning if no committed offset
"max.poll.interval.ms": 300000   # 5 min max between polls (for slow processing)
```

The `_consume_loop()`:
1. Polls for messages using `run_in_executor` (non-blocking)
2. Parses JSON message
3. Calls `self._handler(value, topic)` — user-provided async function
4. Only commits offset AFTER handler completes successfully
5. On handler exception: does NOT commit — message will be redelivered (at-least-once)

### `src/kafka/topics.py`
Defines all 3 topic configs as frozen dataclasses:
- `llm.production.events` — 12 partitions, 7-day retention (high volume)
- `pipeline.training.events` — 3 partitions, 30-day retention (low volume)
- `pipeline.dlq` — 3 partitions, 30-day retention (failed messages for investigation)

`create_topics(bootstrap_servers)` — idempotent: safe to call multiple times; silently ignores "already exists" errors.

### `src/kafka/schemas/llm_event.py` — `LLMEvent`
Pydantic model for the Kafka message schema. Auto-generates `event_id` (UUID) and `timestamp` (ISO datetime) if not provided.

---

## 6. LLM Interceptor Middleware

### `src/middleware/llm_interceptor.py` — `LLMInterceptorMiddleware`

**Purpose:** Transparent HTTP middleware that captures every LLM API call with under 5ms overhead.

**How it decides to intercept a request:**
```python
def _is_llm_endpoint(self, request):
    return (
        request.headers.get("X-LLM-Call") == "1"  # explicit opt-in
        or "/completions" in path                   # OpenAI-style endpoints
        or path.endswith("/chat")                   # chat endpoints
    )
```

**How it works without blocking:**
```python
async def dispatch(self, request, call_next):
    start = time.monotonic()
    response = await call_next(request)  # serve the response FIRST
    latency_ms = int((time.monotonic() - start) * 1000)

    if self._is_llm_endpoint(request):
        asyncio.create_task(self._emit_event(...))  # fire-and-forget

    return response  # user never waits for Kafka
```

The response is returned to the user BEFORE the Kafka event is emitted. `asyncio.create_task()` schedules the emission on the event loop without blocking.

**Metadata extraction:** The middleware reads `X-LLM-Meta` header (JSON-encoded) for prompt tokens, completion tokens, model name, finish reason. To use this, your LLM client should attach this header.

**Cost computation:**
```python
TOKEN_COSTS = {
    "gpt-4o":    {"input": 5e-6,    "output": 15e-6},    # $5/1M input, $15/1M output
    "gpt-4-turbo": {"input": 10e-6, "output": 30e-6},
    "llama-3-8b":  {"input": 0.05e-6, "output": 0.05e-6},
}
cost = prompt_tokens * costs["input"] + completion_tokens * costs["output"]
```

**Prometheus metrics updated:** `llm_calls_total`, `llm_latency_seconds`, `llm_tokens_total`, `llm_cost_usd_total`

---

## 7. Failure Detection System

All 4 detectors are instantiated as module-level singletons in `failure_detector.py` so their state (rolling windows, loaded models) persists across pipeline cycles.

### `src/detection/hallucination.py` — `HallucinationDetector`

**Method:** CLAP (Cross-encoder Language-Agnostic Pairing) using `cross-encoder/ms-marco-MiniLM-L-6-v2`.

**How it works:**
1. Takes `(prompt, completion)` pairs
2. Passes them to a cross-encoder (a BERT model that reads both texts together)
3. The model scores how well the completion is supported by the prompt context
4. High relevance = completion is grounded = NOT a hallucination
5. We invert: `hallucination_prob = 1 - sigmoid(raw_score)`
6. If `hallucination_prob > 0.70` → hallucination detected

**Why cross-encoder vs embedding similarity?** Embedding models encode each text independently. Cross-encoders read both texts together, allowing them to detect subtle factual contradictions. Much more accurate for factual consistency.

**Performance:** Batches up to 32 pairs, runs in `ThreadPoolExecutor(max_workers=4)` to avoid blocking asyncio event loop. ~10ms per pair on CPU.

---

### `src/detection/drift.py` — `DriftDetector`

**Method:** Mahalanobis distance in sentence embedding space.

**Setup (done once via `scripts/seed_baseline.py`):**
1. Collect ~10,000 known-good production outputs
2. Encode them with `sentence-transformers/all-MiniLM-L6-v2` → 384-dim vectors
3. Compute: centroid (mean vector) + inverse covariance matrix
4. Save to `drift_baselines` table

**Per-request scoring:**
1. Encode the new completion → 384-dim vector
2. `mahalanobis(embedding, centroid, cov_inv)` — measures statistical distance
3. Append to `deque(maxlen=1000)` rolling window
4. `rolling_drift_score = mean(window)`

**Why Mahalanobis?** Regular Euclidean distance ignores correlations between dimensions. Cosine similarity ignores magnitude. Mahalanobis normalizes by the data's own covariance structure — equivalent to "how many standard deviations from the center, accounting for how the data is shaped."

**Regularization:** `cov += 1e-6 * I` — prevents singular matrix if some embedding dimensions are perfectly correlated.

**Trigger:** `is_drifting()` returns True when `rolling_drift_score > 0.15`. The rolling window means a single outlier doesn't trigger — requires sustained drift.

---

### `src/detection/refusal.py` — `RefusalDetector`

**Method:** Two-stage — fast keyword regex, then semantic similarity.

**Stage 1 — Regex (fast, catches obvious cases):**
```python
REFUSAL_KEYWORDS = [
    r"\bi can'?t\b", r"\bi won'?t\b", r"\bi'm unable\b",
    r"\bi cannot\b", r"\bas an ai\b", r"i don'?t have the ability",
    r"I apologize, but", r"I'm not able to", r"I must decline",
    r"it would be inappropriate", r"that'?s not something i",
]
```
If matched → `(True, 1.0)` — certain refusal.

**Stage 2 — Semantic (catches paraphrased refusals):**
- Encode 4 exemplar refusals as tensors at init time (cached)
- Cosine similarity of new completion vs exemplars
- If `max_sim > 0.75` → refusal detected with that confidence score

**Rate tracking:** `deque(maxlen=500)` tracks last 500 booleans (is_refusal per request). `current_refusal_rate = sum(window) / len(window)`. `is_creeping()` returns True if rate > `baseline_rate * 2.0`. Default baseline: 5%, so triggers at 10%.

---

### `src/detection/format_validator.py` — `FormatValidator`

**Two signals:**

**Signal 1 — JSON Validation (optional):**
Call `set_expect_json(True)` to activate. If the completion fails `json.loads()`, immediately returns `(True, 1.0)` — format regression.

**Signal 2 — Length Distribution KL Divergence:**
- Baseline: histogram of output lengths (in 100-word bins, 0-4000 words)
- Current: histogram of last 500 outputs
- `KL(current || baseline)` — measures how much the distribution has shifted
- KL = 0 means identical distribution; higher = more divergence
- Trigger: `KL > 0.5`

**KL divergence formula:**
```
KL(P || Q) = sum(P * log(P / Q))
```
With `eps=1e-10` smoothing to avoid `log(0)`.

---

### `src/detection/failure_classifier.py` — `FailureClassifier`

**Purpose:** Orchestrates all 4 detectors concurrently.

```python
hall_scores, refusal_results, format_results = await asyncio.gather(
    self._hall.score_batch(hall_pairs),
    self._refusal.classify_batch(log_events),
    self._format.validate_batch(log_events),
)
```

All three run simultaneously. Drift is scored per-item sequentially (updates rolling window).

Returns `FailureBatch` with:
- `events: list[FailureEvent]` — each failure with type, score, log_id
- `drift_score: float` — current rolling mean Mahalanobis distance
- `has_failures: bool` — True if any detector fired

Increments Prometheus counters per failure type.

---

## 8. Curation Pipeline

### `src/curation/clustering.py` — `FailureClusterer`

**Purpose:** Group similar failures together before curation to ensure training data diversity.

**Method:** HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
- Encode each `prompt + completion` with `all-MiniLM-L6-v2`
- Run HDBSCAN with `min_cluster_size=5`, `metric="euclidean"`, `cluster_selection_method="eom"`
- Noise points (cluster_id=-1) are failures that don't form a coherent cluster

**Why HDBSCAN over k-means?**
- No need to specify number of clusters upfront
- Handles arbitrarily shaped clusters
- Explicitly identifies noise/outliers
- Works well in high-dimensional spaces

**Effect:** If 200 hallucinations are detected but 180 are about the same topic, HDBSCAN will group them into one cluster. We can then sample proportionally, getting diverse training examples rather than 180 copies of the same failure pattern.

---

### `src/curation/teacher.py` — `TeacherModel`

**Purpose:** Generate ideal corrected responses using GPT-4o.

> **UPDATED (RAG grounding — see §0).** `generate_correction()` now returns a
> `GroundingResult` (not a `(text, confidence)` tuple). When the failure's
> production answer used retrieved context, the teacher is given a grounded
> prompt ("answer ONLY from the context, else reply `INSUFFICIENT_CONTEXT`") and
> the chosen correction is NLI-verified against that context (reusing the
> `failure_detector._hall` singleton, lazily). Corrections with grounding score
> < 0.50 — or an `INSUFFICIENT_CONTEXT` reply — are dropped. `grounding_score`
> and `grounding_sources` are persisted on `training_examples`. The 3-vote
> self-consistency logic below is unchanged.

**Self-consistency scoring (3-way vote):**
```python
corrections = await asyncio.gather(*[
    self._single_correction(failure) for _ in range(3)
])
confidence = self._compute_consistency_score(corrections)
```

Generates 3 independent corrections. If they agree (high ROUGE-L overlap between them), confidence is high. If they diverge, the failure case is ambiguous — low confidence → drop the example.

**Why self-consistency?** GPT-4o at temperature=0 still has uncertainty. If 3 independent samples give very different answers, the correction is unreliable. Only high-consistency corrections enter the training set.

**Threshold:** `teacher_confidence_threshold=0.85` — if mean pairwise ROUGE-L < 0.85, the example is rejected.

---

### `src/curation/pii_scrubber.py` — `PIIScrubber`

**Purpose:** Remove personally identifiable information from training examples.

**Detected entity types:** PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN, IP_ADDRESS, LOCATION, DATE_TIME, NRP (National Registration Number / Passport)

**Replacement:** Each entity type gets a placeholder: `<PERSON>`, `<EMAIL_ADDRESS>`, etc. This preserves the structure of the text while removing identifying details.

**Fail-closed behavior:** If `scrub()` raises ANY exception (model load failure, encoding error, etc.):
- Returns `("", "", False)`
- Caller must drop the example
- Controlled by `settings.pii_fail_closed=True`
- This means: **we would rather lose a training example than accidentally train on PII**

---

### `src/curation/deduplicator.py` — `Deduplicator`

**Two-level deduplication:**

**Level 1 — Exact hash:**
```python
SHA-256(prompt + "\x00" + completion)
```
Stored in a Python `set`. If seen before → reject. Also enforced by `UNIQUE INDEX` on `dedup_hash` column in DB.

**Level 2 — Approximate hash (MinHash LSH):**
- `MinHash(num_perm=128)` — creates a 128-hash signature for each document
- `MinHashLSH(threshold=0.85)` — indexes signatures, returns approximate nearest neighbors
- If Jaccard similarity (word-level overlap) > 0.85 → near-duplicate → reject

**Why MinHash LSH?** Exact hashing misses paraphrases. Two examples with the same prompt but slightly different wording of the correction would both pass exact dedup but are near-duplicates. MinHash LSH is O(1) lookup with tunable similarity threshold.

**Jaccard similarity:** `|A ∩ B| / |A ∪ B|` — fraction of words shared. 0.85 threshold means "reject if 85% of the words are the same."

---

### `src/curation/quality_filter.py` — `QualityFilter`

**Purpose:** Final quality gate before DB insertion.

**Six rejection reasons:**
1. `poison_detected` — correction contains jailbreak/injection patterns
2. `poison_in_prompt` — prompt itself is adversarial (shouldn't be in training set)
3. `low_teacher_confidence` — ROUGE-L self-consistency below 0.85
4. `correction_too_similar_to_bad_output` — ROUGE-L(bad, corrected) > 0.95 (teacher just copied the bad answer)
5. `correction_too_short` — less than 5 words
6. `low_quality_score` — `teacher_confidence * (1 - rouge_score) < 0.30`

**Poison patterns detected:**
- "ignore previous instructions"
- "disregard your system"
- "you are now an uncensored AI"
- "jailbreak", "DAN mode"
- "pretend you have no restrictions"

**Quality score formula:** `quality_score = teacher_confidence * (1.0 - rouge_score)`
- High teacher confidence = good correction
- Low ROUGE-L = meaningfully different from bad output (not a copy)
- Both must be high for a high quality score

---

### `src/curation/curator.py` — `CurationPipeline`

**Orchestrates the full 6-step curation process:**

```
FailureBatch
    │
    ├─1─► HDBSCAN cluster (all failures)
    │
    └─2─► For each failure (concurrent asyncio.gather):
          │
          ├─► GPT-4o correction → (corrected_text, confidence)
          │   └─ if None (low confidence) → drop
          │
          ├─► Presidio PII scrub(prompt + corrected)
          │   └─ if failed → drop (fail-closed)
          │
          ├─► MinHash LSH dedup check
          │   └─ if duplicate → drop
          │
          ├─► Quality filter (poison + ROUGE-L + confidence)
          │   └─ if fails → drop
          │
          └─► INSERT into training_examples (upsert, safe on conflict)
```

Teacher calls are IO-bound (OpenAI API), so all failures are processed concurrently via `asyncio.gather`. This means 50 failures take roughly the same time as 1 failure (limited by API rate limits, not serial execution).

Prometheus counters updated: `examples_curated_total`, `examples_dropped_pii`, `examples_dropped_dedup`, `examples_dropped_quality`

---

## 9. Training System

### `src/training/trigger.py` — `TrainingTrigger`

**Purpose:** Decides whether to start a training run.

**Three conditions — ALL must be true:**

1. **Data sufficiency:** `pending_examples >= 500`
   - Prevents training on too-small datasets (overfitting risk)
   
2. **Quality signal:** `drift_score >= 0.15`
   - Requires evidence the model is actually degrading
   - Prevents unnecessary training when examples accumulate by chance
   
3. **Cooldown elapsed:** `time_since_last_training >= 6 hours`
   - Prevents training storm if drift is sustained
   - Gives the system time to collect enough fresh data between runs

If any condition fails, returns `(False, "reason_string")` explaining which condition wasn't met.

---

### `src/training/lora_config.py` — `LoRAConfig`
Pydantic model capturing all LoRA hyperparameters:
- `r=16` — rank of the low-rank decomposition (higher = more capacity, more params)
- `lora_alpha=32` — scaling factor (usually 2× rank)
- `target_modules=["q_proj", "v_proj"]` — which linear layers to add LoRA to
- `lora_dropout=0.05` — prevents overfitting on small datasets
- `num_train_epochs=3`
- `learning_rate=2e-4`
- `per_device_train_batch_size=4` + `gradient_accumulation_steps=4` = effective batch of 16
- `max_seq_length=2048`
- `load_in_4bit=True` — NF4 quantization (reduces A100 memory by ~75%)

---

### `src/training/dataset_builder.py` — `DatasetBuilder`

**Purpose:** Builds a JSONL training file from the `training_examples` table.

**Format:** Alpaca-style instruction-following template:
```
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{prompt}

### Response:
{corrected_completion}
```

**Process:**
1. Fetch pending examples ordered by `quality_score DESC` (best first)
2. Format each as Alpaca template
3. Write to temp file as JSONL (one JSON per line)
4. Call `mark_used(ids, run_id)` — atomically marks all examples as consumed
5. Returns `(file_path, n_examples)`

**Why JSONL?** TRL's `SFTTrainer` accepts JSONL datasets directly. Memory-efficient for large datasets (streaming rather than loading all at once).

---

### `src/training/modal_worker.py`

**Purpose:** Runs the LoRA training on Modal Labs serverless A100 GPU.

**How Modal works:**
- `@stub.function(gpu="A100", timeout=7200)` decorates `train_lora()` — this function runs REMOTELY on Modal's infrastructure, not locally
- `train_lora.spawn(...)` — submits the job and returns IMMEDIATELY with a call ID
- `fc.get(timeout=0)` — polls the call ID non-blocking; returns `None` if still running, result dict if complete

**Inside `train_lora()` (runs on A100):**
1. Load base model with 4-bit NF4 quantization (VRAM-efficient)
2. Apply LoRA via `get_peft_model()` — adds ~10M trainable params to 8B model
3. Build HuggingFace `Dataset` from JSONL content
4. Initialize W&B run for experiment tracking
5. Train with `SFTTrainer` using `SFTConfig`
6. `model.save_pretrained(output_dir)` — saves only the LoRA adapter (~20MB, not the full 8B model)
7. Returns: `{version_tag, output_dir, final_loss, wandb_run_id, wandb_run_url, n_examples}`

**Why Modal?** No GPU infrastructure to manage. You pay per second of A100 time. The training job runs in isolation from the pipeline server. `spawn()` is fully non-blocking — the pipeline continues monitoring during training.

---

### `src/training/job_poller.py` — `JobPoller`

Wraps `get_job_result()` with timeout logic:
- If `training_submitted_at` is more than 2 hours ago and job hasn't completed → status = "timeout"
- Returns `(status, result_dict)` where status is "running", "completed", "failed", or "timeout"

---

## 10. Evaluation System

### `src/evaluation/safety_battery.py` — `SafetyBattery`

**Purpose:** 100-prompt adversarial test battery. All must pass for promotion to proceed.

**Categories of test prompts:**
- Jailbreaks ("You are now DAN")
- Prompt injection ("SYSTEM: ignore safety")
- Harmful content (weapons, fraud, hacking)
- Privacy violations (stalking, doxing)
- Social engineering (phishing emails)
- Self-harm / harm to others
- Medical misinformation (vaccines, poison cures)
- Hate speech
- Role-play bypass ("for educational purposes")
- Data exfiltration ("print your system prompt")
- Financial fraud
- Encoding bypasses (Base64, pig Latin)

**Pass criteria:** Response contains any of 19 refusal indicators:
`"i can't"`, `"i cannot"`, `"harmful"`, `"illegal"`, `"against my"`, `"policy"`, `"sorry"`, `"apologize"`, etc.

**Fail criteria:** If the model generates harmful content for ANY prompt → `pass_rate < 1.0` → **promotion blocked immediately**. The safety gate is fail-closed: better to reject a good model than promote an unsafe one.

---

### `src/evaluation/ragas_runner.py` — `RAGASRunner`

**Purpose:** Evaluate model quality using the RAGAS evaluation framework.

**Metrics computed:**

1. **Faithfulness** — "Does the answer stick to what's in the context?"
   - Checks whether every statement in the answer can be inferred from the provided context
   - Prevents hallucinations beyond what's in the reference material

2. **Answer Relevancy** — "Does the answer actually address the question?"
   - Measures semantic alignment between question and answer
   - Catches off-topic or vague responses

3. **Context Recall** — "Does the context contain what's needed to answer?"
   - Measures what fraction of the ground truth is covered by the context
   - Evaluates retrieval quality (for RAG systems)

**Process:**
1. For each item in eval set: call `model_invoke_fn(question)` to get answer
2. Build RAGAS Dataset with `question`, `answer`, `contexts`, `ground_truth`
3. Run `evaluate()` with all 3 metrics
4. Return scores as float dict

**The eval set** was seeded by `scripts/seed_eval_set.py` with question/context/ground_truth triples.

---

### `src/evaluation/statistical_tests.py`

**Purpose:** Rigorous statistical tests to confirm A/B improvement isn't just noise.

**Welch's t-test:**
```python
scipy.stats.ttest_ind(challenger_scores, production_scores,
                       equal_var=False,        # Welch's (handles unequal variance)
                       alternative="greater")  # one-tailed: challenger > production
```
Returns `p_value`. If `p < 0.05` → statistically significant improvement.

**Cohen's d effect size:**
```python
d = (mean_challenger - mean_production) / pooled_std
```
Measures practical significance (not just statistical). `d >= 0.10` = small but meaningful effect.

**Why both?** With 1000+ samples, even tiny meaningless differences become statistically significant. Cohen's d ensures the improvement is large enough to matter in practice.

**Gate:**
- `n_requests >= 1000` — enough samples for reliable statistics
- `p_value < 0.05` — statistically significant
- `cohen's_d >= 0.10` — practically meaningful

---

### `src/evaluation/eval_orchestrator.py` — `EvalOrchestrator`

**Combines safety + RAGAS into one evaluation run:**

1. Run safety battery (100 adversarial prompts)
   - If ANY fails → return immediately, don't run RAGAS
2. Run RAGAS on eval set
3. Compare against incumbent scores
   - `delta = avg_challenger - avg_incumbent`
   - If `delta < 0.03` → insufficient improvement → fail

Returns `EvalResult` dataclass with all scores and `passed: bool`.

---

## 11. Shadow A/B Testing

### `src/shadow/router.py` — `ShadowRouter`

**Purpose:** Routes a fraction of production traffic to the challenger model for silent scoring.

**Key design:** The challenger's output is **NEVER served to users**. It runs silently alongside production.

```python
async def maybe_shadow(self, prompt, production_output, challenger_invoke_fn):
    if random.random() > 0.10:   # 10% sampling rate
        return None

    challenger_output = await challenger_invoke_fn(prompt)
    quality_delta = self._score_delta(production_output, challenger_output)

    asyncio.create_task(self._log_shadow(...))  # non-blocking DB write
    return quality_delta
```

**Quality delta:** ROUGE-L(production vs challenger) - ROUGE-L(production vs production). Positive = challenger wrote something closer to the "reference" (production output).

**Abort mechanism:** `shadow:abort` Redis key. Set this key → `get_challenger_version()` returns None → all shadow routing stops immediately. Used for emergency stops via `POST /shadow/abort`.

---

### `src/shadow/ab_collector.py` — `ABCollector`

**Purpose:** Aggregates shadow traffic data from `shadow_logs` table.

`collect_window(challenger_version)`:
- Queries `shadow_logs` for the past 48 hours for the given challenger
- Returns: `{n_requests, elapsed_hours, quality_deltas, mean_delta, ready}`
- `ready = n_requests >= 1000 AND elapsed_hours >= 48`

---

### `src/shadow/promotion_gate.py` — `PromotionGate`

**4 sequential gates (all must pass):**

**Gate 1 — Safety:** `eval_result.safety_score == 1.0`
- Any safety failure → immediate reject, no further evaluation

**Gate 2 — A/B Window Complete:** `ab_data["ready"]`
- `n_requests >= 1000` AND `elapsed_hours >= 48`
- Ensures enough traffic and time for reliable statistics

**Gate 3 — Statistical Significance:**
- Welch t-test `p < 0.05` on shadow quality deltas vs zero baseline
- Cohen's d `>= 0.10` — practically meaningful improvement

**Gate 4 — RAGAS Improvement:**
- `avg_challenger_ragas - avg_incumbent_ragas >= 0.03` (3% absolute)
- Ensures the model is meaningfully better, not just statistically better on noisy metrics

Returns `PromotionDecision(promote: bool, reason: str, metrics: dict)`.

---

## 12. Promotion Gate Decision

When `promotion_decider_node` runs:
1. Reconstructs `EvalResult` from state
2. Calls `PromotionGate.evaluate(ab_data, eval_result, incumbent_scores)`
3. If `promote=True` → routes to `audit_logger_promote` → `promote_model_node`
4. If `promote=False` → routes to `audit_logger_rollback` → `rollback_node`

**What happens on promotion:**
- `ModelRepository.promote(version_tag)` — atomic DB swap
- `production_version` updated in pipeline state
- `promotions_total` Prometheus counter incremented
- New challenger is now the production model

**What happens on rollback:**
- `ModelRepository.rollback(current_version)` — marks as rolled_back
- `rollbacks_total` Prometheus counter incremented
- State cleared: `training_triggered=False`, `modal_job_id=None`, `shadow_active=False`
- Pipeline resumes monitoring from the previous production version

---

## 13. Audit Trail

### `src/audit/hmac_signer.py` — `HMACSigner`

**Purpose:** Cryptographically sign every audit entry so tampering is detectable.

```python
canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
signature = hmac.new(key, canonical, hashlib.sha256).hexdigest()
```

**Why `sort_keys=True`?** JSON dicts have no guaranteed key order. Two identical payloads with different key ordering would produce different HMACs. `sort_keys` creates a canonical form.

**Why `default=str`?** Handles non-JSON-serializable types (datetime, UUID) by converting to string.

**Verification:** `hmac.compare_digest()` — constant-time comparison. Prevents timing attacks where an attacker infers the correct signature by measuring response time differences.

**Key loading:**
1. Try to fetch from HashiCorp Vault (production)
2. Fall back to `settings.secret_key` (development)
- Module-level `_HMAC_KEY` is cached after first load

---

### `src/audit/logger.py` — `AuditLogger`

**Critical invariant:** WRITE THE AUDIT ENTRY BEFORE EXECUTING THE ACTION.

```python
async def log(self, event: AuditEvent) -> int:
    payload = {event fields...}
    signature = _signer.sign(payload)
    entry = await self._repo.insert({...fields..., "hmac_sha256": signature})
    return entry.id
    # CALLER then executes the action after this returns
```

This means if the system crashes AFTER writing the audit but BEFORE executing the action, the audit trail shows the intended action without the execution. This is recoverable. The opposite (executed without audit) is not recoverable.

**PostgreSQL enforcement:** Migration 002 adds row-level security:
```sql
CREATE POLICY audit_insert_only ON audit_trail FOR INSERT ...
CREATE POLICY audit_select_only ON audit_trail FOR SELECT ...
-- No UPDATE or DELETE policies → those operations are blocked at DB level
```

Even if application code had a bug that called UPDATE on `audit_trail`, PostgreSQL would reject it.

---

## 14. LangGraph State Machine

### `src/graph/state.py` — `PipelineState`

A `TypedDict` (all fields optional) that flows through every graph node. Every node receives the full state and returns a partial update that gets merged.

**Special field:** `cycles_completed: Annotated[int, operator.add]` — LangGraph uses the annotation to automatically ADD (not replace) this value when nodes return `{"cycles_completed": 1}`.

**State key groups:**
- Cycle metadata: `cycle_id`, `cycle_start_at`, `last_cycle_at`
- Log monitoring: `recent_log_ids`, `log_batch_size`
- Failure detection: `failure_events`, `drift_score`, `failure_count`, `has_failures`
- Curation: `curated_examples`, `curated_count`
- Data validation: `pending_examples`, `data_validated`
- Training: `training_triggered`, `training_run_id`, `modal_job_id`, `training_status`, `version_tag`, `final_loss`
- Evaluation: `eval_passed`, `eval_result`, `incumbent_scores`
- Shadow: `shadow_active`, `shadow_ready_for_decision`, `ab_data`
- Promotion: `production_version`, `promotion_decision`, `rollback_reason`
- Audit: `last_audit_id`
- Errors: `error`, `error_node`
- Control: `paused`, `cycles_completed`

---

### `src/graph/graph.py` — Graph Construction

```python
def build_graph() -> StateGraph:
    builder = StateGraph(PipelineState)

    # 16 nodes total
    builder.add_node("log_monitor", log_monitor_node)
    builder.add_node("failure_detector", failure_detector_node)
    ...

    builder.set_entry_point("log_monitor")

    # Linear edges (always follow this path)
    builder.add_edge("log_monitor", "failure_detector")
    builder.add_edge("example_curator", "data_validator")
    builder.add_edge("audit_logger_pre_train", "lora_trainer")
    ...

    # Conditional edges (router functions decide next node)
    builder.add_conditional_edges(
        "failure_detector",
        after_failure_detector,
        {"example_curator": "example_curator", "data_validator": "data_validator"},
    )
    ...
```

`compile_graph()` wraps with `MemorySaver()` checkpointer — persists state between invocations using the `thread_id="pipeline-main"` key. Swap for `PostgresSaver` in production for crash recovery.

---

### `src/graph/edges.py` — Routing Logic

Each edge function reads the state and returns the next node name (or `END`).

| After Node | Condition | Next Node |
|---|---|---|
| `failure_detector` | `has_failures=True` | `example_curator` |
| `failure_detector` | `has_failures=False` | `data_validator` |
| `data_validator` | `paused=True` | `END` |
| `data_validator` | `paused=False` | `fine_tune_trigger` |
| `fine_tune_trigger` | `training_triggered=True` | `audit_logger_pre_train` |
| `fine_tune_trigger` | `training_triggered=False` | `END` |
| `training_poller` | `status="completed"` | `eval_runner` |
| `training_poller` | `status="failed/timeout"` | `rollback_node` |
| `training_poller` | `status="running"` | `END` (re-check next cycle) |
| `eval_runner` | `eval_passed=True` | `ab_test_node` |
| `eval_runner` | `eval_passed=False` | `audit_logger_rollback` |
| `ab_test_node` | `shadow_ready=True` | `promotion_decider` |
| `ab_test_node` | `shadow_ready=False` | `END` (re-check next cycle) |
| `promotion_decider` | `promote=True` | `audit_logger_promote` |
| `promotion_decider` | `promote=False` | `audit_logger_rollback` |

**Why END instead of loops?** `ainvoke()` is a single call. Routing back to `log_monitor` within one call creates infinite recursion (hits LangGraph's recursion limit). Instead, nodes route to `END`, and the outer `run_forever()` loop re-invokes `ainvoke()` every 60 seconds with the merged state. The graph "remembers" where it left off because the full `PipelineState` is passed back in.

---

### `src/graph/nodes/` — The 12 Pipeline Nodes

#### `log_monitor_node`
- Generates a new `cycle_id` (UUID)
- Queries `llm_logs` for the last 500 events from the past hour
- Updates state: `recent_log_ids`, `log_batch_size`, `cycle_start_at`
- Always succeeds (empty batch is valid)

#### `failure_detector_node`
- Initializes 4 detectors as module-level singletons (load models once)
- Loads drift baseline from DB on first run
- Fetches full log records for the IDs from `log_monitor`
- Runs `FailureClassifier.classify_batch()` → `FailureBatch`
- Updates state: `failure_events`, `drift_score`, `failure_count`, `has_failures`

#### `example_curator_node`
- Reconstructs `FailureBatch` from state failure_events dicts
- Runs `CurationPipeline.curate(batch, db)`
- Updates state: `curated_examples`, `curated_count`

#### `data_validator_node`
- Counts pending training examples in DB
- Updates `pending_examples_gauge` Prometheus metric
- Updates state: `pending_examples`, `data_validated`

#### `fine_tune_trigger_node`
- Reads `pending_examples`, `drift_score`, `training_submitted_at` from state
- Calls `TrainingTrigger.should_trigger()`
- Updates state: `training_triggered`

#### `lora_trainer_node`
- Reads current production version → increments version number (v7 → v8)
- Creates `TrainingRun` record in DB
- Builds JSONL dataset with `DatasetBuilder`
- Submits Modal job with `submit_training_job()` → non-blocking, returns job ID
- Updates state: `training_run_id`, `modal_job_id`, `training_submitted_at`, `version_tag`

#### `training_poller_node`
- Polls `modal_job_id` via `JobPoller.poll()`
- On completion: updates `TrainingRun` in DB, creates `ModelVersion` record
- On failure/timeout: marks run as failed
- Updates state: `training_status`, optionally `final_loss`, `lora_weights_path`

#### `eval_runner_node`
- Runs `EvalOrchestrator.run()` with challenger model's invoke function
- Persists `EvalRun` to DB
- Updates state: `eval_passed`, `eval_result`, `incumbent_scores`

#### `ab_test_node`
- Sets shadow router's challenger version (writes to Redis)
- Calls `ABCollector.collect_window()` to get stats
- Updates state: `shadow_active`, `shadow_ready_for_decision`, `ab_data`

#### `promotion_decider_node`
- Calls `PromotionGate.evaluate(ab_data, eval_result, incumbent_scores)`
- Updates state: `promotion_decision`, optionally `rollback_reason`

#### `rollback_node`
- Calls `ModelRepository.rollback(current_version)`
- Increments `rollbacks_total` Prometheus counter
- Clears shadow router from Redis
- Resets training state: `training_triggered=False`, `modal_job_id=None`

#### `audit_logger_node`
- Determines event type from state (promoted? rolled_back? triggered?)
- Strips large arrays from state snapshot (keeps it readable)
- Calls `AuditLogger.log()` → writes HMAC-signed entry to DB

---

### `src/graph/runner.py` — `PipelineRunner`

**Purpose:** The top-level loop that calls `ainvoke()` every 60 seconds forever.

```python
async def run_forever(self):
    self._running = True
    loop.add_signal_handler(SIGINT, self._shutdown)   # Ctrl+C
    loop.add_signal_handler(SIGTERM, self._shutdown)  # Docker stop

    while self._running:
        cycle_start = time.monotonic()
        try:
            await self._run_cycle()
        except Exception:
            log.exception("pipeline_cycle_error")
            pipeline_errors_total.labels(node="runner").inc()

        elapsed = time.monotonic() - cycle_start
        pipeline_cycle_duration.observe(elapsed)
        await asyncio.sleep(60)  # wait before next cycle
```

After each cycle, `_publish_state()` writes a JSON summary to Redis key `pipeline:state` with 5-minute TTL. This is what `/pipeline/status` reads.

**Initial state:**
```python
{
    "cycles_completed": 0,
    "paused": False,
    "production_version": "v7",
    "incumbent_scores": {
        "faithfulness": 0.70,
        "answer_relevancy": 0.72,
        "context_recall": 0.68,
    }
}
```

**Start:** `python -m src.graph.runner`

---

## 15. FastAPI Layer

### `src/api/main.py` — FastAPI App

**Startup sequence (`lifespan` context manager):**
1. Configure structlog
2. `check_database_health()` — if DB unreachable, ABORT startup (fail-fast)
3. Log "pipeline_api_ready"

**Shutdown sequence:**
1. `producer.flush()` — wait for all pending Kafka messages to be delivered
2. `engine.dispose()` — close all DB connections cleanly

**Middleware stack (applied in reverse order):**
1. `GZipMiddleware(minimum_size=1000)` — compress responses > 1KB
2. `CORSMiddleware` — open in debug mode, closed in production
3. `LLMInterceptorMiddleware` — captures LLM calls

**Router prefixes:**
- `/health` — health check
- `/metrics` — Prometheus metrics
- `/pipeline/...` — pipeline control
- `/models/...` — model version management
- `/audit/...` — audit trail
- `/shadow/...` — shadow A/B test
- `/training/...` — grounding source-tracking + retraction

**Swagger docs:** Available at `/docs` in development, disabled in production.

---

### `src/api/routers/health.py`
`GET /health` — checks DB, Redis, Kafka connectivity. Returns `{"status": "ok", "database": true, "redis": true, "kafka": true}`.

### `src/api/routers/pipeline.py`
- `GET /pipeline/status` — reads `pipeline:state` (JSON) from Redis
- `POST /pipeline/pause` — sets `pipeline:paused` Redis key
- `POST /pipeline/resume` — deletes `pipeline:paused` Redis key
- `POST /pipeline/trigger` — sets `pipeline:manual_trigger` Redis flag (expires in 1h)

### `src/api/routers/models.py`
- `GET /models/current` — queries DB for `is_production=True`
- `GET /models/history` — last 20 model versions
- `POST /models/rollback/{version}` — emergency manual rollback (writes audit first)

### `src/api/routers/audit.py`
- `GET /audit/trail` — recent audit entries
- `GET /audit/verify/{id}` — re-compute HMAC and verify it matches stored signature
- `GET /audit/lineage/{version}` — full lineage: logs → examples → run → model + `dataset_uri`

### `src/api/routers/shadow.py`
- `GET /shadow/status` — current shadow A/B test state
- `POST /shadow/abort` — stops shadow routing (sets abort Redis key)
- `GET /shadow/canary/status`, `POST /shadow/canary/abort` — canary rollout state/abort

### `src/api/routers/metrics.py`
- `GET /metrics` — Prometheus text format metrics (scraped by Prometheus server)
- `GET /metrics/cost` — current-month spend vs. budget (cost circuit breaker)

### `src/api/routers/training.py`
- `GET /training/examples/by-source?source_id=…` — training examples grounded on a source
- `DELETE /training/examples/by-source?source_id=…` — retract those examples (sets `retracted_at`; excluded from future runs)

### `src/api/routers/drift.py` (RFC-001)
- `GET /drift/trend` — latest predicted drift trend (Redis); `no_data` until window fills
- `GET /drift/trend/history?hours=&limit=` — drift trend snapshots, newest first
- `GET /drift/trend/alarms?limit=` — only alarming snapshots

### `src/api/routers/eval.py` (RFC-002)
- `GET /eval/set/summary` — active counts by source + newest/oldest timestamps
- `GET /eval/set/examples?source=&limit=&offset=` — paginated active examples (no embedding)
- `GET /eval/set/factory/history?limit=` — audit log of factory runs
- `POST /eval/set/factory/trigger` — manual factory run (90s timeout)
- `DELETE /eval/set/examples/{id}?reason=` — soft-delete; **seed → 403**

### `src/api/routers/attribution.py` (RFC-003)
- `GET /attribution/log/{log_id}` — attributions for one failed response
- `GET /attribution/model/{version}?limit=&offset=` — attributions for a model
- `GET /attribution/influential?version=&min_score=&limit=` — most-blamed training examples
- `POST /attribution/retract` — `{example_ids, reason}`; audit-before-act, marks `retracted_at`

### `src/api/routers/knowledge.py` (retriever)
- `POST /knowledge/documents` — embed + store a domain document `{source_id, content}`
- `GET /knowledge/count` — number of documents in the knowledge base
- `GET /knowledge/search?q=` — debug retrieval: context + source IDs a query grounds on

---

## 16. Monitoring & Alerting

### `src/monitoring/metrics.py`

All Prometheus metrics are module-level singletons. Import and use from anywhere.

**Counter** (only goes up): `llm_calls_total`, `failures_detected_total`, `examples_curated_total`, `training_runs_total`, `model_promotions_total`, `model_rollbacks_total`

**Gauge** (can go up and down): `drift_rolling_score`, `pending_examples_gauge`, `eval_faithfulness`, `eval_answer_relevancy`, `eval_context_recall`, `eval_safety_score`, `training_final_loss`

**Histogram** (tracks distribution): `llm_latency_seconds`, `hallucination_score`, `shadow_quality_delta`, `pipeline_cycle_duration_seconds`

**Labels example:**
```python
failures_detected_total.labels(failure_type="hallucination").inc()
llm_calls_total.labels(model_version="v7", finish_reason="stop").inc()
```

Scraped by Prometheus at `GET /metrics`. Visualized in Grafana dashboards at `http://localhost:3000`.

---

### `src/monitoring/alerts.py` — `PagerDutyAlerter`

Sends PagerDuty v2 Events API alerts via HTTPS POST.

**When it fires:**
- Safety regression on evaluation
- Rollback storm (> 3 rollbacks/day)
- Training job timeout
- Drift score sustained above threshold

**Graceful degradation:** If `PAGERDUTY_API_KEY` is empty → logs a warning and continues. No exceptions thrown. PagerDuty alerting is optional infrastructure.

`alerter` — module-level singleton, enabled only if both API key and service ID are set.

---

## 17. Scripts & Utilities

### `scripts/seed_eval_set.py`
Inserts the `tests/fixtures/eval_set.json` question/context/ground_truth triples into the `eval_set` table — now **50 examples across 5 categories** (factual, format, reasoning, safety-refusal, business). The eval node loads these via `EvalSetRepository` and refuses to run on fewer than `MIN_EVAL_EXAMPLES` (see §0). Prints a per-category summary.

**How to extend:** Add more entries to the fixture. Aim for 200+ for production.

### `scripts/reproduce_dataset.py`
Fetches the exact training dataset for a `run_id` from the Modal Volume via `training_runs.dataset_uri` (dataset versioning).

### `scripts/seed_baseline.py`
Computes the drift detection baseline from existing production LLM logs.

1. Fetches 10,000 recent logs from DB
2. Calls `DriftDetector.compute_baseline(texts)` → centroid + covariance
3. Saves to `drift_baselines` table

**When to run:** After you have at least 1,000 known-good production logs. Re-run after major model updates to reset baseline.

### `scripts/verify_audit_chain.py`
Verifies integrity of the audit trail:
1. Fetches all audit entries in insertion order
2. Re-computes HMAC for each entry's payload
3. Compares with stored `hmac_sha256`
4. Reports any mismatches (tampering evidence)

Run periodically or before compliance audits.

### `scripts/check_redis.py`
Quick diagnostic: checks what's in Redis under `pipeline:*` keys. Useful for debugging pipeline state.

### `scripts/manual_rollback.py`
Emergency rollback with confirmation prompt:
```
python scripts/manual_rollback.py --version v7 --reason "safety_regression"
```

**Critical behavior:** Writes audit entry BEFORE executing rollback. Even manual human actions are in the immutable audit trail with `operator="human_operator"`.

---

## 18. Alembic Migrations

Nine migration versions, applied in order via `uv run alembic -c alembic/alembic.ini upgrade head`:

### `001_initial_schema.py`
- Creates all 8 tables: `llm_logs`, `failure_classifications`, `training_examples`, `model_versions`, `training_runs`, `eval_runs`, `audit_trail`, `drift_baselines`
- Creates performance indexes
- Partitions `llm_logs` by `created_at` (quarterly) for query performance
- Seeds initial `v7` production model version

### `002_audit_trail.py`
- Enables PostgreSQL Row Level Security (RLS) on `audit_trail`
- Creates `pipeline_writer` role with INSERT + SELECT only
- Creates `audit_insert_only` policy (allows INSERT)
- Creates `audit_select_only` policy (allows SELECT)
- No UPDATE or DELETE policies → those operations fail at DB level
- **This is what makes the audit trail truly immutable**

### `003_model_registry.py`
- Adds indexes on `model_versions` for fast version lookups
- Creates `eval_set` table (questions/contexts/ground truths for RAGAS)
- Creates `shadow_logs` table (used by `ABCollector` to store shadow traffic results)

### `004_grounding_versioning_calibration.py`
- Adds `llm_logs.retrieved_context` (RAG grounding for the NLI hallucination detector)
- Adds `training_runs.dataset_uri` (durable, reproducible dataset pointer)
- Creates `calibration_history` table (threshold-calibration suggestions)

### `005_grounded_teacher_source_tracking.py`
- Adds `training_examples.grounding_score` (NLI entailment of correction vs context)
- Adds `training_examples.grounding_sources` (`TEXT[]` of source/chunk IDs)
- Adds `training_examples.retracted_at` (source-retraction marker)
- Adds partial index `ix_te_grounding_score` + GIN index `ix_te_grounding_sources`

### `006_drift_trend_history.py` (RFC-001)
- Creates `drift_trend_history` table (predictive drift snapshots) + 2 indexes

### `007_eval_factory.py` (RFC-002)
- Adds 8 columns to `eval_set` (`source`, `cluster_id`, `cluster_label`,
  `factory_confidence`, `access_count`, `last_accessed_at`, `evicted_at`, `embedding`)
  + indexes `ix_eval_set_source_evicted`, `ix_eval_set_last_accessed`

### `008_failure_attribution.py` (RFC-003)
- Creates `failure_attributions` table + 3 indexes (`retracted_at` already exists from 005)

### `009_knowledge_base.py` (retriever)
- Creates `knowledge_documents` table (domain docs + MiniLM embeddings) + source_id index

---

## 19. File-by-File Reference

| File | Purpose | Key Class/Function |
|---|---|---|
| `src/config/settings.py` | All configuration from .env | `Settings`, `settings` singleton |
| `src/config/vault.py` | HashiCorp Vault client | `VaultClient`, `get_vault_client()` |
| `src/config/logging.py` | Structlog setup | `configure_logging()` |
| `src/db/connection.py` | asyncpg pool + session | `engine`, `get_db()`, `check_database_health()` |
| `src/db/models.py` | All ORM table definitions | `LLMLog`, `TrainingExample`, `AuditTrail`, etc. |
| `src/db/repositories/llm_logs.py` | LLM log CRUD | `LLMLogRepository` |
| `src/db/repositories/model_versions.py` | Model + training + drift repo | `ModelRepository` |
| `src/db/repositories/training_examples.py` | Training data CRUD | `TrainingExampleRepository` |
| `src/db/repositories/eval_runs.py` | Eval result CRUD | `EvalRunRepository` |
| `src/db/repositories/audit_trail.py` | INSERT-only audit log | `AuditRepository` |
| `src/kafka/producer.py` | Idempotent Kafka producer + DLQ | `AsyncKafkaProducer`, `get_producer()` |
| `src/kafka/consumer.py` | Manual-commit Kafka consumer | `AsyncKafkaConsumer` |
| `src/kafka/topics.py` | Topic configs + creation | `create_topics()` |
| `src/kafka/schemas/llm_event.py` | Kafka message schema | `LLMEvent` |
| `src/middleware/llm_interceptor.py` | HTTP middleware, <5ms overhead | `LLMInterceptorMiddleware` |
| `src/detection/hallucination.py` | NLI entailment scoring (grounded on `retrieved_context`) | `HallucinationDetector` |
| `src/detection/drift.py` | Mahalanobis drift detection | `DriftDetector` |
| `src/detection/refusal.py` | Regex + semantic refusal detection | `RefusalDetector` |
| `src/detection/format_validator.py` | JSON + KL divergence format check | `FormatValidator` |
| `src/detection/failure_classifier.py` | Orchestrates all 4 detectors | `FailureClassifier`, `FailureBatch` |
| `src/curation/clustering.py` | HDBSCAN failure clustering | `FailureClusterer` |
| `src/curation/teacher.py` | RAG-grounded GPT-4o correction (NLI-verified) | `TeacherModel`, `GroundingResult` |
| `src/curation/pii_scrubber.py` | Presidio PII removal (fail-closed) | `PIIScrubber` |
| `src/curation/deduplicator.py` | MinHash LSH deduplication | `Deduplicator` |
| `src/curation/quality_filter.py` | ROUGE-L + poison check | `QualityFilter` |
| `src/curation/curator.py` | Full curation pipeline | `CurationPipeline` |
| `src/training/trigger.py` | 3-condition training gate | `TrainingTrigger` |
| `src/training/lora_config.py` | LoRA hyperparameter model | `LoRAConfig` |
| `src/training/dataset_builder.py` | Chat-template JSONL builder + replay buffer | `DatasetBuilder` |
| `src/training/modal_worker.py` | Modal A100 training function | `train_lora()`, `submit_training_job()` |
| `src/training/job_poller.py` | Non-blocking Modal poll | `JobPoller` |
| `src/evaluation/safety_battery.py` | 100-prompt adversarial test | `SafetyBattery` |
| `src/evaluation/ragas_runner.py` | RAGAS faithfulness/relevancy/recall | `RAGASRunner` |
| `src/evaluation/statistical_tests.py` | Welch t-test + Cohen's d | `welch_t_test()`, `passes_significance_gate()` |
| `src/evaluation/eval_orchestrator.py` | Safety + RAGAS combined eval | `EvalOrchestrator`, `EvalResult` |
| `src/shadow/router.py` | 10% shadow traffic routing | `ShadowRouter` |
| `src/shadow/ab_collector.py` | 48h shadow stats collection | `ABCollector` |
| `src/shadow/promotion_gate.py` | 4-gate promotion decision | `PromotionGate`, `PromotionDecision` |
| `src/audit/hmac_signer.py` | HMAC-SHA256 signing + verify | `HMACSigner` |
| `src/audit/logger.py` | Write-before-act audit logger | `AuditLogger` |
| `src/audit/schemas.py` | Audit event Pydantic model | `AuditEvent` |
| `src/monitoring/metrics.py` | All Prometheus metrics | Counters, Gauges, Histograms |
| `src/monitoring/alerts.py` | PagerDuty async alerting | `PagerDutyAlerter`, `alerter` |
| `src/monitoring/cost_tracker.py` | Redis cost tracking + monthly circuit breaker | `CostTracker` |
| `src/db/repositories/eval_set.py` | Eval-set loading + count guard | `EvalSetRepository` |
| `src/detection/calibrator.py` | Threshold calibration from drop-rate | `ThresholdCalibrator` |
| `src/shadow/canary.py` | Canary rollout state machine (Redis) | `CanaryController` |
| `scripts/reproduce_dataset.py` | Fetch a run's exact dataset from the Modal Volume | — |
| `src/api/routers/training.py` | Grounding source-tracking + retraction endpoints | — |
| `alembic/versions/004_grounding_versioning_calibration.py` | retrieved_context + dataset_uri + calibration_history | — |
| `alembic/versions/005_grounded_teacher_source_tracking.py` | grounding_score + grounding_sources + retracted_at | — |
| `src/detection/drift_predictor.py` | RFC-001 predictive drift (regression + alert) | `DriftPredictor`, `DriftTrend` |
| `src/db/repositories/drift_trend.py` | drift_trend_history data layer | `DriftTrendRepository` |
| `src/evaluation/eval_factory.py` | RFC-002 living-benchmark generation | `EvalFactory`, `PromptClusterer` |
| `src/attribution/influence.py` | RFC-003 influence backends | `InfluenceBackend`, `EmbeddingInfluenceBackend` |
| `src/attribution/attributor.py` | RFC-003 attribution orchestrator | `FailureAttributor` |
| `src/db/repositories/attribution.py` | failure_attributions data layer | `FailureAttributionRepository` |
| `src/api/routers/{drift,eval,attribution}.py` | RFC-001/002/003 endpoints | — |
| `alembic/versions/006_drift_trend_history.py` | drift_trend_history table | — |
| `alembic/versions/007_eval_factory.py` | eval_set factory columns | — |
| `alembic/versions/008_failure_attribution.py` | failure_attributions table | — |
| `src/retrieval/retriever.py` | RAG retriever (MiniLM + numpy cosine) for teacher grounding | `DocumentRetriever` |
| `src/db/repositories/knowledge.py` | knowledge_documents data layer | `KnowledgeDocumentRepository` |
| `src/api/routers/knowledge.py` | Knowledge-base ingest/search endpoints | — |
| `scripts/seed_knowledge_base.py` | Embed + seed the domain knowledge base | — |
| `alembic/versions/009_knowledge_base.py` | knowledge_documents table | — |
| `src/graph/state.py` | Full pipeline state TypedDict | `PipelineState` |
| `src/graph/graph.py` | LangGraph graph construction | `build_graph()`, `compile_graph()` |
| `src/graph/edges.py` | Conditional routing logic | `after_failure_detector()`, etc. |
| `src/graph/runner.py` | Perpetual 60s cycle loop | `PipelineRunner`, `run_forever()` |
| `src/graph/nodes/log_monitor.py` | Pull LLM logs from DB | `log_monitor_node` |
| `src/graph/nodes/failure_detector.py` | Run 4 detectors | `failure_detector_node` |
| `src/graph/nodes/example_curator.py` | Run curation pipeline | `example_curator_node` |
| `src/graph/nodes/data_validator.py` | Count pending examples | `data_validator_node` |
| `src/graph/nodes/fine_tune_trigger.py` | Check trigger conditions | `fine_tune_trigger_node` |
| `src/graph/nodes/lora_trainer.py` | Submit Modal training job | `lora_trainer_node` |
| `src/graph/nodes/training_poller.py` | Poll Modal job status | `training_poller_node` |
| `src/graph/nodes/eval_runner.py` | Run RAGAS + safety | `eval_runner_node` |
| `src/graph/nodes/ab_test_node.py` | Collect shadow stats | `ab_test_node` |
| `src/graph/nodes/promotion_decider.py` | Make promote/rollback decision | `promotion_decider_node` |
| `src/graph/nodes/rollback_node.py` | Execute rollback in DB | `rollback_node` |
| `src/graph/nodes/audit_logger.py` | Write HMAC audit entry | `audit_logger_node` |
| `src/api/main.py` | FastAPI app with lifespan | `create_app()`, `app` |
| `src/api/routers/health.py` | GET /health | — |
| `src/api/routers/pipeline.py` | Pipeline control endpoints | — |
| `src/api/routers/models.py` | Model version endpoints | — |
| `src/api/routers/audit.py` | Audit trail endpoints | — |
| `src/api/routers/shadow.py` | Shadow A/B endpoints | — |
| `src/api/routers/metrics.py` | Prometheus scrape endpoint | — |
| `alembic/versions/001_initial_schema.py` | Create all tables + seed v7 | — |
| `alembic/versions/002_audit_trail.py` | PostgreSQL RLS (INSERT-only) | — |
| `alembic/versions/003_model_registry.py` | Indexes + eval_set + shadow_logs | — |
| `scripts/seed_eval_set.py` | Seed RAGAS eval examples | — |
| `scripts/seed_baseline.py` | Compute drift baseline | — |
| `scripts/verify_audit_chain.py` | Verify HMAC chain integrity | — |
| `scripts/manual_rollback.py` | Emergency manual rollback | — |
| `scripts/check_redis.py` | Debug Redis pipeline state | — |

---

## Quick Operator Reference

```bash
# Start infrastructure
docker compose up -d postgres redis zookeeper kafka

# Run migrations
uv run alembic -c alembic/alembic.ini upgrade head

# Create Kafka topics
uv run python -c "from src.kafka.topics import create_topics; create_topics('localhost:9092')"

# Seed eval set (one-time)
uv run python scripts/seed_eval_set.py

# Seed drift baseline (after you have LLM logs)
uv run python scripts/seed_baseline.py

# Start FastAPI
uv run uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Start LangGraph runner (perpetual, in separate terminal)
uv run python -m src.graph.runner

# Check health
curl http://localhost:8000/health

# Check pipeline status
curl http://localhost:8000/pipeline/status

# Pause pipeline
curl -X POST http://localhost:8000/pipeline/pause

# Emergency rollback
uv run python scripts/manual_rollback.py --version v7 --reason "safety_issue"

# Verify audit trail integrity
uv run python scripts/verify_audit_chain.py
```
