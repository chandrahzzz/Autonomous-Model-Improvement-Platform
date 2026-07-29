# How This Product Functions — Complete Working Reference

> A practical, file-by-file map of the **Continuous Fine-Tuning Pipeline**: what
> the product is, how it works end-to-end, which file is responsible for each
> step, and a full data-flow diagram. For the exhaustive per-feature deep-dive
> see [`prevknowledgehere.md`](prevknowledgehere.md); this file is the "how it
> all fits together" overview.

---

## 1. What This Product Is (Brief)

This is a **fully autonomous LLM continuous fine-tuning pipeline**. You point your
application's LLM traffic at it, and from then on it runs **24/7 with zero human
intervention** to keep your model from silently degrading.

In one sentence: *it watches every LLM call your app makes, automatically notices
when the model starts failing, writes corrected training data with a teacher model,
fine-tunes a new LoRA adapter, proves the new model is actually better through a
held-out eval + a 48-hour live shadow A/B test, and then either promotes it to
production or rolls back — recording every decision in a tamper-evident audit trail.*

**The core problem it solves:** models drift. A model that was great at launch
slowly gets worse as real-world inputs shift away from its training distribution.
Normally a human has to *notice* this, collect failures, label corrections, retrain,
evaluate, and deploy. This product closes that entire loop automatically and safely.

**What makes it "production-grade" and not a toy:**
- **Safe by construction** — a challenger is only promoted after passing 4 hard
  gates (safety, A/B completeness, statistical significance, quality improvement),
  and anything that fails auto-rolls-back.
- **Compliant** — PII is scrubbed *before* any text is sent to a third-party teacher API.
- **Auditable** — every promote/rollback/train decision is HMAC-signed and written
  to an INSERT-only Postgres table before the action executes.
- **Resilient** — it resumes in-flight training jobs and shadow windows after a crash,
  persists detector state, and stops spending when it hits a monthly cost budget.

---

## 2. The 10-Step Lifecycle (and which file owns each step)

The whole product is a **perpetual loop**. One pass through the loop is a "cycle".
Here is what happens each cycle and the file responsible:

| # | What happens | Primary file(s) responsible |
|---|---|---|
| 1 | Every LLM call your app makes is captured (<5ms overhead) and emitted to Kafka | [`src/middleware/llm_interceptor.py`](src/middleware/llm_interceptor.py), [`src/kafka/producer.py`](src/kafka/producer.py) |
| 2 | Events land in Postgres; the loop pulls the last ~500 | [`src/graph/nodes/log_monitor.py`](src/graph/nodes/log_monitor.py), [`src/db/repositories/llm_logs.py`](src/db/repositories/llm_logs.py) |
| 3 | 4 detectors run to find failures (hallucination / drift / refusal / format) | [`src/detection/`](src/detection/), orchestrated by [`failure_classifier.py`](src/detection/failure_classifier.py) |
| 4 | Failures are clustered, PII-scrubbed, corrected by a teacher (GPT-4o), de-duped, quality-filtered, and stored | [`src/curation/`](src/curation/), orchestrated by [`curator.py`](src/curation/curator.py) |
| 5 | When enough clean examples + drift + cooldown align, a training run is triggered | [`src/training/trigger.py`](src/training/trigger.py), [`src/graph/nodes/fine_tune_trigger.py`](src/graph/nodes/fine_tune_trigger.py) |
| 6 | A LoRA adapter is trained on a remote Modal A100 GPU | [`src/training/modal_worker.py`](src/training/modal_worker.py), [`dataset_builder.py`](src/training/dataset_builder.py) |
| 7 | The trained challenger is evaluated: safety battery + RAGAS, on the real model | [`src/evaluation/`](src/evaluation/), [`src/inference/challenger.py`](src/inference/challenger.py) |
| 8 | A 48h shadow A/B test runs the challenger silently on live traffic | [`src/shadow/router.py`](src/shadow/router.py), [`ab_collector.py`](src/shadow/ab_collector.py) |
| 9 | The promotion gate decides promote vs rollback (optionally via a live canary) | [`src/shadow/promotion_gate.py`](src/shadow/promotion_gate.py), [`canary.py`](src/shadow/canary.py) |
| 10 | The decision is HMAC-audited, then the DB swap (promote) or rollback executes | [`src/audit/`](src/audit/), [`src/graph/graph.py`](src/graph/graph.py) `promote_model_node` |

**Who drives the loop?** Two long-running processes:
- [`src/graph/runner.py`](src/graph/runner.py) — the **brain**. It compiles the
  state machine and calls it forever, pacing itself per phase, persisting state,
  refreshing baselines, calibrating thresholds, replaying the DLQ, and enforcing
  the cost budget.
- [`src/api/main.py`](src/api/main.py) — the **front door**. A FastAPI app that
  hosts the interceptor middleware (capturing traffic) and all the
  observability/control endpoints.

---

## 3. Complete Data-Flow Diagram

```
                          YOUR APPLICATION
                                │  HTTP request (X-LLM-Call: 1 header)
                                ▼
        ┌───────────────────────────────────────────────┐
        │  FastAPI  (src/api/main.py)                     │
        │  └─ LLMInterceptorMiddleware  (<5ms overhead)   │   serves user FIRST,
        │     captures prompt/completion/tokens/cost/ver  │   then fire-and-forget
        └───────────────────────┬───────────────────────┘
                                 │ asyncio.create_task → produce()
                                 ▼
                  Kafka topic: llm.production.events
                  (producer.py → consumer.py → DB)            failures → pipeline.dlq
                                 │                                      │
                                 ▼                                      ▼
                   Postgres: llm_logs table                    dlq_consumer.py
                                 │                              (replayed by runner)
   ┌─────────────────────────────┼──────────────────────────────────────────────┐
   │            LangGraph STATE MACHINE   (src/graph/graph.py)                    │
   │            driven forever by  src/graph/runner.py                            │
   │                                                                             │
   │   conditional entry point (route_cycle_start):                              │
   │     • resume training_poller  (if a Modal job is in flight)                 │
   │     • resume ab_test_node     (if a shadow window is open)                  │
   │     • resume canary_node      (if a canary is in flight)                    │
   │     • else ↓ start fresh                                                    │
   │                                                                             │
   │   log_monitor ──► failure_detector ──(has_failures?)──► example_curator     │
   │     pull 500        4 detectors:        │  no                  │            │
   │     recent logs     • Hallucination     │                      ▼            │
   │                       (NLI entailment)  └────────────►  data_validator      │
   │                     • Drift (Mahalanobis)                      │            │
   │                     • Refusal (regex+semantic)                 ▼            │
   │                     • Format (JSON + KL div)          fine_tune_trigger     │
   │                                                       (≥500 ex + drift +    │
   │   example_curator pipeline (curator.py):              6h cooldown?)         │
   │     1 HDBSCAN cluster                                          │ yes        │
   │     2 Presidio PII scrub  ← BEFORE teacher (compliance)        ▼            │
   │     3 GPT-4o teacher correction (RAG-grounded)        audit_logger_pre_train│
   │     4 Presidio scrub again (defence-in-depth)                  │ (HMAC)     │
   │     5 MinHash LSH dedup                                        ▼            │
   │     6 Quality filter (poison/ROUGE/confidence)        lora_trainer ─────────┼──► Modal A100 GPU
   │     7 INSERT training_examples                                 │            │    (modal_worker.py)
   │                                                                ▼            │    trains LoRA adapter,
   │                                                       training_poller ◄─────┼──  saves to Modal Volume
   │                                                       (poll Modal)          │
   │                                              completed │  │ failed/timeout   │
   │                                                        ▼  └──► rollback_node │
   │                                                  eval_runner                 │
   │                                                  • Safety battery (Llama Guard 3)
   │                                                  • RAGAS (faithfulness/relevancy/recall)
   │                                                  • REAL challenger model (challenger.py)
   │                                              passed │  │ failed              │
   │                                                     ▼  └──► audit_logger_rollback
   │                                                ab_test_node                  │
   │                                                (collect 48h / 1000 req       │
   │                                                 shadow traffic)              │
   │                                            ready │  │ not ready → END (recheck next cycle)
   │                                                  ▼                           │
   │                                          promotion_decider                   │
   │                                          PromotionGate — 4 gates:            │
   │                                            1 safety == 1.0                    │
   │                                            2 A/B window complete              │
   │                                            3 stat. significance (Welch + d)   │
   │                                            4 RAGAS improvement ≥ 0.03          │
   │                                       promote │  │ reject                     │
   │                                              ▼  └──► audit_logger_rollback ──► rollback_node
   │                                          canary_node (optional live rollout)  │
   │                                       promote │  │ rollback                   │
   │                                              ▼  └──► audit_logger_rollback     │
   │                                       audit_logger_promote  (HMAC, write-before-act)
   │                                              ▼                                │
   │                                          promote_model  (atomic DB swap +     │
   │                                          refresh drift/format baselines)      │
   │                                              ▼                                │
   │                                             END  ──► runner sleeps (phase-aware) ──► next cycle
   └─────────────────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
        Postgres side-tables written throughout:
          failure_classifications · training_examples · training_runs ·
          eval_runs · model_versions · audit_trail (INSERT-only, HMAC) ·
          drift_baselines · shadow_logs · drift_trend_history ·
          failure_attributions · knowledge_documents · eval_set · pipeline_metrics

        Redis (cross-cycle state):  pipeline:full_state (durable resume) ·
          pipeline:state (API summary) · detector windows · cost counters ·
          shadow:abort / canary keys

        Observability:  Prometheus /metrics  →  Grafana dashboards + alerts
```

**Key timing detail:** the loop does **not** block waiting for slow work. Training,
the 48h shadow window, and the canary all "park" by returning `END`, and the next
cycle's **conditional entry point** jumps straight back to the in-flight node. The
runner paces itself per phase (60s monitoring → 30min during the A/B window) so it
isn't doing 2,880 pointless wake-ups during a 48h test.

---

## 4. What Every Folder Does (top level)

| Folder / file | Responsibility |
|---|---|
| [`src/`](src/) | All application source code (everything below). |
| [`src/api/`](src/api/) | FastAPI app, middleware wiring, and all HTTP routers (control + observability). |
| [`src/middleware/`](src/middleware/) | The LLM interceptor that captures production traffic with <5ms overhead. |
| [`src/kafka/`](src/kafka/) | Event streaming — producer, consumer, DLQ replayer, topic defs, message schemas. |
| [`src/db/`](src/db/) | Database layer — connection pool, ORM models, and the repository pattern (no raw SQL in nodes). |
| [`src/detection/`](src/detection/) | The 4 failure detectors + their orchestrator, plus drift prediction and threshold calibration. |
| [`src/curation/`](src/curation/) | Turns raw failures into clean training examples (cluster → PII → teacher → dedup → quality). |
| [`src/training/`](src/training/) | Decides when to train, builds the dataset, and runs LoRA training on Modal. |
| [`src/inference/`](src/inference/) | Loads the trained base+adapter model for real evaluation (no stub in prod). |
| [`src/evaluation/`](src/evaluation/) | Safety battery, RAGAS, statistical tests, the eval orchestrator, and the eval-set factory. |
| [`src/shadow/`](src/shadow/) | Shadow A/B routing + collection, the 4-gate promotion decision, and the canary controller. |
| [`src/audit/`](src/audit/) | HMAC signing + the write-before-act audit logger (tamper-evident decision log). |
| [`src/graph/`](src/graph/) | The LangGraph state machine: state shape, nodes, edges, graph build, and the perpetual runner. |
| [`src/retrieval/`](src/retrieval/) | The RAG retriever (numpy-cosine vector store) that grounds the teacher in domain knowledge. |
| [`src/attribution/`](src/attribution/) | Failure attribution — which past training examples most influenced a given failure. |
| [`src/monitoring/`](src/monitoring/) | Prometheus metrics, cost tracking, PagerDuty alerts, Grafana dashboards + alert rules. |
| [`src/config/`](src/config/) | Settings (pydantic-settings), Vault secret fetching, structured logging config. |
| [`alembic/`](alembic/) | Database schema migrations 001–012 (head = 012). |
| [`scripts/`](scripts/) | One-off operational scripts: seeding baselines/eval/KB, manual rollback, audit verification, reports. |
| [`tests/`](tests/) | Unit + integration tests (~166 collected) and JSON fixtures. |
| [`docs/`](docs/) | Operator docs: data retention, retrieval scaling, multi-tenancy. |
| [`k8s/`](k8s/) | Kubernetes manifests (deployments, service, HPA, ingress, secrets, configmap). |
| [`monitoring/`](monitoring/) | Prometheus scrape config. |
| `docker-compose*.yml`, `Makefile`, `pyproject.toml`, `uv.lock` | Local stack, task shortcuts, dependencies/lockfile. |
| `.env`, `.env.example` | Runtime configuration (see deployment notes in [`prevknowledgehere.md`](prevknowledgehere.md) §0). |

---

## 5. File-by-File Reference

### `src/api/` — HTTP front door
| File | What it does |
|---|---|
| [`main.py`](src/api/main.py) | Builds the FastAPI app, attaches Gzip/CORS/interceptor middleware, mounts all routers, checks DB health at startup. |
| [`dependencies.py`](src/api/dependencies.py) | Shared FastAPI dependencies (DB sessions, etc.). |
| [`routers/health.py`](src/api/routers/health.py) | `/health` — liveness + degraded flags (Vault, premise-missing, etc.). |
| [`routers/metrics.py`](src/api/routers/metrics.py) | `/metrics` (Prometheus) and `/metrics/cost`. |
| [`routers/pipeline.py`](src/api/routers/pipeline.py) | `/pipeline/status` — current phase, cycle counts, drift, from the Redis summary. |
| [`routers/models.py`](src/api/routers/models.py) | Model registry queries (versions, production model). |
| [`routers/audit.py`](src/api/routers/audit.py) | Read-only audit access + `/audit/lineage/{version}` (logs→examples→run→model). |
| [`routers/shadow.py`](src/api/routers/shadow.py) | Shadow controls incl. `/shadow/abort` and `/shadow/canary/*`. |
| [`routers/training.py`](src/api/routers/training.py) | Training views + `GET\|DELETE /training/examples/by-source` (retract by source doc). |
| [`routers/drift.py`](src/api/routers/drift.py) | `/drift/trend*` — predictive drift early-warning (RFC-001). |
| [`routers/eval.py`](src/api/routers/eval.py) | `/eval/set/*` — eval-set summary, factory history/trigger, example delete (RFC-002). |
| [`routers/attribution.py`](src/api/routers/attribution.py) | `/attribution/*` — influence reports + retraction (RFC-003). |
| [`routers/knowledge.py`](src/api/routers/knowledge.py) | `/knowledge/*` — ingest/search domain docs; KB size guard. |

### `src/middleware/`
| File | What it does |
|---|---|
| [`llm_interceptor.py`](src/middleware/llm_interceptor.py) | Captures every LLM request after it's served (fire-and-forget), computes cost/latency, emits to Kafka, updates Prometheus. |

### `src/kafka/` — event streaming
| File | What it does |
|---|---|
| [`producer.py`](src/kafka/producer.py) | Idempotent exactly-once producer; routes undeliverable messages to the DLQ. |
| [`consumer.py`](src/kafka/consumer.py) | At-least-once consumer; commits offset only after the handler succeeds. |
| [`dlq_consumer.py`](src/kafka/dlq_consumer.py) | `DLQReplayer` — drains `pipeline.dlq`, replays to the original topic, drops after N attempts. |
| [`topics.py`](src/kafka/topics.py) | The 3 topic configs (events / training / dlq) + idempotent topic creation. |
| [`schemas/llm_event.py`](src/kafka/schemas/llm_event.py) | `LLMEvent` Pydantic message schema (auto event_id/timestamp; `is_rag`). |
| [`schemas/training_event.py`](src/kafka/schemas/training_event.py) | Training-lifecycle event schema. |

### `src/db/` — persistence
| File | What it does |
|---|---|
| [`connection.py`](src/db/connection.py) | asyncpg engine + pool; `get_db()` (commit-on-success); startup health check. |
| [`models.py`](src/db/models.py) | All SQLAlchemy 2.0 ORM tables (llm_logs, failure_classifications, training_examples, model_versions, training_runs, eval_runs, audit_trail, drift_baselines, eval_set, knowledge_documents, …). |
| [`repositories/llm_logs.py`](src/db/repositories/llm_logs.py) | Insert/query raw LLM events; recent completions for baselines/replay. |
| [`repositories/model_versions.py`](src/db/repositories/model_versions.py) | Atomic promote/rollback, training-run records, active drift baseline. |
| [`repositories/training_examples.py`](src/db/repositories/training_examples.py) | Upsert (dedup-safe), pending counts, mark-used, dedup rehydrate, by-source retraction. |
| [`repositories/eval_runs.py`](src/db/repositories/eval_runs.py) | Eval-run CRUD + latest-for-version lookups. |
| [`repositories/eval_set.py`](src/db/repositories/eval_set.py) | Eval-set CRUD, factory inserts, weighted eviction, mark-accessed. |
| [`repositories/audit_trail.py`](src/db/repositories/audit_trail.py) | **Insert/select only** — no update/delete methods exist (by design). |
| [`repositories/drift_trend.py`](src/db/repositories/drift_trend.py) | Predictive-drift trend history persistence (RFC-001). |
| [`repositories/attribution.py`](src/db/repositories/attribution.py) | Failure-attribution storage + retraction (RFC-003). |
| [`repositories/knowledge.py`](src/db/repositories/knowledge.py) | Knowledge-document storage + embedding retrieval support. |

### `src/detection/` — failure detection
| File | What it does |
|---|---|
| [`hallucination.py`](src/detection/hallucination.py) | NLI entailment (`nli-deberta-v3-base`); premise = retrieved context or prompt; score = 1 − entailment, flag > 0.50. |
| [`drift.py`](src/detection/drift.py) | Mahalanobis distance vs a known-good baseline; rolling window; baseline staleness/refresh. |
| [`refusal.py`](src/detection/refusal.py) | Two-stage (regex + semantic) refusal detection; rolling refusal-rate creep. |
| [`format_validator.py`](src/detection/format_validator.py) | Optional JSON validity + output-length KL divergence; refreshable length baseline. |
| [`failure_classifier.py`](src/detection/failure_classifier.py) | Runs all 4 detectors concurrently, collapses correlated failures, emits per-type counters. |
| [`drift_predictor.py`](src/detection/drift_predictor.py) | RFC-001 — linear regression on drift window → predicts hours-to-threshold. |
| [`calibrator.py`](src/detection/calibrator.py) | Suggests threshold adjustments from observed curation drop-rate (suggest-only by default). |

### `src/curation/` — failure → training data
| File | What it does |
|---|---|
| [`curator.py`](src/curation/curator.py) | Orchestrates the 7-step pipeline; **PII-scrubs before the teacher call**; persists examples. |
| [`clustering.py`](src/curation/clustering.py) | HDBSCAN clustering of failures (scales down to small batches) for diverse data. |
| [`teacher.py`](src/curation/teacher.py) | GPT-4o teacher; RAG-grounded corrections; semantic (MiniLM) self-consistency; retry/backoff + cost breaker; returns a `GroundingResult`. |
| [`pii_scrubber.py`](src/curation/pii_scrubber.py) | Presidio PII scrubbing; fail-closed (drop on any error). |
| [`deduplicator.py`](src/curation/deduplicator.py) | Exact SHA-256 + MinHash LSH near-dup detection; restart-survivable index. |
| [`quality_filter.py`](src/curation/quality_filter.py) | Final gate: poison detection, ROUGE-L checks, confidence/quality thresholds. |

### `src/training/` — when & how to train
| File | What it does |
|---|---|
| [`trigger.py`](src/training/trigger.py) | Decides whether to train: ≥500 examples, drift (soft gate, format/refusal-exempt), 6h cooldown. |
| [`dataset_builder.py`](src/training/dataset_builder.py) | Builds JSONL via the model's **chat template**; mixes recency-weighted replay (~25%). |
| [`lora_config.py`](src/training/lora_config.py) | LoRA hyperparameters (r=16, alpha=32, 3 epochs, 4-bit). |
| [`modal_worker.py`](src/training/modal_worker.py) | Submits/runs LoRA training on a remote Modal A100; persists artifacts to a Modal Volume. |
| [`job_poller.py`](src/training/job_poller.py) | Wraps Modal polling with timeout → status (running/completed/failed/timeout). |

### `src/inference/`
| File | What it does |
|---|---|
| [`challenger.py`](src/inference/challenger.py) | `HFModelRunner` loads base + LoRA via `merge_and_unload()`; `verify_adapter_distinct()` blocks no-op adapters. |

### `src/evaluation/` — proving the challenger is better
| File | What it does |
|---|---|
| [`safety_battery.py`](src/evaluation/safety_battery.py) | 100 adversarial prompts scored by **Llama Guard 3** (keyword fallback); fail-closed in prod. |
| [`ragas_runner.py`](src/evaluation/ragas_runner.py) | RAGAS faithfulness / answer-relevancy / context-recall on the eval set. |
| [`statistical_tests.py`](src/evaluation/statistical_tests.py) | Welch t-test + Cohen's d + regression guard (zero-variance safe). |
| [`eval_orchestrator.py`](src/evaluation/eval_orchestrator.py) | Combines safety + RAGAS; re-evaluates the incumbent on the same locked set. |
| [`eval_factory.py`](src/evaluation/eval_factory.py) | RFC-002 — auto-grows the eval set from clustered prod prompts; weighted eviction. |

### `src/shadow/` — A/B + promotion decision
| File | What it does |
|---|---|
| [`router.py`](src/shadow/router.py) | Routes ~10% of traffic to the challenger silently; stratified sampling; abort key. |
| [`ab_collector.py`](src/shadow/ab_collector.py) | Aggregates the 48h/1000-req shadow window; prunes old shadow logs. |
| [`promotion_gate.py`](src/shadow/promotion_gate.py) | The 4 sequential promotion gates (safety, window, significance, RAGAS). |
| [`canary.py`](src/shadow/canary.py) | Optional live canary with real-time auto-abort on error/safety spikes. |

### `src/audit/` — tamper-evident decisions
| File | What it does |
|---|---|
| [`hmac_signer.py`](src/audit/hmac_signer.py) | Canonical-JSON HMAC-SHA256 signing; constant-time verify; Vault key loading. |
| [`logger.py`](src/audit/logger.py) | **Writes the audit entry before the action executes** (recoverable-by-design). |
| [`schemas.py`](src/audit/schemas.py) | `AuditEvent` shape (event_type, decision, rationale, snapshot, before/after). |

### `src/graph/` — the orchestrator
| File | What it does |
|---|---|
| [`runner.py`](src/graph/runner.py) | The perpetual loop: phase-aware pacing, state rehydrate/persist, baseline refresh, calibration, DLQ replay, cost breaker, graceful shutdown. |
| [`graph.py`](src/graph/graph.py) | Builds/compiles the LangGraph; conditional entry point; promote/rollback/audit/canary nodes; pluggable checkpointer. |
| [`state.py`](src/graph/state.py) | `PipelineState` TypedDict that flows through every node. |
| [`edges.py`](src/graph/edges.py) | All conditional routing functions between nodes. |
| [`nodes/log_monitor.py`](src/graph/nodes/log_monitor.py) | Pulls the last ~500 LLM events. |
| [`nodes/failure_detector.py`](src/graph/nodes/failure_detector.py) | Runs detection; holds detector singletons; fires drift-prediction + attribution. |
| [`nodes/example_curator.py`](src/graph/nodes/example_curator.py) | Invokes the curation pipeline on detected failures. |
| [`nodes/data_validator.py`](src/graph/nodes/data_validator.py) | Counts pending examples; respects pause. |
| [`nodes/fine_tune_trigger.py`](src/graph/nodes/fine_tune_trigger.py) | Applies the training trigger conditions. |
| [`nodes/lora_trainer.py`](src/graph/nodes/lora_trainer.py) | Builds the dataset + submits the Modal training job. |
| [`nodes/training_poller.py`](src/graph/nodes/training_poller.py) | Polls Modal; routes to eval (done) or rollback (failed/timeout). |
| [`nodes/eval_runner.py`](src/graph/nodes/eval_runner.py) | Loads the real challenger, runs safety + RAGAS, locks the eval snapshot. |
| [`nodes/ab_test_node.py`](src/graph/nodes/ab_test_node.py) | Collects the shadow window; marks ready when 48h/1000 req reached. |
| [`nodes/promotion_decider.py`](src/graph/nodes/promotion_decider.py) | Runs the promotion gate; routes to canary/promote or rollback. |
| [`nodes/rollback_node.py`](src/graph/nodes/rollback_node.py) | Executes rollback + clears in-flight state markers. |
| [`nodes/audit_logger.py`](src/graph/nodes/audit_logger.py) | Generic audit node helper. |

### `src/retrieval/` · `src/attribution/` · `src/monitoring/` · `src/config/`
| File | What it does |
|---|---|
| [`retrieval/retriever.py`](src/retrieval/retriever.py) | `DocumentRetriever` — numpy-cosine RAG over `knowledge_documents`; 3rd tier of teacher grounding. |
| [`attribution/influence.py`](src/attribution/influence.py) | `InfluenceBackend` protocol + embedding-based influence scoring. |
| [`attribution/attributor.py`](src/attribution/attributor.py) | Scores which training examples most influenced a failure; stores top-K. |
| [`monitoring/metrics.py`](src/monitoring/metrics.py) | All Prometheus counters/gauges/histograms. |
| [`monitoring/cost_tracker.py`](src/monitoring/cost_tracker.py) | Tracks teacher + GPU + judge spend in Redis; monthly budget breaker. |
| [`monitoring/alerts.py`](src/monitoring/alerts.py) | PagerDuty alerting (deduped). |
| [`monitoring/grafana/alerts.yaml`](src/monitoring/grafana/alerts.yaml) | Alert rules as code (predictive drift, DLQ backlog, canary abort, …). |
| [`monitoring/grafana_dashboards/`](src/monitoring/grafana_dashboards/) | Dashboard JSON (model quality, pipeline overview). |
| [`config/settings.py`](src/config/settings.py) | Single source of truth for all config (pydantic-settings). |
| [`config/vault.py`](src/config/vault.py) | Fetches secrets (HMAC key) from Vault; fail-loud in prod. |
| [`config/logging.py`](src/config/logging.py) | structlog config (console in dev, JSON in prod). |

### `alembic/` — schema migrations
| File | What it does |
|---|---|
| [`versions/001_initial_schema.py`](alembic/versions/001_initial_schema.py) | Creates the 8 core tables + indexes; seeds the `v7` model. (`llm_logs` is a **plain, unpartitioned** table.) |
| [`versions/002_audit_trail.py`](alembic/versions/002_audit_trail.py) | Row-Level Security making `audit_trail` INSERT/SELECT-only. |
| [`versions/003_model_registry.py`](alembic/versions/003_model_registry.py) | Model-registry refinements. |
| `versions/004`–`012` | Grounding/versioning/calibration, grounded-teacher source tracking, drift trend, eval factory, failure attribution, knowledge base, replay distribution, pipeline metrics, tenant scaffold. **Head = 012.** |

### `scripts/` — operational tooling
| File | What it does |
|---|---|
| [`seed_baseline.py`](scripts/seed_baseline.py) | Seeds the drift baseline (centroid + inverse covariance). **Run before first traffic.** |
| [`seed_eval_set.py`](scripts/seed_eval_set.py) | Seeds the held-out eval set. **Run before first traffic.** |
| [`seed_knowledge_base.py`](scripts/seed_knowledge_base.py) | Seeds domain docs for RAG grounding. **Run before first traffic.** |
| [`reproduce_dataset.py`](scripts/reproduce_dataset.py) | Fetches the exact dataset of a past run from the Modal Volume. |
| [`manual_rollback.py`](scripts/manual_rollback.py) | Operator-triggered rollback (audited). |
| [`verify_audit_chain.py`](scripts/verify_audit_chain.py) | Re-verifies the HMAC chain of the audit trail. |
| [`attribution_report.py`](scripts/attribution_report.py) | CLI report of failure attributions. |
| [`eval_factory_status.py`](scripts/eval_factory_status.py) | CLI status of the eval-set factory. |
| [`check_redis.py`](scripts/check_redis.py) | Redis connectivity/inspection helper. |

### `tests/` · `docs/` · `k8s/` · `monitoring/`
| Path | What it does |
|---|---|
| [`tests/`](tests/) | ~166 tests: unit (`tests/unit/...`), integration (`tests/integration/...`), hardening suites, and JSON fixtures (`tests/fixtures/`). |
| [`docs/DATA_RETENTION.md`](docs/DATA_RETENTION.md) | Real retention options for `llm_logs` (pg_partman / scheduled DELETE). |
| [`docs/RETRIEVAL_SCALING.md`](docs/RETRIEVAL_SCALING.md) | pgvector/IVFFlat path for scaling the KB beyond numpy-cosine. |
| [`docs/MULTI_TENANCY.md`](docs/MULTI_TENANCY.md) | The path to multi-tenancy (currently single-tenant). |
| [`k8s/`](k8s/) | Kubernetes manifests for the API + pipeline deployments, service, HPA, ingress, secrets, configmap. |
| [`monitoring/prometheus.yml`](monitoring/prometheus.yml) | Prometheus scrape configuration. |

---

## 6. The Two Processes You Actually Run

1. **The API** (`uvicorn src.api.main:app`) — receives your traffic, captures it,
   and serves the control/observability endpoints. This is what your application
   talks to.
2. **The runner** (`python -m src.graph.runner`) — the autonomous brain that loops
   forever, doing detection → curation → training → eval → A/B → promote/rollback.

They communicate through **Postgres** (durable records), **Kafka** (the event stream),
and **Redis** (cross-cycle state, cost counters, abort switches). Everything the
runner decides is observable through the API and Prometheus, and every consequential
action is written — signed — to the audit trail first.

> **Before first production traffic**, seed the three baselines (drift, eval set,
> knowledge base) and set the production env flags (`CHECKPOINTER_BACKEND=postgres`,
> `EVAL_REAL_INFERENCE=true`, …). See [`prevknowledgehere.md`](prevknowledgehere.md)
> §0 "Deployment-day requirements" for the full checklist.
