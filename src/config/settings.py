from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal
import secrets
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    secret_key: str = Field(default_factory=lambda: secrets.token_hex(32))

    # Database
    database_url: str = "postgresql+asyncpg://pipeline:password@localhost:5432/finetuning_pipeline"
    database_pool_size: int = 20
    database_max_overflow: int = 10
    database_pool_timeout: int = 30

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 50

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_consumer_group: str = "finetuning-pipeline"
    kafka_topic_llm_events: str = "llm.production.events"
    kafka_topic_training_events: str = "pipeline.training.events"
    kafka_topic_dlq: str = "pipeline.dlq"

    # LLM
    openai_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "continuous-finetuning"
    base_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    teacher_model: str = "gpt-4o"

    # Training Trigger
    training_trigger_dataset_size: int = 500
    training_trigger_drift_threshold: float = 0.15
    training_min_interval_hours: int = 6
    # Drift is a HARD requirement only for failure types that actually move the
    # embedding distribution. Format/refusal regressions accumulate examples
    # without shifting embeddings, so for those the drift gate is soft — example
    # count alone may fire the trigger (#T3).
    trigger_drift_exempt_failure_types: list[str] = ["format_regression", "refusal_creep"]

    # Replay buffer recency weighting (#T2): older known-good logs may encode an
    # earlier, weaker model's behaviour, so we sample recent logs with higher
    # probability via exponential decay over recency rank.
    replay_ratio: float = 0.25
    replay_recency_decay: float = 0.9      # per-rank decay; <1 favours recent logs
    replay_candidate_pool_multiplier: int = 5  # oversample pool before weighting

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = ["q_proj", "v_proj"]
    lora_training_epochs: int = 3
    lora_learning_rate: float = 2e-4
    lora_batch_size: int = 4
    lora_gradient_accumulation: int = 4

    # Evaluation
    eval_set_size: int = 200
    eval_improvement_threshold: float = 0.03
    eval_safety_battery_size: int = 100
    min_eval_examples: int = 50  # Absolute floor before an eval run is trusted
    ab_shadow_traffic_pct: float = 0.10
    ab_min_requests: int = 1000
    ab_min_hours: float = 48.0
    ab_pvalue_threshold: float = 0.05
    ab_cohens_d_threshold: float = 0.10
    # Shadow quality scoring (replaces the old circular self-comparison metric)
    shadow_scoring_strategy: Literal["reference_rouge", "llm_judge"] = "llm_judge"
    shadow_reference_sim_threshold: float = 0.85
    shadow_judge_model: str = "gpt-4o-mini"

    # Real challenger inference (#T1): when enabled, the eval node loads the base
    # model + the trained LoRA adapter (merge_and_unload) and runs eval against
    # that — and verifies the merged model's output actually DIFFERS from the base
    # (i.e. the adapter was applied) before trusting any score. Off in dev (no GPU
    # / gated weights); MUST be on in production so gates can't pass on the base.
    eval_real_inference: bool = False
    eval_require_adapter_verification: bool = True  # enforced only in production
    adapter_verification_probes: int = 3
    # Re-evaluate the incumbent on the SAME locked eval-set snapshot as the
    # challenger (#E2) so a delta reflects the model, not eval-set drift.
    eval_lock_set_snapshot: bool = True
    eval_factory_evict_confidence_quartile: float = 0.25  # evict lowest-conf first (#E3)

    # Safety gate fail-closed (#E1): in production a real safety classifier
    # (Llama Guard) is REQUIRED — if it's unavailable the battery treats responses
    # as unsafe (blocking promotion) instead of silently using brittle keywords.
    safety_require_classifier: bool = True   # enforced only in production
    safety_max_tokens_after_refusal: int = 50  # content past a refusal phrase ⇒ unsafe

    # Detection
    hallucination_threshold: float = 0.50  # NLI: flag if mean non-entailment > this
    drift_mahalanobis_threshold: float = 0.15
    drift_baseline_min_samples: int = 500
    refusal_rate_multiplier: float = 2.0
    format_kl_threshold: float = 0.5

    # Detection hardening (production gaps closed June 2026) — all additive.
    # #2/#5 Baselines auto-refresh after promotion AND age out independently so a
    # long 48h shadow window / rollback storm can't leave them stale.
    drift_baseline_max_age_hours: float = 24.0
    format_baseline_max_age_hours: float = 24.0
    baseline_refresh_check_interval_cycles: int = 10   # how often the runner checks age
    # #3 Rate/window-based signals refuse to fire on too-few samples (returns
    # "insufficient_data" instead of a noisy false-positive on quiet traffic).
    drift_min_window: int = 50
    refusal_min_samples: int = 50
    format_min_samples: int = 100
    # #4 Rolling-window state persists to Redis so a process restart doesn't reset
    # refusal rate / drift trend to a misleading clean slate.
    detector_state_persist_enabled: bool = True
    detector_state_redis_prefix: str = "pipeline:detector_state"
    # #6 Collapse multiple detectors firing on the same log into one failure event
    # (highest severity) so the curator generates one correction, not duplicates.
    correlate_failures_enabled: bool = True

    # Continuous eval factory (RFC-002) — purely additive, off via the flag.
    eval_factory_enabled: bool = True
    eval_factory_trigger_every_n_requests: int = 1000
    eval_factory_max_eval_set_size: int = 500
    eval_factory_min_confidence: float = 0.80
    eval_factory_max_examples_per_run: int = 10
    eval_factory_dedup_cosine_threshold: float = 0.90
    eval_factory_request_counter_key: str = "pipeline:eval_factory:request_count"

    # Domain knowledge retrieval — lets the grounded teacher fetch context when
    # the upstream app didn't attach `retrieved_context`. Additive, off via flag.
    retrieval_enabled: bool = True
    retrieval_top_k: int = 3
    # MiniLM cosine: genuinely-relevant domain queries score ~0.38-0.42, off-topic
    # queries score < 0.05 — so 0.30 grounds real matches while rejecting noise.
    retrieval_min_similarity: float = 0.30
    retrieval_max_context_chars: int = 4000

    # Failure attribution (RFC-003) — purely additive, off via the flag.
    attribution_enabled: bool = True
    attribution_top_k: int = 10
    attribution_max_candidates: int = 500
    attribution_max_failures_per_cycle: int = 3
    attribution_retention_days: int = 30

    # Predictive drift early warning (RFC-001) — purely additive, off via the flag.
    drift_prediction_enabled: bool = True
    drift_alert_horizon_hours: float = 24.0
    drift_min_window_for_prediction: int = 50
    drift_prediction_interval_cycles: int = 5
    drift_trend_redis_key: str = "pipeline:drift_trend"
    drift_trend_redis_ttl_seconds: int = 3600
    drift_alert_dedupe_seconds: int = 7200

    # Curation
    teacher_confidence_threshold: float = 0.85
    # Self-consistency is now semantic (MiniLM cosine across the votes) instead of
    # ROUGE-L surface overlap, so paraphrases of the same answer count as agreement
    # and divergent meanings don't. Cosine sits lower than ROUGE for paraphrases,
    # hence a dedicated (slightly lower) threshold.
    teacher_semantic_consistency_threshold: float = 0.80
    teacher_consistency_temperature: float = 0.7  # >0 so self-consistency votes differ
    max_concurrent_teacher_calls: int = 20
    curation_cost_budget_usd: float = 25.0  # Per-run circuit breaker
    # OpenAI rate-limit resilience: retry 429 / timeout / connection errors with
    # exponential backoff + jitter instead of silently dropping the example.
    teacher_max_retries: int = 5
    teacher_retry_base_delay_seconds: float = 1.0
    teacher_retry_max_delay_seconds: float = 30.0
    dedup_jaccard_threshold: float = 0.85
    # Rehydrate the in-memory MinHash LSH near-dup index from existing
    # training_examples on the first curation cycle so a restart doesn't let
    # near-duplicates (that the DB exact-hash index can't catch) re-enter.
    dedup_rehydrate_enabled: bool = True
    dedup_rehydrate_limit: int = 50000
    quality_rouge_threshold: float = 0.30
    pii_fail_closed: bool = True

    # Safety classifier
    safety_classifier: Literal["llama_guard", "keyword_fallback"] = "llama_guard"
    together_api_key: str = ""
    llama_guard_model: str = "meta-llama/Meta-Llama-Guard-3-8B"

    # Threshold calibration
    allow_auto_calibration: bool = False   # suggest-only unless True
    calibration_fp_target: float = 0.30    # tolerated curation drop-rate (FP proxy)
    calibration_step: float = 0.05
    calibration_interval_cycles: int = 100
    calibration_min_samples: int = 20

    # Canary (live partial rollout before full promotion)
    # Off by default: it requires a serving layer to call CanaryController.record_result.
    # When enabled without that integration, promotions are gated until verified.
    canary_enabled: bool = False
    canary_traffic_pct: float = 0.05
    canary_window_minutes: int = 60
    canary_min_requests: int = 50
    canary_max_error_rate: float = 0.01
    # Real-time canary abort (#S2): if the running error rate spikes past
    # canary_max_error_rate × this multiplier — once enough requests have been
    # seen — abort immediately mid-window instead of waiting for it to close.
    canary_abort_error_multiplier: float = 2.0
    canary_min_requests_for_abort: int = 20

    # Shadow sampling + retention
    # Stratified temporal sampling (#S1): balance shadow samples across time
    # buckets so a 48h A/B doesn't oversample peak hours.
    shadow_stratified_sampling_enabled: bool = True
    shadow_stratify_bucket_hours: int = 4
    shadow_samples_bucket_key: str = "shadow:samples_by_bucket"
    # shadow_logs retention (#S3): the table is append-only per shadow request;
    # prune rows older than this so it doesn't grow unboundedly.
    shadow_logs_retention_days: int = 30
    shadow_logs_cleanup_interval_cycles: int = 100

    # LangGraph checkpointer (#L1): "postgres" gives crash-resilient resume but
    # needs the langgraph-checkpoint-postgres extra; falls back to in-memory with
    # a loud warning if unavailable. Redis full_state remains the fast recovery path.
    checkpointer_backend: Literal["memory", "postgres"] = "memory"

    # Event-driven fast-path (#L3): when a cycle detects a large failure burst,
    # skip the inter-cycle sleep and re-run immediately (bounded) so curation
    # isn't delayed ~60s behind a severe regression.
    high_severity_failure_threshold: int = 500
    max_consecutive_fast_cycles: int = 5

    # Kafka DLQ replay (#I1): periodically replay dead-lettered events back into
    # the pipeline with bounded attempts instead of letting them rot.
    dlq_replay_enabled: bool = True
    dlq_replay_interval_cycles: int = 20
    dlq_replay_max_per_cycle: int = 100
    dlq_replay_max_attempts: int = 5

    # Knowledge base size guard (#I3): warn (and surface) when the numpy-scan KB
    # grows past the point where per-query latency starts to matter; the
    # documented migration path is pgvector + IVFFlat.
    knowledge_base_size_warn_threshold: int = 5000

    # Cost controls
    monthly_budget_usd: float = 500.0
    single_run_budget_usd: float = 50.0
    modal_a100_hourly_usd: float = 4.0  # Estimated A100 rate for GPU cost tracking

    # Audit
    vault_url: str = "http://localhost:8200"
    vault_token: str = "root"
    hmac_key_path: str = "secret/pipeline/hmac_key"
    vault_required: bool = True  # Production: fail loudly if Vault is unreachable

    # Monitoring
    pagerduty_api_key: str = ""
    pagerduty_service_id: str = ""
    wandb_api_key: str = ""
    wandb_project: str = "continuous-finetuning"

    # Modal
    modal_token_id: str = ""
    modal_token_secret: str = ""

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        if not v.startswith("postgresql+asyncpg://"):
            v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    @field_validator("lora_target_modules", mode="before")
    @classmethod
    def parse_target_modules(cls, v: object) -> list[str]:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return [m.strip() for m in v.split(",")]
        return v  # type: ignore[return-value]


settings = Settings()
