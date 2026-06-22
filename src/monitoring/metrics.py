"""
Prometheus metrics for the entire pipeline.
Import and increment these counters from anywhere in the codebase.
"""

from prometheus_client import Counter, Histogram, Gauge, Summary

# ── LLM Traffic ──────────────────────────────────────────────────────────────
llm_calls_total = Counter(
    "llm_calls_total",
    "Total number of LLM calls intercepted",
    ["model_version", "finish_reason"],
)

llm_latency_histogram = Histogram(
    "llm_latency_seconds",
    "LLM call latency in seconds",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

llm_tokens_counter = Counter(
    "llm_tokens_total",
    "Total LLM tokens generated",
    ["token_type"],  # prompt, completion
)

llm_cost_counter = Counter(
    "llm_cost_usd_total",
    "Total LLM cost in USD",
)

# ── Failure Detection ─────────────────────────────────────────────────────────
failures_detected_total = Counter(
    "failures_detected_total",
    "Total detected LLM failures",
    ["failure_type"],
)

drift_score_gauge = Gauge(
    "drift_rolling_score",
    "Rolling mean Mahalanobis drift score",
)

hallucination_score_histogram = Histogram(
    "hallucination_score",
    "Distribution of hallucination probe scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

# ── Detection hardening (June 2026) ────────────────────────────────────────────
hallucination_premise_missing_total = Counter(
    "hallucination_premise_missing_total",
    "RAG-flagged calls whose retrieved_context was missing, so the NLI premise "
    "silently fell back to the prompt (factual grounding could not be verified)",
)
detector_insufficient_data_total = Counter(
    "detector_insufficient_data_total",
    "Times an aggregate detector signal was suppressed for too few samples",
    ["detector"],  # drift | refusal | format
)
correlated_failures_collapsed_total = Counter(
    "correlated_failures_collapsed_total",
    "Duplicate failure events on the same log collapsed into one (multi-detector)",
)
drift_baseline_age_hours = Gauge(
    "drift_baseline_age_hours",
    "Age of the active drift baseline in hours (-1 if none loaded)",
)
format_baseline_age_hours = Gauge(
    "format_baseline_age_hours",
    "Age of the format length baseline in hours (-1 if none seeded)",
)
detector_window_size = Gauge(
    "detector_window_size",
    "Current number of samples in a detector's rolling window",
    ["detector"],  # drift | refusal | format
)

# ── Curation ─────────────────────────────────────────────────────────────────
examples_curated_total = Counter(
    "examples_curated_total",
    "Total training examples curated",
)

examples_dropped_pii = Counter(
    "examples_dropped_pii_total",
    "Examples dropped due to PII scrub failure",
)

examples_dropped_dedup = Counter(
    "examples_dropped_dedup_total",
    "Examples dropped as duplicates",
)

examples_dropped_quality = Counter(
    "examples_dropped_quality_total",
    "Examples dropped due to low quality score",
)

# ── Curation hardening (June 2026) ─────────────────────────────────────────────
examples_pre_scrubbed_total = Counter(
    "examples_pre_scrubbed_total",
    "Examples whose prompt+completion were PII-scrubbed BEFORE the teacher API "
    "call (so raw PII never leaves for OpenAI)",
)
examples_dropped_pre_scrub_total = Counter(
    "examples_dropped_pre_scrub_total",
    "Examples dropped (fail-closed) because PII scrubbing failed before the "
    "teacher was ever called",
)
teacher_rate_limit_retries_total = Counter(
    "teacher_rate_limit_retries_total",
    "Teacher API calls retried after a rate-limit / transient error",
)
teacher_dropped_rate_limited_total = Counter(
    "teacher_dropped_rate_limited_total",
    "Teacher corrections dropped after exhausting rate-limit retries",
)
clustering_bypassed_total = Counter(
    "clustering_bypassed_total",
    "Curation cycles where clustering was bypassed (all failures noise-labelled)",
)
dedup_index_size = Gauge(
    "dedup_index_size",
    "Number of entries in the in-memory MinHash LSH near-duplicate index",
)

pending_examples_gauge = Gauge(
    "training_examples_pending",
    "Training examples waiting to be used in a run",
)

# ── Training ─────────────────────────────────────────────────────────────────
training_runs_total = Counter(
    "training_runs_total",
    "Total training runs submitted",
    ["status"],  # submitted, completed, failed
)

training_loss_gauge = Gauge(
    "training_final_loss",
    "Final training loss of the last completed run",
)

# ── Evaluation ────────────────────────────────────────────────────────────────
eval_faithfulness_gauge = Gauge("eval_faithfulness", "RAGAS faithfulness score")
eval_relevancy_gauge = Gauge("eval_answer_relevancy", "RAGAS answer relevancy score")
eval_recall_gauge = Gauge("eval_context_recall", "RAGAS context recall score")
eval_safety_gauge = Gauge("eval_safety_score", "Safety battery pass rate")

# ── Training / evaluation hardening (June 2026) ────────────────────────────────
challenger_adapter_checks_total = Counter(
    "challenger_adapter_checks_total",
    "Adapter-applied verification checks before eval",
    ["result"],  # verified | identical | skipped | error
)
safety_classifier_unavailable_blocks_total = Counter(
    "safety_classifier_unavailable_blocks_total",
    "Responses blocked (treated unsafe) because the safety classifier was "
    "unavailable and a real classifier is required",
)
eval_incumbent_reeval_total = Counter(
    "eval_incumbent_reeval_total",
    "Incumbent re-evaluations on the locked eval-set snapshot for a fair delta",
)
training_trigger_drift_exempt_total = Counter(
    "training_trigger_drift_exempt_total",
    "Training triggers fired on example count alone (format/refusal-dominant, "
    "drift gate exempted)",
)
challenger_regression_blocks_total = Counter(
    "challenger_regression_blocks_total",
    "Promotions blocked early because the challenger's mean quality delta was negative",
)
teacher_model_cost_usd = Gauge(
    "teacher_model_cost_usd", "Estimated teacher (GPT-4o) spend on the last curation run"
)
replay_buffer_ratio = Gauge(
    "replay_buffer_ratio", "Fraction of the last training dataset that was replay (known-good) examples"
)
teacher_corrections_rejected_grounding_total = Counter(
    "teacher_corrections_rejected_grounding_total",
    "Teacher corrections dropped because NLI grounding score was below threshold",
)
# Predictive drift early warning (RFC-001)
drift_early_warnings_total = Counter(
    "drift_early_warnings_total",
    "Number of predictive drift alerts sent",
)
drift_predicted_trigger_hours = Gauge(
    "drift_predicted_trigger_hours",
    "Predicted hours until drift threshold is crossed (-1.0 if not trending up)",
)
drift_trend_slope = Gauge(
    "drift_trend_slope",
    "Linear regression slope of the drift rolling window (positive = worsening)",
)
drift_r_squared = Gauge(
    "drift_r_squared",
    "R-squared goodness-of-fit for the current drift trend line",
)
# Continuous eval factory (RFC-002)
eval_set_size = Gauge(
    "eval_set_size_total", "Current active eval set size", ["source"],
)
eval_factory_runs_total = Counter(
    "eval_factory_runs_total", "Total number of eval factory runs completed",
)
eval_factory_examples_added_total = Counter(
    "eval_factory_examples_added_total", "Total eval examples added by the factory",
)
eval_factory_examples_evicted_total = Counter(
    "eval_factory_examples_evicted_total", "Total eval examples LRU-evicted by the factory",
)
eval_factory_examples_skipped_total = Counter(
    "eval_factory_examples_skipped_total", "Eval factory candidates skipped", ["reason"],
)
# Failure attribution (RFC-003)
attribution_computations_total = Counter(
    "attribution_computations_total", "Total failure attribution computations completed",
)
attribution_latency_ms = Histogram(
    "attribution_latency_ms",
    "Time taken for one influence scoring operation (milliseconds)",
    buckets=[10, 50, 100, 250, 500, 1000, 2000, 5000],
)
attribution_candidates_scored = Histogram(
    "attribution_candidates_scored",
    "Number of training candidates scored per attribution",
    buckets=[10, 50, 100, 200, 500],
)
retracted_examples_total = Counter(
    "retracted_examples_total", "Training examples surgically retracted via attribution API",
)
# Domain knowledge retrieval (grounded-teacher fallback)
retrieval_queries_total = Counter(
    "retrieval_queries_total", "Document retrieval queries", ["result"],  # hit | miss
)
knowledge_base_documents = Gauge(
    "knowledge_base_documents", "Number of documents in the knowledge base",
)

promotions_total = Counter(
    "model_promotions_total",
    "Total model promotions to production",
)

rollbacks_total = Counter(
    "model_rollbacks_total",
    "Total model rollbacks",
    ["reason"],
)

# ── Shadow Traffic ────────────────────────────────────────────────────────────
shadow_requests_total = Counter(
    "shadow_requests_total",
    "Total requests routed to shadow model",
    ["challenger_version"],
)

shadow_quality_delta_histogram = Histogram(
    "shadow_quality_delta",
    "Quality delta: challenger - production",
    buckets=[-0.2, -0.1, -0.05, 0, 0.05, 0.1, 0.2],
)

# ── Pipeline State ────────────────────────────────────────────────────────────
pipeline_cycle_duration = Histogram(
    "pipeline_cycle_duration_seconds",
    "Duration of a full pipeline cycle",
)

pipeline_errors_total = Counter(
    "pipeline_errors_total",
    "Total pipeline node errors",
    ["node"],
)
