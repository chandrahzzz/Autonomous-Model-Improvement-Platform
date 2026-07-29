"""Simulator configuration and shared constants."""

from __future__ import annotations

from dataclasses import dataclass, field

# The four failure types the pipeline's detectors recognise. `None`/"healthy"
# means a clean call. These strings are used only as internal labels + stored in
# the row metadata for debugging; the REAL failure_type is assigned by the
# detectors when the pipeline scores the row.
FAILURE_TYPES = ("hallucination", "refusal", "drift", "format_break")

DOMAINS = ("factual", "business_policy", "json_format", "reasoning")

# User cohorts a real product might segment traffic by.
COHORTS = ("free", "pro", "enterprise", "trial", "internal")

# Model to attribute the traffic to for cost calc (matches the interceptor's
# TOKEN_COSTS table keys). Kept cheap so simulated cost stays tiny.
DEFAULT_MODEL = "llama-3-8b"
TOKEN_COSTS = {
    "gpt-4o": {"input": 5e-6, "output": 15e-6},
    "gpt-4o-mini": {"input": 0.15e-6, "output": 0.6e-6},
    "llama-3-8b": {"input": 0.05e-6, "output": 0.05e-6},
    "default": {"input": 0.05e-6, "output": 0.05e-6},
}


@dataclass
class SimConfig:
    """One run's knobs. Rates are per-call probabilities in [0, 1]; how they are
    applied over the run is decided by the scenario (see sim/scenarios.py)."""

    scenario: str = "mixed"
    mode: str = "db"                      # db | kafka | http
    count: int = 500                      # total calls (ignored if duration set)
    duration_seconds: float | None = None  # if set, run for this long at `rate`
    rate: float = 50.0                    # calls/sec (pacing; 0 = as fast as possible)

    # Per-failure-type injection rates (peak rates the scenario ramps toward).
    hallucination_rate: float = 0.0
    refusal_rate: float = 0.0
    format_break_rate: float = 0.0
    drift_rate: float = 0.0

    # Identity / attribution
    model_version: str = "v7"
    model_name: str = DEFAULT_MODEL

    # HTTP mode
    base_url: str = "http://localhost:8000"
    http_path: str = "/sim/llm-call"

    # Reproducibility + realism
    seed: int = 1234
    backfill_minutes: float = 0.0         # spread created_at over the past N minutes
    db_batch_size: int = 200              # rows per commit in db mode

    # Optionally call Groq for genuinely-good healthy answers (default off →
    # templated, deterministic, no network).
    use_groq: bool = False

    cohorts: tuple[str, ...] = field(default_factory=lambda: COHORTS)

    def peak_rate_for(self, failure_type: str) -> float:
        return {
            "hallucination": self.hallucination_rate,
            "refusal": self.refusal_rate,
            "format_break": self.format_break_rate,
            "drift": self.drift_rate,
        }.get(failure_type, 0.0)

    @property
    def any_failures_configured(self) -> bool:
        return any(self.peak_rate_for(f) > 0 for f in FAILURE_TYPES)
