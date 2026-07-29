"""
Orchestrates all 4 failure detectors in parallel.
All detectors run concurrently — total latency ≈ slowest single detector.
Processes 50–500 events per call.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Literal

import structlog

from src.config.settings import settings
from src.detection.hallucination import HallucinationDetector
from src.detection.drift import DriftDetector
from src.detection.refusal import RefusalDetector
from src.detection.format_validator import FormatValidator
from src.monitoring.metrics import (
    failures_detected_total,
    drift_score_gauge,
    hallucination_premise_missing_total,
    correlated_failures_collapsed_total,
)

log = structlog.get_logger()

FailureType = Literal["hallucination", "semantic_drift", "refusal_creep", "format_regression"]

# When several detectors fire on the SAME log, we keep one failure event so the
# curator generates a single correction instead of near-duplicate examples that
# each carry a different failure_type (and would slip past dedup). Higher number
# = higher priority to survive the collapse.
_SEVERITY_PRIORITY: dict[str, int] = {
    "hallucination": 4,
    "semantic_drift": 3,
    "refusal_creep": 2,
    "format_regression": 1,
}


@dataclass
class FailureEvent:
    llm_log_id: str
    prompt: str
    completion: str
    failure_type: FailureType
    score: float
    metadata: dict = field(default_factory=dict)


@dataclass
class FailureBatch:
    events: list[FailureEvent] = field(default_factory=list)
    drift_score: float = 0.0
    total_processed: int = 0

    @property
    def has_failures(self) -> bool:
        return len(self.events) > 0

    def by_type(self, ftype: FailureType) -> list[FailureEvent]:
        return [e for e in self.events if e.failure_type == ftype]


class FailureClassifier:
    def __init__(
        self,
        hallucination_detector: HallucinationDetector,
        drift_detector: DriftDetector,
        refusal_detector: RefusalDetector,
        format_validator: FormatValidator,
    ) -> None:
        self._hall = hallucination_detector
        self._drift = drift_detector
        self._refusal = refusal_detector
        self._format = format_validator

    async def classify_batch(self, log_events: list[dict]) -> FailureBatch:
        """
        Process a batch through all 4 detectors concurrently.
        Each detector is independent — we gather them all.
        """
        batch = FailureBatch(total_processed=len(log_events))

        if not log_events:
            return batch

        # ── NLI premise selection (#1) ─────────────────────────────────────────
        # Prefer the retrieved grounding context; fall back to the prompt only for
        # genuine non-RAG traffic. A call explicitly flagged `is_rag` but missing
        # its context is an upstream logging bug: we still score it (against the
        # prompt) but record premise_source="prompt_fallback" so the score is
        # treated as ungrounded downstream, and we count it for /health.
        hall_pairs: list[tuple[str, str]] = []
        premise_sources: list[str] = []
        for e in log_events:
            ctx = e.get("retrieved_context") or ""
            is_rag = bool(e.get("is_rag", False))
            if ctx:
                premise_sources.append("context")
                hall_pairs.append((ctx, e.get("completion", "")))
            else:
                if is_rag:
                    premise_sources.append("prompt_fallback")  # context expected, absent
                    hallucination_premise_missing_total.inc()
                else:
                    premise_sources.append("prompt")
                hall_pairs.append((e.get("prompt", ""), e.get("completion", "")))

        # Score the three text-pair detectors concurrently. Drift is scored
        # separately (below) because it mutates a rolling window and must run
        # exactly once per completion.
        hall_scores, refusal_results, format_results = await asyncio.gather(
            self._hall.score_batch(hall_pairs),
            self._refusal.classify_batch(log_events),
            self._format.validate_batch(log_events),
        )

        # Score drift exactly once per event (this previously ran twice — once
        # here and again in the is_drifting() block — double-counting every
        # completion into the rolling window and double-running the encoder).
        drift_scores = [self._drift.score(e.get("completion", "")) for e in log_events]

        batch.drift_score = self._drift.rolling_drift_score
        drift_score_gauge.set(batch.drift_score)
        drifting = self._drift.is_drifting()

        # ── Gather candidate failures per log, then collapse correlated ones ────
        for i, event in enumerate(log_events):
            log_id = str(event.get("id", event.get("event_id", f"unknown_{i}")))
            prompt = event.get("prompt", "")
            completion = event.get("completion", "")
            candidates: list[FailureEvent] = []

            # Hallucination. NLI against the bare prompt is an unreliable signal
            # for non-RAG QA (a question doesn't entail its answer), so optionally
            # require real grounding context before flagging.
            hall_score = hall_scores[i] if i < len(hall_scores) else 0.0
            hall_has_context = premise_sources[i] == "context"
            hall_gated = hall_has_context or not settings.hallucination_require_context
            if hall_gated and hall_score > settings.hallucination_threshold:
                candidates.append(FailureEvent(
                    llm_log_id=log_id, prompt=prompt, completion=completion,
                    failure_type="hallucination", score=hall_score,
                    metadata={"premise_source": premise_sources[i],
                              "grounded": premise_sources[i] == "context"},
                ))
                failures_detected_total.labels(failure_type="hallucination").inc()

            # Semantic drift — only when the batch-level signal is firing (it has
            # its own min-window guard; see DriftDetector.is_drifting).
            if drifting and drift_scores[i] > 0:
                candidates.append(FailureEvent(
                    llm_log_id=log_id, prompt=prompt, completion=completion,
                    failure_type="semantic_drift", score=drift_scores[i],
                ))
                failures_detected_total.labels(failure_type="semantic_drift").inc()

            # Refusal
            ref_type, ref_score = refusal_results[i] if i < len(refusal_results) else ("", 0.0)
            if ref_type == "refusal_creep":
                candidates.append(FailureEvent(
                    llm_log_id=log_id, prompt=prompt, completion=completion,
                    failure_type="refusal_creep", score=ref_score,
                ))
                failures_detected_total.labels(failure_type="refusal_creep").inc()

            # Format
            fmt_type, fmt_score = format_results[i] if i < len(format_results) else ("", 0.0)
            if fmt_type == "format_regression":
                candidates.append(FailureEvent(
                    llm_log_id=log_id, prompt=prompt, completion=completion,
                    failure_type="format_regression", score=fmt_score,
                ))
                failures_detected_total.labels(failure_type="format_regression").inc()

            batch.events.extend(self._collapse(candidates))

        log.info(
            "failure_classification_complete",
            total=batch.total_processed,
            failures=len(batch.events),
            drift_score=batch.drift_score,
            drifting=drifting,
        )
        return batch

    @staticmethod
    def _collapse(candidates: list[FailureEvent]) -> list[FailureEvent]:
        """Collapse multiple detector hits on one log into a single event (#6).

        Keeps the highest-severity failure type (tie-broken by score) and records
        every type that fired in `metadata.all_failure_types` + `is_correlated`,
        so the curator produces one correction while retaining full visibility.
        Returns the candidates unchanged when correlation is disabled or there's
        nothing to collapse.
        """
        if not candidates:
            return []
        if len(candidates) == 1 or not settings.correlate_failures_enabled:
            return candidates

        all_types = [c.failure_type for c in candidates]
        winner = max(
            candidates,
            key=lambda c: (_SEVERITY_PRIORITY.get(c.failure_type, 0), c.score),
        )
        winner.metadata = {
            **winner.metadata,
            "is_correlated": True,
            "all_failure_types": all_types,
        }
        correlated_failures_collapsed_total.inc(len(candidates) - 1)
        log.debug("correlated_failures_collapsed", kept=winner.failure_type, all=all_types)
        return [winner]
