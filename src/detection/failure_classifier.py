"""
Orchestrates all 4 failure detectors in parallel.
All detectors run concurrently — total latency ≈ slowest single detector.
Processes 50–500 events per call.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Literal

import structlog

from src.detection.hallucination import HallucinationDetector
from src.detection.drift import DriftDetector
from src.detection.refusal import RefusalDetector
from src.detection.format_validator import FormatValidator
from src.monitoring.metrics import failures_detected_total, drift_score_gauge

log = structlog.get_logger()

FailureType = Literal["hallucination", "semantic_drift", "refusal_creep", "format_regression"]


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

        # NLI premise: prefer the retrieved grounding context; fall back to the
        # prompt when a call carried no context (non-RAG traffic).
        hall_pairs = [
            (e.get("retrieved_context") or e.get("prompt", ""), e.get("completion", ""))
            for e in log_events
        ]

        hall_scores, refusal_results, format_results = await asyncio.gather(
            self._hall.score_batch(hall_pairs),
            self._refusal.classify_batch(log_events),
            self._format.validate_batch(log_events),
        )

        for i, event in enumerate(log_events):
            log_id = str(event.get("id", event.get("event_id", f"unknown_{i}")))
            prompt = event.get("prompt", "")
            completion = event.get("completion", "")

            # Hallucination
            hall_score = hall_scores[i] if i < len(hall_scores) else 0.0
            if hall_score > 0.0:
                from src.config.settings import settings
                if hall_score > settings.hallucination_threshold:
                    batch.events.append(FailureEvent(
                        llm_log_id=log_id, prompt=prompt, completion=completion,
                        failure_type="hallucination", score=hall_score,
                    ))
                    failures_detected_total.labels(failure_type="hallucination").inc()

            # Drift (per-event score, rolling window updated inside scorer)
            drift_score = self._drift.score(completion)

            # Refusal
            ref_type, ref_score = refusal_results[i] if i < len(refusal_results) else ("", 0.0)
            if ref_type == "refusal_creep":
                batch.events.append(FailureEvent(
                    llm_log_id=log_id, prompt=prompt, completion=completion,
                    failure_type="refusal_creep", score=ref_score,
                ))
                failures_detected_total.labels(failure_type="refusal_creep").inc()

            # Format
            fmt_type, fmt_score = format_results[i] if i < len(format_results) else ("", 0.0)
            if fmt_type == "format_regression":
                batch.events.append(FailureEvent(
                    llm_log_id=log_id, prompt=prompt, completion=completion,
                    failure_type="format_regression", score=fmt_score,
                ))
                failures_detected_total.labels(failure_type="format_regression").inc()

        # Semantic drift is a batch-level signal (rolling window)
        batch.drift_score = self._drift.rolling_drift_score
        drift_score_gauge.set(batch.drift_score)

        if self._drift.is_drifting():
            for i, event in enumerate(log_events):
                log_id = str(event.get("id", event.get("event_id", f"unknown_{i}")))
                drift_score = self._drift.score(event.get("completion", ""))
                if drift_score > 0:
                    batch.events.append(FailureEvent(
                        llm_log_id=log_id,
                        prompt=event.get("prompt", ""),
                        completion=event.get("completion", ""),
                        failure_type="semantic_drift",
                        score=drift_score,
                    ))
            failures_detected_total.labels(failure_type="semantic_drift").inc(len(log_events))

        log.info(
            "failure_classification_complete",
            total=batch.total_processed,
            failures=len(batch.events),
            drift_score=batch.drift_score,
        )
        return batch
