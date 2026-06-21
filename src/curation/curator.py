"""
Curation pipeline orchestrator.

Pipeline for each failure event:
  1. Cluster (HDBSCAN) — group similar failures
  2. Teacher correction (GPT-4o) — generate ideal response
  3. PII scrub (Presidio) — fail-closed
  4. Dedup (MinHash LSH) — skip near-duplicates
  5. Quality filter (ROUGE-L + confidence + poison check)
  6. Persist to training_examples table
"""

import asyncio
import structlog

from src.curation.clustering import FailureClusterer
from src.curation.teacher import TeacherModel
from src.curation.pii_scrubber import PIIScrubber
from src.curation.deduplicator import Deduplicator
from src.curation.quality_filter import QualityFilter
from src.detection.failure_classifier import FailureBatch, FailureEvent
from src.monitoring.metrics import (
    examples_curated_total,
    examples_dropped_pii,
    examples_dropped_dedup,
    examples_dropped_quality,
    teacher_model_cost_usd,
)

log = structlog.get_logger()


class CurationPipeline:
    def __init__(self) -> None:
        self._clusterer = FailureClusterer()
        self._teacher = TeacherModel()
        self._pii = PIIScrubber()
        self._dedup = Deduplicator()
        self._quality = QualityFilter()

    async def curate(self, failure_batch: FailureBatch, db) -> list[dict]:
        """
        Process a FailureBatch into validated training examples.
        Returns list of dicts ready for DB insertion.
        """
        from src.db.repositories.training_examples import TrainingExampleRepository
        repo = TrainingExampleRepository(db)

        if not failure_batch.has_failures:
            return []

        # Reset the per-run teacher cost counter (the TeacherModel is a
        # long-lived singleton, so without this the budget breaker would
        # accumulate across every run and trip permanently).
        self._teacher.reset_cost()

        # Step 1: Cluster
        clustered = self._clusterer.cluster(failure_batch.events)

        # Step 2: Process each failure concurrently (teacher calls are IO-bound)
        tasks = [self._process_failure(f) for f in clustered]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        saved = []
        for result in results:
            if isinstance(result, Exception):
                log.exception("curation_failure_skipped", error=str(result))
                continue
            if result is None:
                continue

            # Step 6: Persist
            inserted = await repo.upsert(result)
            if inserted:
                saved.append(result)
                examples_curated_total.inc()

        teacher_model_cost_usd.set(self._teacher.total_cost_usd)
        try:
            from src.monitoring.cost_tracker import CostTracker
            await CostTracker().record_spend("teacher_model", self._teacher.total_cost_usd)
        except Exception:
            log.warning("cost_tracker_record_failed")
        log.info(
            "curation_complete",
            input_failures=len(failure_batch.events),
            saved_examples=len(saved),
            teacher_cost_usd=round(self._teacher.total_cost_usd, 4),
        )
        return saved

    async def _process_failure(self, failure: FailureEvent) -> dict | None:
        # Step 2: Teacher correction
        grounding_result = await self._teacher.generate_correction(failure)
        if grounding_result is None:
            examples_dropped_quality.inc()
            return None

        corrected = grounding_result.correction
        confidence = grounding_result.confidence

        # Step 3: PII scrub (fail-closed)
        scrubbed_prompt, scrubbed_completion, pii_ok = self._pii.scrub_example(
            failure.prompt, corrected
        )
        if not pii_ok:
            examples_dropped_pii.inc()
            return None

        # Step 4: Dedup
        dedup_hash = self._dedup.compute_hash(scrubbed_prompt, scrubbed_completion)
        if self._dedup.is_duplicate(scrubbed_prompt, scrubbed_completion):
            examples_dropped_dedup.inc()
            return None

        # Step 5: Quality filter
        passes, quality_score, reason = self._quality.passes(
            scrubbed_prompt, failure.completion, scrubbed_completion, confidence
        )
        if not passes:
            log.debug("quality_filter_rejected", reason=reason)
            examples_dropped_quality.inc()
            return None

        return {
            "llm_log_id": failure.llm_log_id if failure.llm_log_id != "unknown" else None,
            "prompt": scrubbed_prompt,
            "bad_completion": failure.completion[:4000],
            "corrected_completion": scrubbed_completion,
            "failure_type": failure.failure_type,
            "cluster_id": failure.metadata.get("cluster_id"),
            "teacher_model": "gpt-4o",
            "teacher_confidence": confidence,
            "pii_scrubbed": True,
            "dedup_hash": dedup_hash,
            "quality_score": quality_score,
            "grounding_score": grounding_result.grounding_score,
            "grounding_sources": grounding_result.grounding_sources or None,
        }
