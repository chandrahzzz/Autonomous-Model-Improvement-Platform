"""
Curation pipeline orchestrator.

Pipeline for each failure event:
  1. Cluster (HDBSCAN) — group similar failures
  2. PII scrub prompt+completion (Presidio) — fail-closed, BEFORE the teacher so
     raw PII never reaches the OpenAI API (GDPR/HIPAA/DPA compliance)
  3. Teacher correction (GPT-4o) — generate ideal response from scrubbed inputs
  4. PII scrub the teacher output (defensive — grounding context may carry PII)
  5. Dedup (MinHash LSH) — skip near-duplicates
  6. Quality filter (ROUGE-L + confidence + poison check)
  7. Persist to training_examples table
"""

import asyncio
from dataclasses import replace

import structlog

from src.config.settings import settings
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
    examples_pre_scrubbed_total,
    examples_dropped_pre_scrub_total,
    teacher_model_cost_usd,
    dedup_index_size,
)

log = structlog.get_logger()


class CurationPipeline:
    def __init__(self) -> None:
        self._clusterer = FailureClusterer()
        self._teacher = TeacherModel()
        self._pii = PIIScrubber()
        self._dedup = Deduplicator()
        self._quality = QualityFilter()
        self._dedup_rehydrated = False

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

        # One-time: rehydrate the near-dup index from prior examples so a restart
        # doesn't let near-duplicates re-enter (the DB unique index only catches
        # EXACT dupes; 85%-Jaccard near-dupes would slip through).
        if settings.dedup_rehydrate_enabled and not self._dedup_rehydrated:
            await self._rehydrate_dedup(db)
            self._dedup_rehydrated = True

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
        # Step 2: PII scrub the inputs BEFORE the teacher sees them. The prompt
        # AND the bad completion are sent to OpenAI for correction, so both must
        # be scrubbed first — otherwise raw PII (names, SSNs, emails) leaves for
        # a third-party API. Fail-closed: drop before any API call on scrub error.
        scrubbed_prompt, scrubbed_bad, pii_ok = self._pii.scrub_example(
            failure.prompt, failure.completion
        )
        if not pii_ok:
            examples_dropped_pre_scrub_total.inc()
            examples_dropped_pii.inc()
            return None
        examples_pre_scrubbed_total.inc()

        # Hand the teacher only the scrubbed text (metadata/llm_log_id preserved so
        # grounding-context resolution still works).
        scrubbed_failure = replace(failure, prompt=scrubbed_prompt, completion=scrubbed_bad)

        # Step 3: Teacher correction (operates on scrubbed inputs)
        grounding_result = await self._teacher.generate_correction(scrubbed_failure)
        if grounding_result is None:
            examples_dropped_quality.inc()
            return None

        confidence = grounding_result.confidence

        # Step 4: Scrub the teacher OUTPUT too — defence in depth, since the
        # grounding context (which the teacher may quote) can itself contain PII.
        scrubbed_completion, _ = self._pii.scrub(grounding_result.correction)
        if grounding_result.correction and not scrubbed_completion:
            # scrub() returns "" only on failure (fail-closed); empty corrections
            # are already excluded by the quality filter below.
            examples_dropped_pii.inc()
            return None

        # Step 5: Dedup
        dedup_hash = self._dedup.compute_hash(scrubbed_prompt, scrubbed_completion)
        if self._dedup.is_duplicate(scrubbed_prompt, scrubbed_completion):
            examples_dropped_dedup.inc()
            return None

        # Step 6: Quality filter (compares against the scrubbed bad completion)
        passes, quality_score, reason = self._quality.passes(
            scrubbed_prompt, scrubbed_bad, scrubbed_completion, confidence
        )
        if not passes:
            log.debug("quality_filter_rejected", reason=reason)
            examples_dropped_quality.inc()
            return None

        return {
            "llm_log_id": failure.llm_log_id if failure.llm_log_id != "unknown" else None,
            "prompt": scrubbed_prompt,
            "bad_completion": scrubbed_bad[:4000],
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

    async def _rehydrate_dedup(self, db) -> None:
        """Reload the MinHash LSH index from existing training examples so
        near-duplicates can't slip back in after a process restart."""
        try:
            from src.db.repositories.training_examples import TrainingExampleRepository
            pairs = await TrainingExampleRepository(db).all_for_dedup(
                limit=settings.dedup_rehydrate_limit
            )
            loaded = self._dedup.preload(pairs)
            dedup_index_size.set(loaded)
            log.info("dedup_index_rehydrated", loaded=loaded)
        except Exception:
            log.warning("dedup_rehydrate_failed")
