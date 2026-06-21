"""
Failure attribution orchestrator (RFC-003).

Given a FailureEvent, fetches the training examples used to build the current
production model, scores each example's influence on the failure, and persists
the top-K to failure_attributions. Called fire-and-forget from
failure_detector_node — never on the cycle critical path. Fully guarded:
attribution failures must never crash the pipeline.
"""

import asyncio
import time

import structlog

from src.config.settings import settings
from src.attribution.influence import InfluenceBackend, get_default_backend
from src.db.repositories.model_versions import ModelRepository
from src.db.repositories.training_examples import TrainingExampleRepository
from src.db.repositories.attribution import FailureAttributionRepository
from src.monitoring.metrics import (
    attribution_computations_total,
    attribution_latency_ms,
    attribution_candidates_scored,
)

log = structlog.get_logger()


class FailureAttributor:
    def __init__(self, backend: InfluenceBackend) -> None:
        self._backend = backend

    async def attribute(self, failure, db) -> bool:
        """Returns True if an attribution row was written, else False (skipped)."""
        try:
            # Real FailureEvent uses `llm_log_id`; tests use `log_id`. Accept either.
            log_id = getattr(failure, "log_id", None) or getattr(failure, "llm_log_id", None)

            model_repo = ModelRepository(db)
            production = await model_repo.get_production_version()
            if production is None:
                return False

            training_run_id = await model_repo.get_training_run_for_version(production.version_tag)
            if training_run_id is None:
                log.info("attribution_skipped_no_training_run", version=production.version_tag)
                return False

            te_repo = TrainingExampleRepository(db)
            candidates = await te_repo.get_used_for_run(
                training_run_id=training_run_id, limit=settings.attribution_max_candidates
            )
            if not candidates:
                return False

            failure_text = f"{failure.prompt} {failure.completion}"
            candidate_texts = [f"{c.prompt} {c.corrected_completion}" for c in candidates]

            start = time.monotonic()
            loop = asyncio.get_event_loop()
            scores = await loop.run_in_executor(
                None, self._backend.score, failure_text, candidate_texts
            )
            computation_ms = int((time.monotonic() - start) * 1000)

            k = settings.attribution_top_k
            indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            top_k = [
                {
                    "example_id": str(candidates[i].id),
                    "influence_score": round(float(scores[i]), 4),
                    "prompt_preview": candidates[i].prompt[:100],
                    "failure_type": candidates[i].failure_type,
                    "quality_score": candidates[i].quality_score,
                }
                for i in indices
                if scores[i] > 0.0
            ]
            if not top_k:
                return False

            classification_id = await self._lookup_classification_id(log_id, db)

            attr_repo = FailureAttributionRepository(db)
            await attr_repo.insert({
                "log_id": log_id,
                "failure_classification_id": classification_id,
                "model_version": production.version_tag,
                "training_run_id": training_run_id,
                "top_k_examples": top_k,
                "total_candidates_scored": len(candidates),
                "backend_used": self._backend.backend_name,
                "computation_ms": computation_ms,
            })

            attribution_computations_total.inc()
            attribution_latency_ms.observe(computation_ms)
            attribution_candidates_scored.observe(len(candidates))
            return True
        except Exception:
            log.exception("attribution_failed")
            return False

    async def _lookup_classification_id(self, log_id, db):
        try:
            from sqlalchemy import select
            from src.db.models import FailureClassification
            result = await db.execute(
                select(FailureClassification.id)
                .where(FailureClassification.llm_log_id == log_id)
                .order_by(FailureClassification.created_at.desc())
                .limit(1)
            )
            row = result.fetchone()
            return row[0] if row else None
        except Exception:
            return None


_attributor_instance: FailureAttributor | None = None


def get_attributor() -> FailureAttributor:
    global _attributor_instance
    if _attributor_instance is None:
        _attributor_instance = FailureAttributor(backend=get_default_backend())
    return _attributor_instance
