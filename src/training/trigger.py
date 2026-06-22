"""
Training trigger: fires when conditions are met:
  1. Pending examples ≥ dataset_size threshold
  2. Drift score ≥ drift threshold — a HARD gate for hallucination/semantic_drift,
     but SOFT (exempted) when the dominant pending failure type is a format or
     refusal regression, which don't move the embedding distribution (#T3)
  3. Min interval since last training run has elapsed
"""

from datetime import datetime, timedelta
import structlog

from src.config.settings import settings

log = structlog.get_logger()


class TrainingTrigger:
    def should_trigger(
        self,
        pending_examples: int,
        drift_score: float,
        last_training_at: datetime | None,
        failure_type_counts: dict[str, int] | None = None,
    ) -> tuple[bool, str]:
        """
        Returns (should_trigger, reason).

        ``failure_type_counts`` is the {failure_type: count} distribution of the
        pending examples. When the dominant type is drift-exempt (format/refusal),
        the drift gate is skipped so a clear structural regression can still
        trigger training without ever crossing the Mahalanobis threshold.
        """
        # Condition 1: enough data
        if pending_examples < settings.training_trigger_dataset_size:
            return False, (
                f"insufficient_data: {pending_examples} < "
                f"{settings.training_trigger_dataset_size}"
            )

        # Condition 2: drift detected — unless the dominant failure type is exempt.
        dominant = None
        if failure_type_counts:
            dominant = max(failure_type_counts, key=failure_type_counts.get)
        drift_exempt = dominant in settings.trigger_drift_exempt_failure_types

        if drift_exempt:
            from src.monitoring.metrics import training_trigger_drift_exempt_total
            training_trigger_drift_exempt_total.inc()
            log.info("training_trigger_drift_exempt", dominant_failure_type=dominant)
        elif drift_score < settings.training_trigger_drift_threshold:
            return False, (
                f"drift_below_threshold: {drift_score:.3f} < "
                f"{settings.training_trigger_drift_threshold}"
            )

        # Condition 3: cooldown period
        if last_training_at is not None:
            elapsed = datetime.utcnow() - last_training_at
            min_interval = timedelta(hours=settings.training_min_interval_hours)
            if elapsed < min_interval:
                remaining = min_interval - elapsed
                return False, f"cooldown: {remaining.seconds // 60}m remaining"

        drift_clause = (
            f"drift_exempt(dominant={dominant})" if drift_exempt
            else f"drift={drift_score:.3f}"
        )
        reason = (
            f"triggered: examples={pending_examples}, "
            f"{drift_clause}, cooldown=cleared"
        )
        log.info("training_trigger_fired", reason=reason)
        return True, reason
