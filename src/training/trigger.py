"""
Training trigger: fires when ALL conditions are met:
  1. Pending examples ≥ dataset_size threshold
  2. Drift score ≥ drift threshold (at least one failure signal)
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
    ) -> tuple[bool, str]:
        """
        Returns (should_trigger, reason).
        All three conditions must pass.
        """
        # Condition 1: enough data
        if pending_examples < settings.training_trigger_dataset_size:
            return False, (
                f"insufficient_data: {pending_examples} < "
                f"{settings.training_trigger_dataset_size}"
            )

        # Condition 2: drift detected
        if drift_score < settings.training_trigger_drift_threshold:
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

        reason = (
            f"triggered: examples={pending_examples}, "
            f"drift={drift_score:.3f}, "
            f"cooldown=cleared"
        )
        log.info("training_trigger_fired", reason=reason)
        return True, reason
