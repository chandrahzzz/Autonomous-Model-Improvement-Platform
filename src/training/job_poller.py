"""
Async poller for Modal training job status.
Checks job every 5 minutes (300s). Non-blocking — used inside LangGraph
training_poller node which re-enters the wait loop via graph edges.
"""

import structlog
from datetime import datetime, timedelta

from src.training.modal_worker import get_job_result

log = structlog.get_logger()

POLL_INTERVAL_SECONDS = 300
MAX_WAIT_HOURS = 3


class JobPoller:
    async def poll(
        self,
        modal_job_id: str,
        submitted_at: datetime,
    ) -> tuple[str, dict | None]:
        """
        Returns (status, result).
        status: 'running' | 'completed' | 'failed' | 'timeout'
        result: Modal result dict if completed, else None.
        """
        elapsed = datetime.utcnow() - submitted_at
        if elapsed > timedelta(hours=MAX_WAIT_HOURS):
            log.error("training_job_timeout", modal_job_id=modal_job_id, elapsed_h=elapsed.seconds // 3600)
            return "timeout", None

        try:
            result = await get_job_result(modal_job_id)
            if result is None:
                log.debug("training_job_still_running", modal_job_id=modal_job_id)
                return "running", None

            log.info(
                "training_job_completed",
                modal_job_id=modal_job_id,
                final_loss=result.get("final_loss"),
            )
            return "completed", result

        except Exception as e:
            log.error("training_job_poll_error", modal_job_id=modal_job_id, error=str(e))
            return "failed", None
