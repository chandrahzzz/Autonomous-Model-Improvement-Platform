"""Training poller node: checks Modal job status (non-blocking)."""
import structlog
from datetime import datetime
from src.graph.state import PipelineState
from src.db.connection import get_db
from src.db.repositories.model_versions import ModelRepository
from src.training.job_poller import JobPoller
from src.monitoring.metrics import training_runs_total, training_loss_gauge

log = structlog.get_logger()
_poller = JobPoller()

async def training_poller_node(state: PipelineState) -> PipelineState:
    modal_job_id = state.get("modal_job_id")
    run_id = state.get("training_run_id")
    submitted_str = state.get("training_submitted_at", datetime.utcnow().isoformat())
    submitted_at = datetime.fromisoformat(submitted_str)

    if not modal_job_id:
        return {**state, "training_status": "failed"}

    status, result = await _poller.poll(modal_job_id, submitted_at)

    updates: dict = {"training_status": status}
    if status == "completed" and result:
        updates["final_loss"] = result.get("final_loss")
        updates["lora_weights_path"] = result.get("output_dir")
        training_loss_gauge.set(result.get("final_loss", 0.0))
        training_runs_total.labels(status="completed").inc()

        # Record estimated GPU spend (wall-clock since submission * hourly rate).
        try:
            from src.config.settings import settings
            from src.monitoring.cost_tracker import CostTracker
            hours = max(0.0, (datetime.utcnow() - submitted_at).total_seconds() / 3600.0)
            await CostTracker().record_spend("modal_gpu", hours * settings.modal_a100_hourly_usd)
        except Exception:
            log.warning("modal_cost_record_failed")

        async with get_db() as db:
            repo = ModelRepository(db)
            if run_id:
                await repo.update_training_run(run_id, {
                    "status": "completed",
                    "final_loss": result.get("final_loss"),
                    "dataset_uri": result.get("dataset_uri"),
                    "wandb_run_id": result.get("wandb_run_id"),
                    "wandb_run_url": result.get("wandb_run_url"),
                    "completed_at": datetime.utcnow(),
                })
            await repo.create_version({
                "version_tag": state.get("version_tag", "v8"),
                "base_model": state.get("base_model", "meta-llama/Meta-Llama-3-8B-Instruct"),
                "lora_weights_path": result.get("output_dir"),
                "training_run_id": run_id,
            })
    elif status in ("failed", "timeout"):
        training_runs_total.labels(status="failed").inc()
        async with get_db() as db:
            repo = ModelRepository(db)
            if run_id:
                await repo.update_training_run(run_id, {"status": "failed", "completed_at": datetime.utcnow()})

    log.info("training_poller_node_complete", status=status)
    return {**state, **updates}
