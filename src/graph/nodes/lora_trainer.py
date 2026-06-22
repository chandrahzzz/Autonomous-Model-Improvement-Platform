"""LoRA trainer node: builds dataset and submits Modal training job."""
import structlog
from datetime import datetime
from src.config.settings import settings
from src.graph.state import PipelineState
from src.db.connection import get_db
from src.db.repositories.model_versions import ModelRepository
from src.training.lora_config import LoRAConfig
from src.training.dataset_builder import DatasetBuilder
from src.training.modal_worker import (
    submit_training_job, persist_dataset_to_volume, dataset_uri_resolvable,
)
from src.monitoring.metrics import training_runs_total

log = structlog.get_logger()

async def lora_trainer_node(state: PipelineState) -> PipelineState:
    async with get_db() as db:
        repo = ModelRepository(db)
        prod = await repo.get_production_version()
        current_version = prod.version_tag if prod else "v7"

        # Determine next version tag
        try:
            n = int(current_version.lstrip("v")) + 1
        except (ValueError, AttributeError):
            n = 8
        version_tag = f"v{n}"

        lora_config = LoRAConfig()
        run_data = {
            "version_tag": version_tag,
            "status": "submitted",
            "dataset_size": state.get("pending_examples", 0),
            "lora_config": lora_config.model_dump(),
        }
        run = await repo.create_training_run(run_data)

        builder = DatasetBuilder(db)
        dataset_path, n_examples = await builder.build(run.id)
        replay_distribution = builder.last_replay_distribution

    # Persist the exact dataset to the durable Volume BEFORE the job runs, and set
    # dataset_uri only once that write is confirmed (#T4). In production, refuse to
    # submit if the artifact isn't resolvable — a failed job must never leave a
    # dangling dataset_uri that reproduce_dataset.py 404s on.
    dataset_uri = await persist_dataset_to_volume(dataset_path, version_tag)
    if dataset_uri is not None:
        dataset_uri = dataset_uri if await dataset_uri_resolvable(version_tag) else None
    if dataset_uri is None and settings.environment == "production":
        async with get_db() as db:
            await ModelRepository(db).update_training_run(run.id, {
                "status": "failed",
                "error_message": "dataset_uri_unresolvable_preflight",
                "completed_at": datetime.utcnow(),
            })
        training_runs_total.labels(status="failed").inc()
        log.error("training_aborted_dataset_not_persisted", version_tag=version_tag)
        return {
            **state,
            "training_triggered": False,
            "training_status": "failed",
            "rollback_reason": "dataset_uri_unresolvable_preflight",
        }

    modal_job_id = await submit_training_job(dataset_path, lora_config, version_tag)

    async with get_db() as db:
        repo = ModelRepository(db)
        await repo.update_training_run(run.id, {
            "modal_job_id": modal_job_id,
            "dataset_size": n_examples,
            "dataset_path": dataset_path,
            "dataset_uri": dataset_uri,
            "replay_distribution": replay_distribution,
            "started_at": datetime.utcnow(),
        })

    training_runs_total.labels(status="submitted").inc()
    log.info("lora_trainer_node_complete", version_tag=version_tag, modal_job_id=modal_job_id)
    return {
        **state,
        "training_run_id": run.id,
        "modal_job_id": modal_job_id,
        "training_submitted_at": datetime.utcnow().isoformat(),
        "version_tag": version_tag,
        "training_status": "submitted",
    }
