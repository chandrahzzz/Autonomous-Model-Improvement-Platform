"""LoRA trainer node: builds dataset and submits Modal training job."""
import structlog
from datetime import datetime
from src.graph.state import PipelineState
from src.db.connection import get_db
from src.db.repositories.model_versions import ModelRepository
from src.training.lora_config import LoRAConfig
from src.training.dataset_builder import DatasetBuilder
from src.training.modal_worker import submit_training_job
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

    modal_job_id = await submit_training_job(dataset_path, lora_config, version_tag)

    async with get_db() as db:
        repo = ModelRepository(db)
        await repo.update_training_run(run.id, {
            "modal_job_id": modal_job_id,
            "dataset_size": n_examples,
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
