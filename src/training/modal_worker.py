"""
Modal Labs GPU worker for Unsloth LoRA fine-tuning.

The training job runs serverlessly on Modal — no GPU infrastructure to manage.
The LangGraph pipeline submits the job and polls for completion asynchronously;
the main graph loop is never blocked waiting for training.

Modal function is defined here but executed remotely on an A100.
"""

import json
import os
import structlog

import modal

from src.training.lora_config import LoRAConfig
from src.config.settings import settings

log = structlog.get_logger()

# Modal app definition
stub = modal.App("continuous-finetuning-pipeline")

# Durable artifact storage. A Modal Volume persists across job containers, so
# LoRA adapters and the exact training dataset survive after the job exits
# (the old /tmp path was ephemeral and gone the moment the container died).
ARTIFACTS_DIR = "/artifacts"
ARTIFACTS_VOLUME = "finetuning-artifacts"
artifacts_volume = modal.Volume.from_name(ARTIFACTS_VOLUME, create_if_missing=True)

GPU_IMAGE = (
    modal.Image.debian_slim()
    .pip_install(
        "unsloth[colab-new]",
        "peft",
        "trl",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
        "wandb",
    )
)


@stub.function(
    image=GPU_IMAGE,
    gpu="A100",
    timeout=7200,         # 2h max
    volumes={ARTIFACTS_DIR: artifacts_volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def train_lora(
    dataset_jsonl: str,          # JSONL content (not path — transferred to Modal)
    lora_config_dict: dict,
    version_tag: str,
    wandb_project: str,
) -> dict:
    """
    Runs inside Modal on an A100. Returns training metadata dict.
    This function is called remotely via modal_worker.submit_training_job().
    """
    import wandb
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    import torch

    config = LoRAConfig(**lora_config_dict)

    # Load quantized base model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    peft_config = LoraConfig(
        r=config.r,
        lora_alpha=config.lora_alpha,
        target_modules=config.target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Build dataset from JSONL content
    records = [json.loads(line) for line in dataset_jsonl.strip().split("\n") if line]
    dataset = Dataset.from_list(records)

    # Archive the exact dataset to the durable volume for reproducibility.
    dataset_uri = f"modal://{ARTIFACTS_VOLUME}/datasets/{version_tag}.jsonl"
    os.makedirs(f"{ARTIFACTS_DIR}/datasets", exist_ok=True)
    with open(f"{ARTIFACTS_DIR}/datasets/{version_tag}.jsonl", "w", encoding="utf-8") as df:
        df.write(dataset_jsonl)

    # wandb tracking
    run = wandb.init(project=wandb_project, name=f"lora-{version_tag}", config=lora_config_dict)

    # Train. Save adapters to the durable volume, not ephemeral /tmp.
    output_dir = f"{ARTIFACTS_DIR}/lora/{version_tag}"
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        optim=config.optim,
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
    )
    trainer.train()

    # Save adapter to the durable volume and commit so it survives the container.
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    artifacts_volume.commit()

    final_loss = float(trainer.state.log_history[-1].get("loss", 0.0))
    run.finish()

    return {
        "version_tag": version_tag,
        "output_dir": output_dir,
        "weights_uri": f"modal://{ARTIFACTS_VOLUME}/lora/{version_tag}",
        "dataset_uri": dataset_uri,
        "final_loss": final_loss,
        "wandb_run_id": run.id,
        "wandb_run_url": run.get_url() or "",
        "n_examples": len(records),
    }


async def submit_training_job(
    dataset_path: str,
    lora_config: LoRAConfig,
    version_tag: str,
) -> str:
    """
    Submit training to Modal. Returns Modal call ID for polling.
    Non-blocking — returns immediately.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_content = f.read()

    # Spawn remote job (non-blocking — Modal handles queuing)
    call = train_lora.spawn(
        dataset_jsonl=dataset_content,
        lora_config_dict=lora_config.model_dump(),
        version_tag=version_tag,
        wandb_project=settings.wandb_project,
    )
    log.info("modal_job_submitted", version_tag=version_tag, call_id=call.object_id)
    return call.object_id


async def get_job_result(call_id: str) -> dict | None:
    """
    Poll a Modal job. Returns result dict if done, None if still running.
    Raises on failure.
    """
    try:
        fc = modal.functions.FunctionCall.from_id(call_id)
        result = fc.get(timeout=0)   # non-blocking poll
        return result
    except TimeoutError:
        return None
    except Exception as e:
        log.error("modal_job_failed", call_id=call_id, error=str(e))
        raise
