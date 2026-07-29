"""
Modal Labs GPU worker for Unsloth LoRA fine-tuning.

The training job runs serverlessly on Modal — no GPU infrastructure to manage.
The LangGraph pipeline submits the job and polls for completion asynchronously;
the main graph loop is never blocked waiting for training.

Modal function is defined here but executed remotely on an A100.
"""

import asyncio
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
    hf_hub_repo: str = "",       # push adapter here when set (free durable storage)
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

    # Also push the adapter (~20 MB) to HuggingFace Hub — free, unlimited, and
    # survives even if the Modal account/volume goes away. Best-effort: a Hub
    # outage must not fail an otherwise-successful training run.
    hf_repo_pushed = ""
    if hf_hub_repo and os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            repo_id = f"{hf_hub_repo}-{version_tag}"
            api = HfApi(token=os.environ["HF_TOKEN"])
            api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
            api.upload_folder(folder_path=output_dir, repo_id=repo_id)
            hf_repo_pushed = repo_id
        except Exception as e:
            print(f"hf_hub_push_failed: {e}")  # remote container: stdout → Modal logs

    final_loss = float(trainer.state.log_history[-1].get("loss", 0.0))
    run.finish()

    return {
        "version_tag": version_tag,
        "output_dir": output_dir,
        "weights_uri": f"modal://{ARTIFACTS_VOLUME}/lora/{version_tag}",
        "hf_hub_repo": hf_repo_pushed,
        "dataset_uri": dataset_uri,
        "final_loss": final_loss,
        "wandb_run_id": run.id,
        "wandb_run_url": run.get_url() or "",
        "n_examples": len(records),
    }


def _dataset_remote_path(version_tag: str) -> str:
    return f"datasets/{version_tag}.jsonl"


def dataset_uri_for(version_tag: str) -> str:
    return f"modal://{ARTIFACTS_VOLUME}/{_dataset_remote_path(version_tag)}"


async def persist_dataset_to_volume(dataset_path: str, version_tag: str) -> str | None:
    """Upload the exact dataset to the durable Modal Volume BEFORE the training job
    runs (#T4). Previously the dataset was only written from inside train_lora, so
    a job that failed before that point left training_runs.dataset_uri pointing at
    a file that never existed. Returns the dataset_uri on a confirmed write, else
    None (caller decides whether to proceed)."""
    def _upload() -> str:
        with artifacts_volume.batch_upload(force=True) as batch:
            batch.put_file(dataset_path, _dataset_remote_path(version_tag))
        return dataset_uri_for(version_tag)

    try:
        loop = asyncio.get_running_loop()
        uri = await loop.run_in_executor(None, _upload)
        log.info("dataset_persisted_to_volume", version_tag=version_tag, uri=uri)
        return uri
    except Exception as e:
        log.error("dataset_volume_upload_failed", version_tag=version_tag, error=str(e))
        return None


async def dataset_uri_resolvable(version_tag: str) -> bool:
    """Pre-training health check: confirm the dataset file actually exists on the
    Volume before the (expensive) GPU job is allowed to run."""
    def _exists() -> bool:
        remote = _dataset_remote_path(version_tag)
        for entry in artifacts_volume.listdir("datasets"):
            if getattr(entry, "path", "").endswith(remote) or getattr(entry, "path", "") == remote:
                return True
        return False

    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _exists)
    except Exception as e:
        log.warning("dataset_uri_resolvable_check_failed", version_tag=version_tag, error=str(e))
        return False


async def _trigger_colab_fallback(version_tag: str, lora_config: LoRAConfig) -> None:
    """Free-tier escape hatch: when Modal can't take the job (credits exhausted,
    auth failure), ping the Colab webhook — an ngrok endpoint on a free T4
    running scripts/colab_fallback_trainer.py. The dataset is already durable
    (persist_dataset_to_volume ran pre-submit, and scripts/reproduce_dataset.py
    can re-fetch it), so the Colab side only needs the version tag + config.
    With no webhook configured this logs loudly and returns."""
    if not settings.colab_webhook_url:
        log.error(
            "modal_submit_failed_no_colab_fallback",
            version_tag=version_tag,
            hint=(
                "Set COLAB_WEBHOOK_URL to an ngrok endpoint running "
                "scripts/colab_fallback_trainer.py, or top up Modal credits."
            ),
        )
        return
    try:
        import httpx
        payload = {
            "version_tag": version_tag,
            "dataset_uri": dataset_uri_for(version_tag),
            "lora_config": lora_config.model_dump(),
            "hf_hub_repo": settings.hf_hub_repo_prefix,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(settings.colab_webhook_url, json=payload)
            resp.raise_for_status()
        log.info("colab_fallback_triggered", version_tag=version_tag)
    except Exception as e:
        log.error("colab_fallback_webhook_failed", version_tag=version_tag, error=str(e))


def _hub_repo_id(version_tag: str) -> str:
    """Hub repo the Colab trainer pushes the adapter to (matches train_lora's
    `{hf_hub_repo}-{version_tag}` convention)."""
    return f"{settings.hf_hub_repo_prefix}-{version_tag}"


async def _submit_colab_job(
    dataset_path: str, lora_config: LoRAConfig, version_tag: str
) -> str:
    """Colab-PRIMARY submit (no Modal). The laptop is behind NAT, so instead of a
    callback we inline the dataset into the webhook POST, let Colab train on a
    free T4 and push the adapter to the Hub, and later POLL the Hub for it (see
    get_job_result). Returns a synthetic `colab:<version_tag>` job id.

    Raises if the webhook is unset or unreachable — a training run that can't be
    submitted must be marked failed (the graph's rollback path handles it), not
    silently swallowed.
    """
    if not settings.colab_webhook_url:
        raise RuntimeError(
            "training_backend=colab but COLAB_WEBHOOK_URL is unset. Run "
            "scripts/colab_fallback_trainer.py --serve on a Colab T4, expose it "
            "with ngrok, and set COLAB_WEBHOOK_URL=<ngrok-url>/train."
        )
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_jsonl = f.read()

    import httpx
    payload = {
        "version_tag": version_tag,
        "lora_config": lora_config.model_dump(),
        "hf_hub_repo": settings.hf_hub_repo_prefix,
        "dataset_jsonl": dataset_jsonl,   # inlined — Colab can't fetch modal:// URIs
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(settings.colab_webhook_url, json=payload)
        resp.raise_for_status()
    log.info(
        "colab_primary_job_submitted",
        version_tag=version_tag, target_repo=_hub_repo_id(version_tag),
    )
    return f"colab:{version_tag}"


async def _poll_hf_hub_adapter(version_tag: str) -> dict | None:
    """Poll the Hub for the Colab-trained adapter. Returns a result dict shaped
    like train_lora's (so training_poller_node needs no changes — it reads
    `output_dir`, which we set to the Hub repo id that HFModelRunner can load),
    or None while the adapter hasn't been pushed yet."""
    repo_id = _hub_repo_id(version_tag)

    def _check() -> dict | None:
        try:
            from huggingface_hub import HfApi
        except Exception:
            log.error("huggingface_hub_not_installed_for_colab_poll")
            return None
        try:
            api = HfApi(token=settings.hf_token or None)
            files = api.list_repo_files(repo_id=repo_id)
        except Exception:
            # Repo not created yet (RepositoryNotFoundError) ⇒ still training.
            return None
        adapter_present = any(
            f.endswith(("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"))
            for f in files
        )
        if not adapter_present:
            return None
        return {
            "version_tag": version_tag,
            "output_dir": repo_id,        # HFModelRunner loads a Hub repo id directly
            "hf_hub_repo": repo_id,
            "final_loss": 0.0,            # not reported over the Hub-poll channel
            "dataset_uri": None,
            "wandb_run_id": None,
            "wandb_run_url": "",
            "n_examples": 0,
        }

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _check)


async def submit_training_job(
    dataset_path: str,
    lora_config: LoRAConfig,
    version_tag: str,
) -> str:
    """
    Submit a training job. Returns a job id for polling. Non-blocking.

    Backends (settings.training_backend):
      - "colab": Colab-primary (no Modal). Inlines the dataset into the webhook,
        returns "colab:<tag>"; get_job_result polls the Hub for the adapter.
      - "modal": Modal A100 primary. If the spawn fails (exhausted free credits,
        auth error, outage), the Colab webhook fallback is pinged and the error
        re-raised so the run is marked failed (graph rollback path handles it).
    """
    if settings.training_backend == "colab":
        return await _submit_colab_job(dataset_path, lora_config, version_tag)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_content = f.read()

    try:
        # Spawn remote job (non-blocking — Modal handles queuing)
        call = train_lora.spawn(
            dataset_jsonl=dataset_content,
            lora_config_dict=lora_config.model_dump(),
            version_tag=version_tag,
            wandb_project=settings.wandb_project,
            hf_hub_repo=settings.hf_hub_repo_prefix,
        )
    except Exception as e:
        log.error("modal_submit_failed", version_tag=version_tag, error=str(e))
        await _trigger_colab_fallback(version_tag, lora_config)
        raise

    log.info("modal_job_submitted", version_tag=version_tag, call_id=call.object_id)
    return call.object_id


async def get_job_result(call_id: str) -> dict | None:
    """
    Poll a training job. Returns result dict if done, None if still running.
    Raises on failure.

    Colab-primary jobs carry a "colab:<version_tag>" id and are polled against
    the Hub; everything else is a Modal FunctionCall id.
    """
    if call_id.startswith("colab:"):
        version_tag = call_id.split(":", 1)[1]
        return await _poll_hf_hub_adapter(version_tag)

    try:
        fc = modal.functions.FunctionCall.from_id(call_id)
        result = fc.get(timeout=0)   # non-blocking poll
        return result
    except TimeoutError:
        return None
    except Exception as e:
        log.error("modal_job_failed", call_id=call_id, error=str(e))
        raise
