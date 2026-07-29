"""
Standalone LoRA trainer for Google Colab (free T4) — the $0 fallback when
Modal's free credits run out.

Two ways to run it in a Colab notebook:

  A) Webhook mode (autonomous). The pipeline POSTs to your ngrok endpoint when
     a Modal submit fails; this script trains and pushes the adapter to the
     HuggingFace Hub.

        !pip -q install flask pyngrok transformers peft trl datasets accelerate \
            bitsandbytes huggingface_hub
        !python colab_fallback_trainer.py --serve --port 8787
        # then: ngrok http 8787  → set COLAB_WEBHOOK_URL=<ngrok-url>/train

  B) Manual mode. Download the dataset (scripts/reproduce_dataset.py, or any
     public URL / HF dataset), upload it to the Colab session, then:

        !python colab_fallback_trainer.py --dataset dataset.jsonl \
            --version-tag v8 --hf-hub-repo yourname/finetuning-adapter

The adapter lands at f"{hf_hub_repo}-{version_tag}" on the Hub. Register it in
the pipeline with a model_versions row pointing lora_weights_path at that repo
id — HFModelRunner loads Hub ids directly.

This file is intentionally dependency-light and self-contained: it must run in
a bare Colab runtime with no access to the pipeline's src/ tree.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_LORA = {
    "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "v_proj"],
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_train_batch_size": 2,   # T4 has 16 GB — half the A100 batch
    "gradient_accumulation_steps": 8,   # keeps effective batch at 16
    "max_seq_length": 2048,
}


def train(dataset_path: str, version_tag: str, hf_hub_repo: str, lora: dict) -> str:
    """Run LoRA SFT on the JSONL dataset and push the adapter to the Hub.
    Returns the Hub repo id the adapter was pushed to."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    cfg = {**DEFAULT_LORA, **lora}
    print(f"[colab-trainer] training {version_tag} on {cfg['base_model']}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        raise SystemExit("dataset is empty")
    dataset = Dataset.from_list(records)
    print(f"[colab-trainer] {len(records)} examples")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, device_map="auto",
        trust_remote_code=True,
    )
    model = get_peft_model(model, LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    output_dir = f"/content/lora-{version_tag}"
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir,
            num_train_epochs=cfg["num_train_epochs"],
            per_device_train_batch_size=cfg["per_device_train_batch_size"],
            gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
            learning_rate=cfg["learning_rate"],
            fp16=True,
            logging_steps=10,
            max_seq_length=cfg["max_seq_length"],
            dataset_text_field="text",
            report_to=[],
        ),
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    final_loss = float(trainer.state.log_history[-1].get("loss", 0.0))
    print(f"[colab-trainer] done, final_loss={final_loss:.4f}")

    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("[colab-trainer] HF_TOKEN unset — adapter stays local at", output_dir)
        return ""
    from huggingface_hub import HfApi
    repo_id = f"{hf_hub_repo}-{version_tag}"
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=output_dir, repo_id=repo_id)
    print(f"[colab-trainer] adapter pushed to https://huggingface.co/{repo_id}")
    return repo_id


def _download(url: str, dest: str) -> str:
    import urllib.request
    print(f"[colab-trainer] downloading {url}")
    urllib.request.urlretrieve(url, dest)
    return dest


def serve(port: int) -> None:
    """Webhook server for the Colab-PRIMARY backend (and the Modal fallback).

    Accepts POST /train with:
        {version_tag, lora_config, hf_hub_repo,
         dataset_jsonl  (inline JSONL — the colab-primary path), or
         dataset_url    (a public URL to fetch)}

    Training runs in a BACKGROUND thread and the handler returns immediately with
    {"status": "accepted"} — training takes ~20-30 min on a T4, far longer than
    any HTTP timeout. The pipeline does not wait on this response; it polls the
    HuggingFace Hub for the adapter to appear at f"{hf_hub_repo}-{version_tag}".
    Set HF_TOKEN in the Colab env so the push (and the pipeline's poll) works.
    """
    import threading

    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.post("/train")
    def _train():
        body = request.get_json(force=True)
        version_tag = body["version_tag"]
        hf_hub_repo = body.get("hf_hub_repo") or "colab-fallback"
        lora = body.get("lora_config") or {}
        dataset_jsonl = body.get("dataset_jsonl", "")
        dataset_url = body.get("dataset_url", "")
        dataset_path = f"dataset-{version_tag}.jsonl"

        if dataset_jsonl:
            with open(dataset_path, "w", encoding="utf-8") as f:
                f.write(dataset_jsonl)
        elif dataset_url.startswith("http"):
            _download(dataset_url, dataset_path)
        elif not os.path.exists(dataset_path):
            return jsonify({
                "status": "error",
                "message": "no dataset: pass dataset_jsonl, dataset_url, or upload one",
            }), 400

        def _bg():
            try:
                repo = train(dataset_path, version_tag, hf_hub_repo, lora)
                print(f"[colab-trainer] {version_tag} done → {repo}")
            except Exception as e:  # noqa: BLE001 - background worker, log only
                print(f"[colab-trainer] training FAILED for {version_tag}: {e}")

        threading.Thread(target=_bg, daemon=True).start()
        return jsonify({
            "status": "accepted",
            "version_tag": version_tag,
            "target_repo": f"{hf_hub_repo}-{version_tag}",
        })

    print(f"[colab-trainer] webhook listening on :{port}/train — expose with ngrok")
    app.run(host="0.0.0.0", port=port)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--serve", action="store_true", help="run as a webhook server")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--dataset", help="path or URL of the JSONL dataset")
    p.add_argument("--version-tag", default="colab")
    p.add_argument("--hf-hub-repo", default="colab-fallback",
                   help="Hub repo prefix; adapter goes to <prefix>-<version_tag>")
    p.add_argument("--lora-config", default="{}", help="JSON overrides for LoRA")
    args = p.parse_args()

    if args.serve:
        serve(args.port)
        return
    if not args.dataset:
        p.error("--dataset required unless --serve")
    dataset_path = args.dataset
    if dataset_path.startswith("http"):
        dataset_path = _download(dataset_path, "dataset.jsonl")
    train(dataset_path, args.version_tag, args.hf_hub_repo, json.loads(args.lora_config))


if __name__ == "__main__":
    sys.exit(main())
