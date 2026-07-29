# Training backends

`settings.training_backend` selects where LoRA training runs.

| Backend | What happens | Needs |
|---|---|---|
| `modal` (default) | Modal A100 primary; Colab webhook as a fallback if the Modal submit fails | Modal token + secrets |
| `colab` | **Colab-primary, no Modal.** Dataset inlined into a webhook POST; Colab trains on a free T4 and pushes the adapter to the HF Hub; the pipeline polls the Hub | ngrok + a Colab notebook + HF token |

## Why polling (not a callback)

A laptop/dev box is behind NAT — Colab can't POST back to it. So the pipeline
**polls the HuggingFace Hub**: when the adapter repo `{HF_HUB_REPO_PREFIX}-{version_tag}`
appears with an `adapter_config.json`, the run is treated as complete. No inbound
connection to your machine is ever needed.

## Colab-primary flow

```
training triggers
  └─ submit_training_job (backend=colab)
       └─ POST {version_tag, lora_config, hf_hub_repo, dataset_jsonl}  → COLAB_WEBHOOK_URL
       └─ returns job id "colab:<version_tag>"   (never blocks, never silently fails)
Colab notebook (/train)
  └─ writes the inlined dataset, starts training in a BACKGROUND thread
  └─ returns {"status": "accepted"} immediately
  └─ ~20-30 min later: pushes adapter → https://huggingface.co/{prefix}-{version_tag}
pipeline (training_poller, each cycle)
  └─ get_job_result("colab:<tag>") → HfApi.list_repo_files({prefix}-{tag})
       └─ adapter present?  → result dict (output_dir = the Hub repo id) → model version created
       └─ not yet?          → None (still running); times out after COLAB_POLL_TIMEOUT_HOURS
  └─ eval → shadow → promote  (HFModelRunner loads the Hub repo id directly)
```

## `.env`

```dotenv
TRAINING_BACKEND=colab
COLAB_WEBHOOK_URL=          # fill after the notebook prints its ngrok URL (+ "/train")
HF_TOKEN=hf_...             # WRITE scope; Colab pushes with it, the pipeline polls with it
HF_HUB_REPO_PREFIX=youruser/finetuning-pipeline
COLAB_POLL_TIMEOUT_HOURS=3.0
```

## The Colab notebook (run these cells)

**Cell 1 — deps + files**
```python
!pip -q install flask pyngrok transformers peft trl datasets accelerate bitsandbytes huggingface_hub
# upload scripts/colab_fallback_trainer.py to the Colab session (Files pane), or:
!wget -q https://raw.githubusercontent.com/<youruser>/<repo>/main/scripts/colab_fallback_trainer.py
```

**Cell 2 — secrets** (Runtime → change type → T4 GPU first)
```python
import os
os.environ["HF_TOKEN"] = "hf_..."          # same WRITE token as .env
```

**Cell 3 — start the webhook + expose it**
```python
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTHTOKEN")
public = ngrok.connect(8787)
print("COLAB_WEBHOOK_URL =", public.public_url + "/train")   # paste into .env
!python colab_fallback_trainer.py --serve --port 8787
```

Put that printed URL into `COLAB_WEBHOOK_URL`, restart the pipeline runner, and
keep the Colab tab open. When training triggers, the notebook trains and pushes;
the pipeline picks the adapter up on its next poll.

## Verifying without a full run

The routing + Hub-poll plumbing is unit-tested in `tests/test_colab_backend.py`
(no GPU, no live Colab). To confirm the real training leg, run one `degrade`
traffic batch with the demo `.env` overrides so a training run triggers, then
watch the Colab tab train and `https://huggingface.co/{prefix}-{version_tag}`
appear. `GET /pipeline/status` should advance past `training` to eval/shadow.

## Manual mode (no webhook)

You can also train a specific version by hand and let the pipeline pick it up:
```bash
python colab_fallback_trainer.py --dataset dataset.jsonl \
    --version-tag v8 --hf-hub-repo youruser/finetuning-pipeline
```
The adapter lands at `youruser/finetuning-pipeline-v8`; the Hub poll finds it.
