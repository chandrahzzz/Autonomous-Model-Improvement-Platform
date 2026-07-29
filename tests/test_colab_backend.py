"""
Tests for the Colab-primary training backend (no Modal, no GPU).

Covers the plumbing that can be verified without a GPU or a live Colab: the
submit routing, the inlined-dataset webhook POST, and the Hub-polling result
path. The actual T4 training run is confirmed manually on Colab (documented in
docs/TRAINING_BACKENDS.md).
"""

import os
import tempfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.config.settings import settings
from src.training import modal_worker
from src.training.lora_config import LoRAConfig


@pytest.fixture
def dataset_file():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write('{"text": "example one"}\n{"text": "example two"}\n')
    yield path
    os.unlink(path)


@pytest.fixture
def colab_settings(monkeypatch):
    monkeypatch.setattr(settings, "training_backend", "colab")
    monkeypatch.setattr(settings, "colab_webhook_url", "http://fake-ngrok/train")
    monkeypatch.setattr(settings, "hf_hub_repo_prefix", "acme/adapter")
    monkeypatch.setattr(settings, "hf_token", "hf_test")


# ── submit routing ────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_submit_routes_to_colab(monkeypatch, dataset_file):
    monkeypatch.setattr(settings, "training_backend", "colab")
    called = {}

    async def fake_colab(path, cfg, tag):
        called["args"] = (path, tag)
        return f"colab:{tag}"

    monkeypatch.setattr(modal_worker, "_submit_colab_job", fake_colab)
    job_id = await modal_worker.submit_training_job(dataset_file, LoRAConfig(), "v8")
    assert job_id == "colab:v8"
    assert called["args"] == (dataset_file, "v8")


@pytest.mark.asyncio
async def test_submit_colab_requires_webhook(monkeypatch, dataset_file):
    monkeypatch.setattr(settings, "training_backend", "colab")
    monkeypatch.setattr(settings, "colab_webhook_url", "")
    with pytest.raises(RuntimeError, match="COLAB_WEBHOOK_URL"):
        await modal_worker._submit_colab_job(dataset_file, LoRAConfig(), "v8")


@pytest.mark.asyncio
async def test_submit_colab_posts_inline_dataset(monkeypatch, dataset_file, colab_settings):
    captured = {}

    class FakeResp:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json=None):
            captured["url"] = url
            captured["json"] = json
            return FakeResp()

    # _submit_colab_job does `import httpx` locally, so patch the httpx module.
    import httpx
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    job_id = await modal_worker._submit_colab_job(dataset_file, LoRAConfig(), "v9")
    assert job_id == "colab:v9"
    assert captured["url"] == "http://fake-ngrok/train"
    body = captured["json"]
    assert body["version_tag"] == "v9"
    assert body["hf_hub_repo"] == "acme/adapter"
    assert "example one" in body["dataset_jsonl"]      # dataset was inlined
    assert "lora_config" in body


# ── Hub-poll result path ──────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_get_job_result_colab_running_when_repo_absent(monkeypatch, colab_settings):
    import huggingface_hub

    class FakeApi:
        def __init__(self, *a, **k):
            pass

        def list_repo_files(self, repo_id=None):
            raise Exception("RepositoryNotFoundError")  # not created yet

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    result = await modal_worker.get_job_result("colab:v8")
    assert result is None  # still training


@pytest.mark.asyncio
async def test_get_job_result_colab_completes_when_adapter_present(monkeypatch, colab_settings):
    import huggingface_hub

    class FakeApi:
        def __init__(self, *a, **k):
            pass

        def list_repo_files(self, repo_id=None):
            return ["adapter_config.json", "adapter_model.safetensors", "README.md"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    result = await modal_worker.get_job_result("colab:v8")
    assert result is not None
    # output_dir must be the Hub repo id so training_poller_node stores it as
    # lora_weights_path and HFModelRunner can load it directly.
    assert result["output_dir"] == "acme/adapter-v8"
    assert result["hf_hub_repo"] == "acme/adapter-v8"
    assert result["version_tag"] == "v8"


@pytest.mark.asyncio
async def test_get_job_result_colab_incomplete_repo_is_still_running(monkeypatch, colab_settings):
    import huggingface_hub

    class FakeApi:
        def __init__(self, *a, **k):
            pass

        def list_repo_files(self, repo_id=None):
            return ["README.md", ".gitattributes"]  # repo exists but no adapter yet

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    result = await modal_worker.get_job_result("colab:v8")
    assert result is None
