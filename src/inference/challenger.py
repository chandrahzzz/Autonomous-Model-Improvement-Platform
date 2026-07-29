"""
Challenger model inference for evaluation (#T1).

The training job saves ONLY the LoRA adapter (~20 MB). Before this module, the
eval node used a hardcoded stub, so promotion gates could (and in dev did) score
a model that was never the trained one. This module loads the base model, applies
the trained adapter via PEFT's ``merge_and_unload()``, and exposes an async invoke
function — plus a hard verification that the merged model's output actually
DIFFERS from the base model (i.e. the adapter was really applied and is not a
no-op), which the eval node runs before trusting any score.

Heavy ML imports are local so importing this module (and the eval node) stays
cheap; the model is loaded lazily on first invoke. Real model loading needs a GPU
and the (gated) base weights, so it is exercised in production, not CI — the unit
tests cover the verification contract with injected fns.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from typing import Awaitable, Callable

import structlog

from src.config.settings import settings

log = structlog.get_logger()

InvokeFn = Callable[[str], Awaitable[str]]

# Short, neutral probes used only to confirm the adapter changed the model's
# behaviour. They are not a quality measure — just "is this the base model?".
DEFAULT_PROBES = [
    "Summarize the purpose of a return policy in one sentence.",
    "What should a helpful assistant do when it doesn't know an answer?",
    "Rewrite this politely: 'go away'.",
]

DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate assistant."


class HFModelRunner:
    """Lazily loads a causal LM (optionally with a merged LoRA adapter) and
    generates deterministically. One instance per model variant."""

    def __init__(self, base_model: str, lora_weights_path: str | None = None) -> None:
        self._base_model = base_model
        self._lora_weights_path = lora_weights_path
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        log.info(
            "challenger_model_loading",
            base_model=self._base_model, adapter=self._lora_weights_path,
        )
        tokenizer = AutoTokenizer.from_pretrained(self._base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # 4-bit NF4 when a CUDA GPU is present (bitsandbytes is CUDA-only);
        # on CPU fall back to full precision — slower but functional, which is
        # what a $0 eval box (Oracle free-tier ARM, student laptop) needs.
        load_kwargs: dict = {"trust_remote_code": True}
        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                load_kwargs["device_map"] = "auto"
            except Exception:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32
        model = AutoModelForCausalLM.from_pretrained(self._base_model, **load_kwargs)
        if self._lora_weights_path:
            from peft import PeftModel
            # Accepts a local dir, a Modal-volume mount path, OR a HuggingFace
            # Hub repo id ("user/finetuning-adapter-v8") — from_pretrained
            # resolves all three, so Hub-stored adapters need no special casing.
            model = PeftModel.from_pretrained(model, self._lora_weights_path)
            # Bake the adapter into the base weights so generation reflects it and
            # there is no adapter-routing ambiguity at inference time.
            model = model.merge_and_unload()
        model.eval()
        self._model = model
        self._tokenizer = tokenizer

    def _generate_sync(self, prompt: str, max_new_tokens: int = 256) -> str:
        self._ensure_loaded()
        import torch

        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = self._tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            ).to(self._model.device)
        except Exception:
            input_ids = self._tokenizer(prompt, return_tensors="pt").input_ids.to(
                self._model.device
            )
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # deterministic — verification needs reproducibility
                pad_token_id=self._tokenizer.pad_token_id,
            )
        text = self._tokenizer.decode(
            out[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        return text.strip()

    async def invoke(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._generate_sync, prompt)


# Merged-model cache: shadow traffic and repeated eval calls must not reload
# (and re-merge) a multi-GB model per request. Bounded to the two variants a
# cycle actually needs (base + current challenger); a new challenger version
# evicts the oldest entry.
_RUNNER_CACHE: OrderedDict[tuple[str, str | None], HFModelRunner] = OrderedDict()
_RUNNER_CACHE_MAX = 2


def get_runner(base_model: str, lora_weights_path: str | None = None) -> HFModelRunner:
    """LRU-cached HFModelRunner per (base, adapter) pair."""
    key = (base_model, lora_weights_path)
    runner = _RUNNER_CACHE.get(key)
    if runner is None:
        runner = HFModelRunner(base_model, lora_weights_path=lora_weights_path)
        _RUNNER_CACHE[key] = runner
        while len(_RUNNER_CACHE) > _RUNNER_CACHE_MAX:
            evicted_key, _ = _RUNNER_CACHE.popitem(last=False)
            log.info("challenger_runner_evicted", key=str(evicted_key))
    else:
        _RUNNER_CACHE.move_to_end(key)
    return runner


def build_base_invoke_fn(base_model: str) -> InvokeFn:
    return get_runner(base_model, lora_weights_path=None).invoke


def build_challenger_invoke_fn(base_model: str, lora_weights_path: str) -> InvokeFn:
    return get_runner(base_model, lora_weights_path=lora_weights_path).invoke


async def verify_adapter_distinct(
    challenger_fn: InvokeFn,
    base_fn: InvokeFn,
    probes: list[str] | None = None,
) -> tuple[bool, dict]:
    """Return (distinct, details). ``distinct`` is True iff the challenger differs
    from the base model on at least one probe — i.e. the LoRA adapter was applied
    and is not a no-op. If every probe is identical, the "challenger" is
    effectively the base model and its eval scores must NOT be trusted.
    """
    probes = probes or DEFAULT_PROBES[: settings.adapter_verification_probes]
    differing = 0
    compared = 0
    for probe in probes:
        try:
            chal, base = await asyncio.gather(challenger_fn(probe), base_fn(probe))
        except Exception:
            log.exception("adapter_verification_probe_failed")
            continue
        compared += 1
        if (chal or "").strip() != (base or "").strip():
            differing += 1
    distinct = differing > 0
    details = {"probes": len(probes), "compared": compared, "differing": differing}
    log.info("adapter_verification", distinct=distinct, **details)
    return distinct, details
