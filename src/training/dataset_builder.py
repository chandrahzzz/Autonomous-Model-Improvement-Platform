"""
Builds a JSONL training dataset from the training_examples table.

Two correctness concerns are handled here:
  - Chat template (S8): instruction-tuned base models (Llama 3, Mistral, etc.)
    expect their native chat template with system/user/assistant roles. Training
    on raw Alpaca "### Instruction/### Response" text fights the base model's
    instruction-following format. We format with the tokenizer's chat template,
    falling back to a Llama-3-style template string when the tokenizer can't be
    loaded locally (e.g. a gated model with no HF token on the build host).
  - Replay buffer (S11): training only on corrected failures is 100% negative
    examples and risks catastrophic forgetting. We mix in known-good production
    examples so the model also reinforces what it already does well.
"""

import json
import os
import random
import tempfile
import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.config.settings import settings
from src.db.repositories.training_examples import TrainingExampleRepository
from src.db.repositories.model_versions import ModelRepository
from src.db.repositories.llm_logs import LLMLogRepository
from src.monitoring.metrics import replay_buffer_ratio

log = structlog.get_logger()

DEFAULT_SYSTEM_PROMPT = "You are a helpful, accurate assistant."

# Fallback chat template used when the real tokenizer can't be loaded on the
# build host (Llama 3 is gated). Mirrors the Llama 3 instruct format.
_LLAMA3_FALLBACK = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n{assistant}<|eot_id|>"
)


class DatasetBuilder:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db
        self._example_repo = TrainingExampleRepository(db)
        self._model_repo = ModelRepository(db)
        self._log_repo = LLMLogRepository(db)
        self._tokenizer = self._load_tokenizer()
        # Populated by build(): {model_version: count} of the replay examples used,
        # so the training run can audit which versions its replay buffer came from.
        self.last_replay_distribution: dict[str, int] = {}

    @staticmethod
    def _recency_weighted_sample(candidates: list, n: int, decay: float) -> list:
        """Sample up to ``n`` items without replacement, weighting newer items
        (earlier in the newest-first list) higher via exponential decay over rank.
        decay<1 favours recent logs; decay==1 is uniform."""
        if n <= 0 or not candidates:
            return []
        if n >= len(candidates):
            return list(candidates)
        weights = [decay ** rank for rank in range(len(candidates))]
        pool = list(zip(candidates, weights))
        chosen = []
        for _ in range(n):
            total = sum(w for _, w in pool)
            if total <= 0:
                break
            r = random.uniform(0, total)
            acc = 0.0
            for idx, (item, w) in enumerate(pool):
                acc += w
                if acc >= r:
                    chosen.append(item)
                    pool.pop(idx)
                    break
        return chosen

    def _load_tokenizer(self):
        """Load the base model's tokenizer for chat-template formatting.
        Returns None if unavailable (gated/offline) — caller uses the fallback."""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(settings.base_model_name)
        except Exception as e:
            log.warning(
                "tokenizer_load_failed_using_fallback_template",
                base_model=settings.base_model_name, error=str(e),
            )
            return None

    def _format_example(self, prompt: str, response: str) -> str:
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        if self._tokenizer is not None:
            try:
                return self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
            except Exception:
                log.warning("apply_chat_template_failed_using_fallback")
        return _LLAMA3_FALLBACK.format(
            system=DEFAULT_SYSTEM_PROMPT, user=prompt, assistant=response
        )

    async def build(
        self, run_id: int, limit: int = 2000, replay_ratio: float | None = None
    ) -> tuple[str, int]:
        """
        Writes JSONL to a temp file. Returns (file_path, n_examples).
        Marks the failure examples as belonging to run_id and mixes in a replay
        buffer of known-good examples to counter catastrophic forgetting.
        """
        if replay_ratio is None:
            replay_ratio = settings.replay_ratio

        examples = await self._example_repo.get_pending(limit=limit)
        if not examples:
            raise ValueError("No pending training examples available")

        records: list[dict] = []
        ids: list[str] = []
        for ex in examples:
            records.append({
                "text": self._format_example(ex.prompt, ex.corrected_completion),
                "source": "failure",
                "failure_type": ex.failure_type,
            })
            ids.append(str(ex.id))

        # Replay buffer: known-good production examples reinforce existing skills.
        n_failure = len(records)
        n_replay = int(n_failure * replay_ratio / (1 - replay_ratio)) if replay_ratio < 1 else 0
        replay_added = 0
        self.last_replay_distribution = {}
        if n_replay > 0:
            prod = await self._model_repo.get_production_version()
            prod_version = prod.version_tag if prod else None
            # Oversample a newest-first candidate pool, then recency-weight the
            # draw so older (weaker-model) logs contribute less (#T2).
            pool_size = n_replay * max(1, settings.replay_candidate_pool_multiplier)
            candidates = await self._log_repo.get_known_good_candidates(
                limit=pool_size, model_version=prod_version
            )
            good = self._recency_weighted_sample(
                candidates, n_replay, settings.replay_recency_decay
            )
            for row in good:
                records.append({
                    "text": self._format_example(row.prompt, row.completion),
                    "source": "replay",
                    "failure_type": "none",
                })
                ver = getattr(row, "model_version", None) or "unknown"
                self.last_replay_distribution[ver] = (
                    self.last_replay_distribution.get(ver, 0) + 1
                )
            replay_added = len(good)

        random.shuffle(records)

        fd, path = tempfile.mkstemp(suffix=".jsonl", prefix="train_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for item in records:
                    f.write(json.dumps(item) + "\n")
        except Exception:
            os.unlink(path)
            raise

        # Only the curated failure examples are "consumed" by the run.
        await self._example_repo.mark_used(ids, run_id)

        total = len(records)
        actual_ratio = (replay_added / total) if total else 0.0
        replay_buffer_ratio.set(actual_ratio)
        log.info(
            "dataset_built",
            run_id=run_id, n_examples=total, n_failure=n_failure,
            n_replay=replay_added, replay_ratio=round(actual_ratio, 3),
            chat_template="tokenizer" if self._tokenizer else "fallback", path=path,
        )
        return path, total
