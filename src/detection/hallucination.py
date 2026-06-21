"""
Hallucination detection via NLI (natural language inference) entailment.

For each (premise, completion) pair we ask an NLI model whether the completion
is *entailed* by the premise. A completion whose sentences are not entailed is a
likely hallucination. This is the right framing for factual grounding — unlike a
passage-retrieval cross-encoder, which only scores topical relevance and would
rate "Capital of France? -> London" as low-risk because it is on-topic.

Model: cross-encoder/nli-deberta-v3-base (3-class: contradiction/entailment/neutral)

GROUNDING: `llm_logs.retrieved_context` (populated from the LLM event) is used
as the NLI premise when present, giving true factual-consistency checking for
RAG traffic. For non-RAG calls with no context, the premise falls back to the
prompt, which only catches completions that contradict the question itself.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import structlog
from sentence_transformers import CrossEncoder

from src.config.settings import settings

log = structlog.get_logger()

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
BATCH_SIZE = 32


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class HallucinationDetector:
    def __init__(self) -> None:
        self._encoder = CrossEncoder(NLI_MODEL)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nli")
        # Resolve the entailment class index robustly (label order is
        # model-specific; don't hardcode index 1).
        id2label = getattr(
            getattr(self._encoder, "model", None), "config", None
        )
        id2label = getattr(id2label, "id2label", None) or {
            0: "contradiction", 1: "entailment", 2: "neutral"
        }
        self._entail_idx = next(
            (int(i) for i, lab in id2label.items() if "entail" in str(lab).lower()), 1
        )

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        text = (text or "").strip()
        if not text:
            return []
        try:
            import nltk
            return nltk.sent_tokenize(text)
        except Exception:
            return [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]

    async def score_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """
        Score (premise, completion) pairs. Returns hallucination score [0,1]
        per pair (0 = fully entailed/grounded, 1 = not entailed/likely hallucinated).
        Runs in a thread pool to avoid blocking the event loop.
        """
        if not pairs:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._run_nli, pairs)

    def _run_nli(self, pairs: list[tuple[str, str]]) -> list[float]:
        # Flatten to (premise, sentence) pairs so we can run a single batched
        # predict, then average entailment per original pair.
        flat: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for premise, completion in pairs:
            sentences = self._split_sentences(completion)
            start = len(flat)
            flat.extend((premise, s) for s in sentences)
            spans.append((start, len(flat)))

        if not flat:
            return [0.0] * len(pairs)

        logits = np.asarray(self._encoder.predict(flat, batch_size=BATCH_SIZE))
        if logits.ndim == 1:  # degenerate single-pair shape guard
            logits = logits.reshape(1, -1)
        entail = _softmax(logits)[:, self._entail_idx]

        scores: list[float] = []
        for start, end in spans:
            if end <= start:
                scores.append(0.0)
            else:
                scores.append(float(1.0 - entail[start:end].mean()))
        return scores

    async def is_hallucination(self, context: str, completion: str) -> tuple[bool, float]:
        scores = await self.score_batch([(context, completion)])
        score = scores[0]
        return score > settings.hallucination_threshold, score
