"""
Refusal creep detector.

Two-stage: keyword pre-filter (fast), then semantic classifier (accurate).
Tracks a rolling baseline refusal rate and fires when current rate
exceeds baseline × refusal_rate_multiplier.
"""

import re
from collections import deque

import structlog
from sentence_transformers import SentenceTransformer, util

from src.config.settings import settings

log = structlog.get_logger()

REFUSAL_KEYWORDS = [
    r"\bi can'?t\b", r"\bi won'?t\b", r"\bi'm unable\b", r"\bi am unable\b",
    r"\bi cannot\b", r"\bas an ai\b", r"\bi don'?t have the ability\b",
    r"\bI apologize, but\b", r"\bI'm not able to\b", r"\bI must decline\b",
    r"it would be inappropriate", r"that'?s not something i",
]

REFUSAL_PATTERN = re.compile("|".join(REFUSAL_KEYWORDS), re.IGNORECASE)

SEMANTIC_REFUSAL_EXEMPLARS = [
    "I'm sorry, I can't help with that.",
    "I'm not able to provide that information.",
    "That falls outside what I can assist with.",
    "I must decline this request.",
]

WINDOW_SIZE = 500


class RefusalDetector:
    def __init__(self) -> None:
        self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self._exemplar_embeddings = self._encoder.encode(
            SEMANTIC_REFUSAL_EXEMPLARS, convert_to_tensor=True
        )
        self._window: deque[bool] = deque(maxlen=WINDOW_SIZE)
        self._baseline_rate: float = 0.05   # 5% baseline refusal rate

    def is_refusal(self, text: str) -> tuple[bool, float]:
        """
        Returns (is_refusal, confidence_score).
        Uses keyword pre-filter then semantic similarity fallback.

        Every classification is recorded in the rolling window so
        `current_refusal_rate` / `is_creeping` reflect all calls, not only
        those routed through `classify_batch`.
        """
        if REFUSAL_PATTERN.search(text):
            result = (True, 1.0)
        else:
            embedding = self._encoder.encode(text, convert_to_tensor=True)
            sims = util.cos_sim(embedding, self._exemplar_embeddings)[0]
            max_sim = float(sims.max())
            result = (True, max_sim) if max_sim > 0.75 else (False, max_sim)

        self._window.append(result[0])
        return result

    async def classify_batch(self, log_events: list[dict]) -> list[tuple[str, float]]:
        """Returns (failure_type_or_none, score) per event."""
        results = []
        for event in log_events:
            is_ref, score = self.is_refusal(event.get("completion", ""))  # records in window
            if is_ref:
                results.append(("refusal_creep", score))
            else:
                results.append(("", score))
        return results

    @property
    def current_refusal_rate(self) -> float:
        if not self._window:
            return 0.0
        return sum(self._window) / len(self._window)

    def is_creeping(self) -> bool:
        return self.current_refusal_rate > self._baseline_rate * settings.refusal_rate_multiplier

    def set_baseline_rate(self, rate: float) -> None:
        self._baseline_rate = rate
