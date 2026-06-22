"""
Refusal creep detector.

Two-stage: keyword pre-filter (fast), then semantic classifier (accurate).
Tracks a rolling baseline refusal rate and fires when current rate
exceeds baseline × refusal_rate_multiplier.
"""

import json
import re
from collections import deque
from typing import Any

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

    @property
    def window_size(self) -> int:
        return len(self._window)

    def is_creeping(self) -> bool:
        """Fire only once the window has enough samples — otherwise a couple of
        refusals in quiet traffic would read as a 100% refusal rate (#3)."""
        if len(self._window) < settings.refusal_min_samples:
            from src.monitoring.metrics import detector_insufficient_data_total
            detector_insufficient_data_total.labels(detector="refusal").inc()
            return False
        return self.current_refusal_rate > self._baseline_rate * settings.refusal_rate_multiplier

    def set_baseline_rate(self, rate: float) -> None:
        self._baseline_rate = rate

    # ── Rolling-window persistence (#4) ────────────────────────────────────────
    # Without this, a restart resets the refusal-rate window to empty, reading as
    # 0% for the first ~500 requests and defeating creep detection right after a
    # deploy. Persist the boolean window to Redis and rehydrate on startup.
    async def save_window_state(self, redis: Any) -> None:
        if not settings.detector_state_persist_enabled:
            return
        key = f"{settings.detector_state_redis_prefix}:refusal_window"
        try:
            await redis.set(key, json.dumps([int(b) for b in self._window]))
        except Exception:
            log.warning("refusal_window_persist_failed")

    async def load_window_state(self, redis: Any) -> None:
        if not settings.detector_state_persist_enabled:
            return
        key = f"{settings.detector_state_redis_prefix}:refusal_window"
        try:
            raw = await redis.get(key)
        except Exception:
            log.warning("refusal_window_rehydrate_failed")
            return
        if not raw:
            return
        try:
            values = json.loads(raw)
            self._window = deque((bool(v) for v in values), maxlen=WINDOW_SIZE)
            log.info("refusal_window_rehydrated", n=len(self._window))
        except (ValueError, TypeError):
            log.warning("refusal_window_rehydrate_parse_failed")
