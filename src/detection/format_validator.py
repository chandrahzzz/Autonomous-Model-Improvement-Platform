"""
Format regression detector.

Two signals:
1. JSON schema validation — if outputs are expected to be JSON, validate them.
2. Length distribution KL divergence — if response lengths shift significantly
   from the baseline distribution, it indicates format regression.
"""

import json
import math
from collections import deque
from typing import Any

import numpy as np
import structlog

from src.config.settings import settings

log = structlog.get_logger()

WINDOW_SIZE = 500
LENGTH_BINS = list(range(0, 4000, 100))   # 100-token bins up to 4000


def _compute_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """KL(P || Q) — measures how P diverges from baseline Q."""
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


class FormatValidator:
    def __init__(self) -> None:
        self._length_window: deque[int] = deque(maxlen=WINDOW_SIZE)
        self._baseline_length_dist: np.ndarray | None = None
        self._expected_json: bool = False

    def set_baseline_lengths(self, lengths: list[int]) -> None:
        hist, _ = np.histogram(lengths, bins=LENGTH_BINS)
        self._baseline_length_dist = hist.astype(float)

    def set_expect_json(self, expected: bool) -> None:
        self._expected_json = expected

    def _validate_json(self, text: str) -> bool:
        try:
            json.loads(text)
            return True
        except (json.JSONDecodeError, ValueError):
            return False

    def score(self, text: str) -> tuple[bool, float]:
        """
        Returns (is_format_regression, kl_divergence_score).
        JSON validation fails → immediate regression.
        KL divergence > threshold → regression.
        """
        length = len(text.split())
        self._length_window.append(length)

        if self._expected_json and not self._validate_json(text):
            return True, 1.0

        if self._baseline_length_dist is None or len(self._length_window) < 100:
            return False, 0.0

        current_hist, _ = np.histogram(list(self._length_window), bins=LENGTH_BINS)
        kl = _compute_kl_divergence(current_hist.astype(float), self._baseline_length_dist)

        return kl > settings.format_kl_threshold, kl

    async def validate_batch(self, log_events: list[dict]) -> list[tuple[str, float]]:
        results = []
        for event in log_events:
            is_regression, score = self.score(event.get("completion", ""))
            if is_regression:
                results.append(("format_regression", score))
            else:
                results.append(("", score))
        return results
