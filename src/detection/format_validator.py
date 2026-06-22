"""
Format regression detector.

Two signals:
1. JSON schema validation — if outputs are expected to be JSON, validate them.
2. Length distribution KL divergence — if response lengths shift significantly
   from the baseline distribution, it indicates format regression.
"""

import json
from collections import deque
from datetime import datetime, timezone
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
        self._baseline_computed_at: datetime | None = None
        self._expected_json: bool = False

    def set_baseline_lengths(self, lengths: list[int]) -> None:
        hist, _ = np.histogram(lengths, bins=LENGTH_BINS)
        self._baseline_length_dist = hist.astype(float)
        self._baseline_computed_at = datetime.now(timezone.utc)

    def set_expect_json(self, expected: bool) -> None:
        self._expected_json = expected

    @property
    def baseline_age_hours(self) -> float | None:
        if self._baseline_computed_at is None:
            return None
        delta = datetime.now(timezone.utc) - self._baseline_computed_at
        return delta.total_seconds() / 3600.0

    def is_baseline_stale(self) -> bool:
        age = self.baseline_age_hours
        return age is not None and age > settings.format_baseline_max_age_hours

    def refresh_baseline(self, completions: list[str]) -> bool:
        """Recompute the length-distribution baseline from recent good outputs.

        Called after a promotion and periodically (#5). Without this, an
        intentional product-wide change in response length (e.g. short → detailed
        answers) makes KL divergence fire forever against a frozen seed. Returns
        False if there aren't enough samples to form a trustworthy baseline.
        """
        if len(completions) < settings.format_min_samples:
            return False
        lengths = [len(c.split()) for c in completions]
        self.set_baseline_lengths(lengths)
        log.info("format_baseline_refreshed", n=len(lengths))
        return True

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

        if self._baseline_length_dist is None or len(self._length_window) < settings.format_min_samples:
            if self._baseline_length_dist is not None:
                from src.monitoring.metrics import detector_insufficient_data_total
                detector_insufficient_data_total.labels(detector="format").inc()
            return False, 0.0

        current_hist, _ = np.histogram(list(self._length_window), bins=LENGTH_BINS)
        kl = _compute_kl_divergence(current_hist.astype(float), self._baseline_length_dist)

        return kl > settings.format_kl_threshold, kl

    @property
    def window_size(self) -> int:
        return len(self._length_window)

    # ── Persistence (#4): length window + baseline survive restarts ────────────
    async def save_window_state(self, redis: Any) -> None:
        if not settings.detector_state_persist_enabled:
            return
        key = f"{settings.detector_state_redis_prefix}:format_state"
        payload = {
            "length_window": list(self._length_window),
            "baseline": self._baseline_length_dist.tolist() if self._baseline_length_dist is not None else None,
            "baseline_computed_at": self._baseline_computed_at.isoformat() if self._baseline_computed_at else None,
        }
        try:
            await redis.set(key, json.dumps(payload))
        except Exception:
            log.warning("format_state_persist_failed")

    async def load_window_state(self, redis: Any) -> None:
        if not settings.detector_state_persist_enabled:
            return
        key = f"{settings.detector_state_redis_prefix}:format_state"
        try:
            raw = await redis.get(key)
        except Exception:
            log.warning("format_state_rehydrate_failed")
            return
        if not raw:
            return
        try:
            payload = json.loads(raw)
            self._length_window = deque(
                (int(v) for v in payload.get("length_window", [])), maxlen=WINDOW_SIZE
            )
            if payload.get("baseline") is not None:
                self._baseline_length_dist = np.array(payload["baseline"], dtype=float)
            ts = payload.get("baseline_computed_at")
            if ts:
                dt = datetime.fromisoformat(ts)
                self._baseline_computed_at = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            log.info("format_state_rehydrated", n=len(self._length_window))
        except (ValueError, TypeError):
            log.warning("format_state_rehydrate_parse_failed")

    async def validate_batch(self, log_events: list[dict]) -> list[tuple[str, float]]:
        results = []
        for event in log_events:
            is_regression, score = self.score(event.get("completion", ""))
            if is_regression:
                results.append(("format_regression", score))
            else:
                results.append(("", score))
        return results
