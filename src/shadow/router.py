"""
Shadow traffic router. Routes 10% of production traffic to the challenger
model for silent scoring. Challenger output is NEVER served to users.
"""

import random
import time
import asyncio
import structlog

import redis.asyncio as aioredis

from src.config.settings import settings
from src.monitoring.metrics import (
    shadow_requests_total,
    shadow_quality_delta_histogram,
    shadow_sample_skipped_overrepresented_total,
)

log = structlog.get_logger()

SHADOW_ACTIVE_KEY = "shadow:active_version"
SHADOW_ABORT_KEY = "shadow:abort"


def _bucket_keep_multiplier(bucket_counts: dict[str, int], current_bucket: str) -> float:
    """Stratified-sampling correction (#S1). Given how many shadow samples each
    time bucket already holds, return a [0,1] multiplier on the base sampling rate
    for the current bucket so that over-represented (peak-hour) buckets are
    down-sampled toward the least-represented bucket. Pure + deterministic."""
    if not bucket_counts:
        return 1.0
    cur = bucket_counts.get(current_bucket, 0)
    min_count = min([*bucket_counts.values(), cur])
    if cur <= min_count:
        return 1.0
    # Accept the current bucket at a rate that pulls it back toward the minimum.
    return max(0.1, (min_count + 1) / (cur + 1))


class ShadowRouter:
    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._judge = None  # lazily-created LLM-as-judge client
        self._eval_cache: list[tuple[str, str]] | None = None  # (question, ground_truth)

    async def get_challenger_version(self) -> str | None:
        """Returns active challenger version tag or None if shadow not active."""
        aborted = await self._redis.exists(SHADOW_ABORT_KEY)
        if aborted:
            return None
        return await self._redis.get(SHADOW_ACTIVE_KEY)

    async def set_challenger(self, version_tag: str) -> None:
        await self._redis.set(SHADOW_ACTIVE_KEY, version_tag)
        await self._redis.delete(SHADOW_ABORT_KEY)
        log.info("shadow_challenger_set", version_tag=version_tag)

    async def clear_challenger(self) -> None:
        await self._redis.delete(SHADOW_ACTIVE_KEY)
        log.info("shadow_challenger_cleared")

    async def maybe_shadow(
        self,
        prompt: str,
        production_output: str,
        challenger_invoke_fn,
    ) -> float | None:
        """
        Routes to shadow with probability ab_shadow_traffic_pct.
        Returns quality delta if shadowed, else None.
        Never serves challenger output to the user.
        """
        challenger_version = await self.get_challenger_version()
        if not challenger_version:
            return None

        if random.random() > settings.ab_shadow_traffic_pct:
            return None

        # Stratified temporal sampling: keep buckets balanced so the A/B sample
        # reflects the full daily traffic distribution, not just peak hours (#S1).
        if settings.shadow_stratified_sampling_enabled and not await self._stratified_admit():
            shadow_sample_skipped_overrepresented_total.inc()
            return None

        try:
            challenger_output = await challenger_invoke_fn(prompt)
            quality_delta = await self._score_delta(
                prompt, production_output, challenger_output
            )
            # Reference scoring may abstain (no comparable ground truth) — skip
            # so we don't pollute the A/B sample with meaningless zeros.
            if quality_delta is None:
                return None

            shadow_requests_total.labels(challenger_version=challenger_version).inc()
            shadow_quality_delta_histogram.observe(quality_delta)

            # Log to DB asynchronously
            asyncio.create_task(
                self._log_shadow(
                    challenger_version, prompt, production_output, challenger_output, quality_delta
                )
            )
            return quality_delta
        except Exception:
            log.exception("shadow_routing_failed")
            return None

    def _current_bucket(self) -> str:
        bucket_secs = max(1, settings.shadow_stratify_bucket_hours) * 3600
        return str(int(time.time() // bucket_secs))

    async def _stratified_admit(self) -> bool:
        """Decide whether to keep this sample given bucket balance, and record it
        if kept. Fails open (admits) on any Redis error so shadowing never stalls."""
        try:
            bucket = self._current_bucket()
            raw = await self._redis.hgetall(settings.shadow_samples_bucket_key)
            counts = {k: int(v) for k, v in raw.items()} if raw else {}
            multiplier = _bucket_keep_multiplier(counts, bucket)
            if random.random() >= multiplier:
                return False
            await self._redis.hincrby(settings.shadow_samples_bucket_key, bucket, 1)
            # Expire the whole map a few buckets out so old tests don't skew new ones.
            await self._redis.expire(
                settings.shadow_samples_bucket_key,
                settings.shadow_stratify_bucket_hours * 3600 * 12,
            )
            return True
        except Exception:
            log.warning("shadow_stratified_admit_failed_open")
            return True

    async def _score_delta(
        self, prompt: str, production: str, challenger: str
    ) -> float | None:
        """Signed quality delta (challenger - production); positive = challenger
        is better. The old metric compared production to *itself* (always 1.0),
        so every challenger scored <= 0 and could never be promoted. We now score
        against an external reference (LLM judge) or eval-set ground truth."""
        if settings.shadow_scoring_strategy == "reference_rouge":
            return self._score_delta_reference(prompt, production, challenger)
        return await self._score_delta_llm_judge(prompt, production, challenger)

    async def _score_delta_llm_judge(
        self, prompt: str, production: str, challenger: str
    ) -> float:
        """Ask a cheap judge model to rate both answers 1-10; return the
        normalized difference in [-1, 1]. On any error, abstain to 0.0 (neutral,
        not penalizing the challenger)."""
        import json as _json
        if self._judge is None:
            from langchain_openai import ChatOpenAI
            self._judge = ChatOpenAI(
                model=settings.shadow_judge_model,
                temperature=0.0,
                api_key=settings.openai_api_key,
            )
        from langchain_core.messages import SystemMessage, HumanMessage
        sys_msg = SystemMessage(content=(
            "You are an impartial evaluator. Rate each assistant answer to the "
            "user prompt on a 1-10 scale for helpfulness, accuracy, and clarity. "
            "Respond ONLY with JSON: {\"production_score\": int, \"challenger_score\": int}."
        ))
        human = HumanMessage(content=(
            f"PROMPT:\n{prompt[:2000]}\n\n"
            f"ANSWER A (production):\n{production[:2000]}\n\n"
            f"ANSWER B (challenger):\n{challenger[:2000]}"
        ))
        try:
            resp = await self._judge.ainvoke([sys_msg, human])
            data = _json.loads(str(resp.content).strip().strip("`"))
            prod_s = float(data["production_score"])
            chal_s = float(data["challenger_score"])
            # Track judge spend (gpt-4o-mini ~ $0.15/$0.60 per 1M in/out tokens).
            try:
                from src.monitoring.cost_tracker import CostTracker
                in_tok = (len(prompt) + len(production) + len(challenger)) / 4.0
                cost = in_tok / 1_000_000 * 0.15 + 64 / 1_000_000 * 0.60
                await CostTracker().record_spend("shadow_judge", cost)
            except Exception:
                pass
            return (chal_s - prod_s) / 10.0
        except Exception:
            log.warning("shadow_llm_judge_failed_abstain")
            return 0.0

    def _score_delta_reference(
        self, prompt: str, production: str, challenger: str
    ) -> float | None:
        """ROUGE-L of each answer against the closest eval-set ground truth.
        Abstains (returns None) when no eval example is similar enough."""
        gt = self._closest_ground_truth(prompt)
        if gt is None:
            return None
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
            chal = scorer.score(gt, challenger)["rougeL"].fmeasure
            prod = scorer.score(gt, production)["rougeL"].fmeasure
            return float(chal - prod)
        except Exception:
            return None

    def _closest_ground_truth(self, prompt: str) -> str | None:
        """Token-Jaccard match of the prompt against the cached eval set. Cheap
        enough for the request path and avoids an embedding model dependency."""
        if self._eval_cache is None:
            self._eval_cache = self._load_eval_cache()
        ptoks = set(prompt.lower().split())
        if not ptoks:
            return None
        best_gt, best_sim = None, 0.0
        for q, gt in self._eval_cache:
            qtoks = set(q.lower().split())
            if not qtoks:
                continue
            sim = len(ptoks & qtoks) / len(ptoks | qtoks)
            if sim > best_sim:
                best_gt, best_sim = gt, sim
        return best_gt if best_sim >= settings.shadow_reference_sim_threshold else None

    def _load_eval_cache(self) -> list[tuple[str, str]]:
        try:
            import json as _json
            from pathlib import Path
            fixture = (
                Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval_set.json"
            )
            data = _json.loads(fixture.read_text(encoding="utf-8"))
            return [(e["question"], e["ground_truth"]) for e in data]
        except Exception:
            return []

    async def _log_shadow(
        self,
        version: str,
        prompt: str,
        prod_out: str,
        chal_out: str,
        delta: float,
    ) -> None:
        """Persist a shadow observation so the A/B window can accumulate samples.
        (Previously a no-op, which meant shadow_logs was never populated and the
        promotion gate could never reach its minimum request count.)"""
        try:
            from src.db.connection import get_db
            from src.shadow.ab_collector import ABCollector
            async with get_db() as db:
                await ABCollector(db).record(
                    challenger_version=version,
                    prompt=prompt,
                    production_output=prod_out,
                    challenger_output=chal_out,
                    quality_delta=delta,
                )
        except Exception:
            log.warning("shadow_log_persist_failed")
