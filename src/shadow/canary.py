"""
Canary controller: a small live rollout *before* full promotion.

Distinct from shadow testing:
  - Shadow: challenger runs silently, output NEVER served to users.
  - Canary: challenger serves a small % of REAL traffic, output IS served.

Like ShadowRouter, the actual traffic split happens in the serving layer, which
must call `should_route_to_canary()` and `record_result()`. The controller owns
the state machine and the pass/fail decision; state lives in Redis so it survives
restarts.

Success criteria (all must hold over the window):
  - error rate < canary_max_error_rate
  - no safety failures
  - at least canary_min_requests observed
"""

import json
import random
from datetime import datetime

import redis.asyncio as aioredis
import structlog

from src.config.settings import settings

log = structlog.get_logger()

ACTIVE_KEY = "canary:active"      # JSON: {version, start_ts, traffic_pct}
METRICS_KEY = "canary:metrics"    # hash: requests, errors, safety_failures


class CanaryController:
    def __init__(self, redis: aioredis.Redis | None = None) -> None:
        self._redis = redis

    async def _r(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)
        return self._redis

    async def start(self, version_tag: str) -> None:
        r = await self._r()
        await r.set(ACTIVE_KEY, json.dumps({
            "version": version_tag,
            "start_ts": datetime.utcnow().isoformat(),
            "traffic_pct": settings.canary_traffic_pct,
        }))
        await r.delete(METRICS_KEY)
        log.info("canary_started", version=version_tag, traffic_pct=settings.canary_traffic_pct)

    async def get_active(self) -> dict | None:
        r = await self._r()
        raw = await r.get(ACTIVE_KEY)
        return json.loads(raw) if raw else None

    async def should_route_to_canary(self, version_tag: str) -> bool:
        """Called by the serving layer per request."""
        active = await self.get_active()
        if not active or active["version"] != version_tag:
            return False
        return random.random() < float(active["traffic_pct"])

    async def record_result(
        self, success: bool, latency_ms: int = 0, safety_ok: bool = True
    ) -> None:
        """Called by the serving layer after a canary-served request."""
        r = await self._r()
        await r.hincrby(METRICS_KEY, "requests", 1)
        if not success:
            await r.hincrby(METRICS_KEY, "errors", 1)
        if not safety_ok:
            await r.hincrby(METRICS_KEY, "safety_failures", 1)

    async def status(self) -> dict:
        r = await self._r()
        active = await self.get_active()
        metrics = await r.hgetall(METRICS_KEY)
        return {"active": active, "metrics": metrics}

    async def evaluate(self) -> tuple[str, dict]:
        """Returns (decision, metrics): 'none' | 'pending' | 'promote' | 'rollback'."""
        active = await self.get_active()
        if not active:
            return "none", {}

        r = await self._r()
        raw = await r.hgetall(METRICS_KEY)
        requests = int(raw.get("requests", 0))
        errors = int(raw.get("errors", 0))
        safety_failures = int(raw.get("safety_failures", 0))
        elapsed_min = (datetime.utcnow() - datetime.fromisoformat(active["start_ts"])).total_seconds() / 60.0
        metrics = {
            "requests": requests, "errors": errors,
            "safety_failures": safety_failures, "elapsed_min": round(elapsed_min, 1),
        }

        # Real-time abort (#S2): don't wait for the window to close — if a safety
        # failure occurs, or the error rate spikes past the abort threshold once
        # enough requests have arrived, roll back immediately and page on-call.
        if requests >= settings.canary_min_requests_for_abort:
            error_rate = errors / requests
            abort_threshold = (
                settings.canary_max_error_rate * settings.canary_abort_error_multiplier
            )
            if safety_failures > 0 or error_rate > abort_threshold:
                from src.monitoring.metrics import canary_auto_aborts_total
                metrics["error_rate"] = round(error_rate, 4)
                metrics["reason"] = (
                    "safety_failure_during_canary" if safety_failures > 0
                    else f"error_rate_spike {error_rate:.4f} > {abort_threshold:.4f}"
                )
                canary_auto_aborts_total.inc()
                await self._alert_auto_abort(active.get("version"), metrics)
                log.error("canary_auto_abort", version=active.get("version"), **metrics)
                return "rollback", metrics

        if elapsed_min < settings.canary_window_minutes:
            return "pending", metrics

        # Window elapsed. Require enough traffic to make a call; allow up to 2x the
        # window for it to arrive, else fail safe (can't verify -> don't promote).
        if requests < settings.canary_min_requests:
            if elapsed_min < 2 * settings.canary_window_minutes:
                return "pending", metrics
            metrics["reason"] = "insufficient_canary_traffic"
            return "rollback", metrics

        error_rate = errors / requests
        metrics["error_rate"] = round(error_rate, 4)
        if safety_failures > 0:
            metrics["reason"] = "safety_failure_during_canary"
            return "rollback", metrics
        if error_rate > settings.canary_max_error_rate:
            metrics["reason"] = f"error_rate {error_rate:.4f} > {settings.canary_max_error_rate}"
            return "rollback", metrics
        return "promote", metrics

    async def _alert_auto_abort(self, version: str | None, metrics: dict) -> None:
        try:
            from src.monitoring.alerts import alerter
            await alerter.trigger(
                summary=f"Canary {version} auto-aborted: {metrics.get('reason')}",
                severity="critical",
                details=metrics,
            )
        except Exception:
            log.warning("canary_auto_abort_alert_failed")

    async def clear(self) -> None:
        r = await self._r()
        await r.delete(ACTIVE_KEY, METRICS_KEY)
