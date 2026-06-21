"""
Pipeline cost tracking and circuit breaker.

Accumulates external API/GPU spend in Redis so the pipeline has a notion of a
monthly budget. Complements the per-run teacher budget enforced in
TeacherModel: this is the global, cross-run view.

Redis keys:
  cost:monthly:{YYYY-MM}  -> accumulated USD for the month
"""

from datetime import datetime

import redis.asyncio as aioredis
import structlog

from src.config.settings import settings

log = structlog.get_logger()


class CostTracker:
    def _month_key(self) -> str:
        return f"cost:monthly:{datetime.utcnow():%Y-%m}"

    async def record_spend(self, category: str, amount_usd: float) -> None:
        if amount_usd <= 0:
            return
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            new_total = await r.incrbyfloat(self._month_key(), amount_usd)
            await r.aclose()
        except Exception:
            log.warning("cost_record_failed", category=category)
            return
        log.info("cost_recorded", category=category, amount=round(amount_usd, 4),
                 monthly_total=round(new_total, 2))
        if new_total >= settings.monthly_budget_usd:
            log.error("monthly_budget_exceeded", total=round(new_total, 2),
                      budget=settings.monthly_budget_usd)
        elif new_total >= 0.8 * settings.monthly_budget_usd:
            log.warning("monthly_budget_warning", total=round(new_total, 2),
                        budget=settings.monthly_budget_usd)

    async def check_budget(self) -> tuple[bool, str]:
        """Returns (within_budget, message). The runner can pause the pipeline
        when this returns False."""
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            raw = await r.get(self._month_key())
            await r.aclose()
        except Exception:
            return True, "cost_check_unavailable"  # fail open
        spent = float(raw or 0.0)
        if spent >= settings.monthly_budget_usd:
            return False, f"monthly_budget_exceeded: ${spent:.2f} >= ${settings.monthly_budget_usd}"
        return True, f"within_budget: ${spent:.2f} / ${settings.monthly_budget_usd}"

    async def get_monthly_summary(self) -> dict:
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            raw = await r.get(self._month_key())
            await r.aclose()
        except Exception:
            raw = None
        spent = float(raw or 0.0)
        return {
            "month": f"{datetime.utcnow():%Y-%m}",
            "spent_usd": round(spent, 2),
            "monthly_budget_usd": settings.monthly_budget_usd,
            "within_budget": spent < settings.monthly_budget_usd,
        }
