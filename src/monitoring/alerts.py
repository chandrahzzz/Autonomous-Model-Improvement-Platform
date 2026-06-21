"""
PagerDuty alerting integration.
Fires when: safety regression, rollback storm (>3 rollbacks/day),
drift score sustained above threshold, training job timeout.
"""

import structlog
import httpx

from src.config.settings import settings

log = structlog.get_logger()

PAGERDUTY_EVENTS_API = "https://events.pagerduty.com/v2/enqueue"


class PagerDutyAlerter:
    def __init__(self) -> None:
        self._api_key = settings.pagerduty_api_key
        self._service_id = settings.pagerduty_service_id
        self._enabled = bool(self._api_key and self._service_id)

    async def trigger(
        self,
        summary: str,
        severity: str = "error",
        source: str = "continuous-finetuning-pipeline",
        details: dict | None = None,
    ) -> None:
        if not self._enabled:
            log.warning("pagerduty_not_configured_skipping_alert", summary=summary)
            return

        payload = {
            "routing_key": self._api_key,
            "event_action": "trigger",
            "payload": {
                "summary": summary,
                "severity": severity,
                "source": source,
                "custom_details": details or {},
                "component": "finetuning-pipeline",
                "group": "ml-infrastructure",
            },
        }

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(PAGERDUTY_EVENTS_API, json=payload)
                resp.raise_for_status()
                log.info("pagerduty_alert_sent", summary=summary, severity=severity)
        except Exception:
            log.exception("pagerduty_alert_failed", summary=summary)

    async def resolve(self, dedup_key: str) -> None:
        if not self._enabled:
            return
        payload = {
            "routing_key": self._api_key,
            "event_action": "resolve",
            "dedup_key": dedup_key,
        }
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(PAGERDUTY_EVENTS_API, json=payload)
                resp.raise_for_status()
        except Exception:
            log.exception("pagerduty_resolve_failed", dedup_key=dedup_key)


alerter = PagerDutyAlerter()
