"""
API authentication.

Every mutating endpoint on this service is a control-plane action:
`POST /models/rollback/{tag}` swaps the production model, `POST /pipeline/pause`
stops the loop, `POST /knowledge/documents` writes into the corpus the teacher
grounds corrections on (a training-data poisoning path), and several routers
expose DELETEs. None of it was authenticated, while `k8s/ingress.yaml` publishes
the service on a public host.

Reads are protected too — the audit trail, model registry and eval set are not
public data. Only liveness/readiness (`/health`), the Prometheus scrape
(`/metrics`) and the simulator's no-op endpoint are left open, because probes and
scrapers cannot carry a key.

Keys come from `API_KEYS` (comma-separated, so multiple clients can be rotated
independently) and are compared in constant time.
"""

from __future__ import annotations

import secrets

import structlog
from fastapi import Header, HTTPException, status

from src.config.settings import settings

log = structlog.get_logger()

_BEARER_PREFIX = "bearer "
_dev_bypass_warned = False


def _configured_keys() -> list[str]:
    return [k.strip() for k in (settings.api_keys or "").split(",") if k.strip()]


def _presented_key(x_api_key: str | None, authorization: str | None) -> str | None:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith(_BEARER_PREFIX):
        return authorization[len(_BEARER_PREFIX):].strip()
    return None


def _matches_any(presented: str, valid: list[str]) -> bool:
    """Constant-time comparison against every configured key.

    Deliberately checks all of them rather than short-circuiting, so response
    time doesn't leak which key position matched.
    """
    ok = False
    for key in valid:
        if secrets.compare_digest(presented, key):
            ok = True
    return ok


async def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
) -> None:
    """Reject the request unless it carries a valid API key.

    Accepts `X-API-Key: <key>` or `Authorization: Bearer <key>`.
    """
    global _dev_bypass_warned

    if not settings.api_auth_enabled:
        return

    valid = _configured_keys()
    if not valid:
        # Misconfiguration. In production this must fail CLOSED: silently
        # serving an unauthenticated control plane is the failure mode this
        # module exists to prevent.
        if settings.environment == "production":
            log.error("api_auth_no_keys_configured_refusing_requests")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="API authentication is enabled but no API_KEYS are configured.",
            )
        if not _dev_bypass_warned:
            log.warning(
                "api_auth_disabled_no_keys",
                note=(
                    "No API_KEYS set and environment is not production, so the "
                    "control plane is UNAUTHENTICATED. Set API_KEYS before "
                    "exposing this service."
                ),
            )
            _dev_bypass_warned = True
        return

    presented = _presented_key(x_api_key, authorization)
    if not presented:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Send X-API-Key or Authorization: Bearer <key>.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not _matches_any(presented, valid):
        log.warning("api_auth_rejected_invalid_key")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )


def reset_for_tests() -> None:
    global _dev_bypass_warned
    _dev_bypass_warned = False
