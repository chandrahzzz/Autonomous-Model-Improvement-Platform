"""
HMAC-SHA256 signer for audit trail entries.

Key is fetched from HashiCorp Vault. Falls back to settings.vault_token
in development mode.

Each audit entry's HMAC covers: id, event_type, decision, rationale,
state_snapshot, operator, created_at — so any modification is detectable.
"""

import hashlib
import hmac
import json
import structlog

log = structlog.get_logger()

_HMAC_KEY: bytes | None = None
_DEGRADED_MODE: bool = False


def is_degraded() -> bool:
    """True if the HMAC key fell back to the local secret instead of Vault.

    When True, audit-trail signatures are NOT backed by Vault and integrity
    guarantees are reduced. Surfaced via /health.
    """
    return _DEGRADED_MODE


def _get_key() -> bytes:
    global _HMAC_KEY, _DEGRADED_MODE
    if _HMAC_KEY is not None:
        return _HMAC_KEY

    from src.config.settings import settings
    try:
        from src.config.vault import get_vault_client
        client = get_vault_client()
        if client.is_authenticated():
            _HMAC_KEY = client.get_hmac_key()
            log.info("hmac_key_loaded_from_vault")
            return _HMAC_KEY
        raise RuntimeError("Vault client is not authenticated")
    except Exception as e:
        # Vault unreachable. In production this MUST fail loudly — a silent
        # downgrade to the .env secret would make the audit trail look
        # cryptographically signed when it is not.
        if settings.vault_required:
            raise RuntimeError(
                f"Vault is required (VAULT_REQUIRED=true) but the HMAC key could "
                f"not be loaded: {e}. Refusing to operate with an unsigned audit "
                f"trail. Fix Vault connectivity or set VAULT_REQUIRED=false for dev."
            ) from e
        log.warning(
            "vault_unavailable_using_fallback_key",
            error=str(e),
            message=(
                "AUDIT TRAIL HMAC IS NOT BACKED BY VAULT — using the local secret "
                "key. This is only acceptable in development."
            ),
        )
        _DEGRADED_MODE = True
        _HMAC_KEY = settings.secret_key.encode("utf-8")
        return _HMAC_KEY


class HMACSigner:
    def sign(self, payload: dict) -> str:
        """Compute HMAC-SHA256 over canonical JSON of payload."""
        canonical = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        key = _get_key()
        signature = hmac.new(key, canonical, hashlib.sha256).hexdigest()
        return signature

    def verify(self, payload: dict, expected_signature: str) -> bool:
        """Constant-time comparison to prevent timing attacks."""
        computed = self.sign(payload)
        return hmac.compare_digest(computed, expected_signature)
