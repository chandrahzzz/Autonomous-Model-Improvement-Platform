"""
HashiCorp Vault client for fetching secrets at runtime.
Falls back to environment variables in development mode.
"""

import hvac
import structlog
from functools import lru_cache

from src.config.settings import settings

log = structlog.get_logger()


class VaultClient:
    def __init__(self) -> None:
        self._client = hvac.Client(url=settings.vault_url, token=settings.vault_token)

    def is_authenticated(self) -> bool:
        try:
            return self._client.is_authenticated()
        except Exception:
            return False

    def get_secret(self, path: str, key: str) -> str:
        """Fetch a KV v2 secret from Vault. Raises on failure — fail closed."""
        try:
            response = self._client.secrets.kv.v2.read_secret_version(path=path)
            value = response["data"]["data"][key]
            log.debug("vault_secret_fetched", path=path, key=key)
            return str(value)
        except Exception as e:
            log.error("vault_secret_fetch_failed", path=path, key=key, error=str(e))
            raise RuntimeError(f"Failed to fetch secret {path}/{key} from Vault") from e

    def get_hmac_key(self) -> bytes:
        """Fetch the HMAC signing key used for audit trail signatures."""
        path_parts = settings.hmac_key_path.split("/", 1)
        if len(path_parts) != 2:
            raise ValueError(f"Invalid hmac_key_path: {settings.hmac_key_path}")
        secret_path = path_parts[1]
        raw = self.get_secret(secret_path, "key")
        return raw.encode("utf-8")


@lru_cache(maxsize=1)
def get_vault_client() -> VaultClient:
    return VaultClient()
