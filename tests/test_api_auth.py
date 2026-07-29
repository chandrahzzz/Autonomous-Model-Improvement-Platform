"""
Tests for control-plane API authentication.

Before this, all 12 mutating endpoints — including `POST /models/rollback/{tag}`
and `POST /knowledge/documents` (which writes the corpus the teacher grounds
corrections on) — were reachable with no credential at all, while
`k8s/ingress.yaml` published the service publicly.
"""

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from src.api import auth
from src.api.auth import require_api_key
from src.config.settings import settings

VALID = "k" * 32
OTHER = "j" * 32


@pytest.fixture(autouse=True)
def _reset():
    auth.reset_for_tests()
    yield
    auth.reset_for_tests()


@pytest.fixture
def client():
    app = FastAPI()

    @app.get("/open")
    async def open_route():
        return {"ok": True}

    @app.post("/guarded", dependencies=[Depends(require_api_key)])
    async def guarded_route():
        return {"ok": True}

    return TestClient(app)


def _configure(monkeypatch, keys: str, environment: str = "development", enabled: bool = True):
    monkeypatch.setattr(settings, "api_keys", keys)
    monkeypatch.setattr(settings, "environment", environment)
    monkeypatch.setattr(settings, "api_auth_enabled", enabled)


# ── Rejection ────────────────────────────────────────────────────────────────
def test_request_without_key_is_rejected(monkeypatch, client):
    _configure(monkeypatch, VALID)
    resp = client.post("/guarded")
    assert resp.status_code == 401


def test_request_with_wrong_key_is_rejected(monkeypatch, client):
    _configure(monkeypatch, VALID)
    resp = client.post("/guarded", headers={"X-API-Key": OTHER})
    assert resp.status_code == 403


def test_empty_key_header_is_rejected(monkeypatch, client):
    _configure(monkeypatch, VALID)
    assert client.post("/guarded", headers={"X-API-Key": "   "}).status_code == 401


def test_malformed_authorization_header_is_rejected(monkeypatch, client):
    _configure(monkeypatch, VALID)
    resp = client.post("/guarded", headers={"Authorization": f"Basic {VALID}"})
    assert resp.status_code == 401


# ── Acceptance ───────────────────────────────────────────────────────────────
def test_valid_key_via_x_api_key(monkeypatch, client):
    _configure(monkeypatch, VALID)
    assert client.post("/guarded", headers={"X-API-Key": VALID}).status_code == 200


def test_valid_key_via_bearer(monkeypatch, client):
    _configure(monkeypatch, VALID)
    resp = client.post("/guarded", headers={"Authorization": f"Bearer {VALID}"})
    assert resp.status_code == 200


def test_bearer_is_case_insensitive(monkeypatch, client):
    _configure(monkeypatch, VALID)
    resp = client.post("/guarded", headers={"Authorization": f"bearer {VALID}"})
    assert resp.status_code == 200


def test_any_configured_key_is_accepted(monkeypatch, client):
    """Multiple keys let clients rotate independently."""
    _configure(monkeypatch, f"{VALID},{OTHER}")
    assert client.post("/guarded", headers={"X-API-Key": VALID}).status_code == 200
    assert client.post("/guarded", headers={"X-API-Key": OTHER}).status_code == 200


def test_whitespace_around_configured_keys_is_tolerated(monkeypatch, client):
    _configure(monkeypatch, f"  {VALID} , {OTHER}  ")
    assert client.post("/guarded", headers={"X-API-Key": VALID}).status_code == 200


def test_unguarded_routes_stay_open(monkeypatch, client):
    _configure(monkeypatch, VALID)
    assert client.get("/open").status_code == 200


# ── Misconfiguration ─────────────────────────────────────────────────────────
def test_production_fails_closed_when_no_keys_configured(monkeypatch, client):
    """Serving an unauthenticated control plane is the failure this prevents."""
    _configure(monkeypatch, "", environment="production")
    resp = client.post("/guarded", headers={"X-API-Key": VALID})
    assert resp.status_code == 503


def test_development_allows_when_no_keys_configured(monkeypatch, client):
    _configure(monkeypatch, "", environment="development")
    assert client.post("/guarded").status_code == 200


def test_kill_switch_disables_auth(monkeypatch, client):
    _configure(monkeypatch, VALID, enabled=False)
    assert client.post("/guarded").status_code == 200


@pytest.mark.asyncio
async def test_dependency_raises_401_not_500_when_key_missing(monkeypatch):
    """Guard against the dependency erroring instead of cleanly rejecting."""
    _configure(monkeypatch, VALID)
    with pytest.raises(HTTPException) as exc:
        await require_api_key(x_api_key=None, authorization=None)
    assert exc.value.status_code == 401


# ── Wiring: the real app protects the control plane, not the probes ──────────
# Asserted functionally rather than by inspecting route objects: this FastAPI
# version wraps included routers, and behaviour is what actually matters.
# TestClient is used WITHOUT a context manager so the lifespan (which requires a
# live database) never runs — auth rejects before any handler or dependency.

OPEN_PREFIXES = ("/health", "/metrics", "/sim", "/docs", "/openapi", "/redoc")


@pytest.fixture(scope="module")
def real_app():
    from src.api.main import create_app

    return create_app()


@pytest.fixture
def real_client(real_app, monkeypatch):
    monkeypatch.setattr(settings, "api_keys", VALID)
    monkeypatch.setattr(settings, "environment", "development")
    monkeypatch.setattr(settings, "api_auth_enabled", True)
    # raise_server_exceptions=False so a handler that fails on missing
    # infrastructure (Redis/Postgres aren't up in unit tests) surfaces as a 500
    # instead of propagating — auth outcomes stay distinguishable from them.
    return TestClient(real_app, raise_server_exceptions=False)


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/pipeline/pause"),
        ("post", "/pipeline/resume"),
        ("post", "/pipeline/trigger"),
        ("post", "/models/rollback/v8"),
        ("post", "/knowledge/documents"),
        ("post", "/shadow/abort"),
        ("get", "/audit/trail"),
    ],
)
def test_control_plane_requires_a_key(real_client, method, path):
    resp = getattr(real_client, method)(path)
    assert resp.status_code == 401, (
        f"{method.upper()} {path} answered {resp.status_code} without a key"
    )


def test_wrong_key_is_forbidden_on_a_real_route(real_client):
    resp = real_client.post("/pipeline/pause", headers={"X-API-Key": OTHER})
    assert resp.status_code == 403


def test_valid_key_passes_authentication_on_a_real_route(real_client):
    """A correct key must get past auth. The handler may still fail on
    infrastructure (no Redis in unit tests) — that is not an auth outcome."""
    resp = real_client.post("/pipeline/pause", headers={"X-API-Key": VALID})
    assert resp.status_code not in (401, 403)


@pytest.mark.parametrize("path", ["/health", "/metrics"])
def test_probe_routes_stay_open(real_client, path):
    """k8s probes and the Prometheus scraper cannot present a key."""
    resp = real_client.get(path)
    assert resp.status_code not in (401, 403), f"{path} must stay unauthenticated"


def test_every_mutating_route_requires_a_key(real_client, real_app):
    """Catch-all so a future endpoint cannot ship unauthenticated.

    Walks the OpenAPI schema and calls every mutating operation outside the open
    set with no credential; each must answer 401.
    """
    schema = real_app.openapi()
    checked, unprotected = 0, []

    for path, operations in schema.get("paths", {}).items():
        if path.startswith(OPEN_PREFIXES):
            continue
        for method in operations:
            if method.lower() not in ("post", "put", "patch", "delete"):
                continue
            # Fill path params with a throwaway value so routing resolves.
            concrete = path
            while "{" in concrete:
                start, end = concrete.index("{"), concrete.index("}")
                concrete = concrete[:start] + "1" + concrete[end + 1:]
            resp = getattr(real_client, method.lower())(concrete)
            checked += 1
            if resp.status_code != 401:
                unprotected.append(f"{method.upper()} {path} -> {resp.status_code}")

    assert checked > 0, "no mutating routes were discovered — test is not exercising anything"
    assert unprotected == [], f"unauthenticated mutating routes: {unprotected}"
