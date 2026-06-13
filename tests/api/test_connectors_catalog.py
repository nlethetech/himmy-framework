"""T1.3 — the read-only ``/v1/connectors`` catalog endpoint.

Asserts the catalog returns the same connector set the CLI/service expose, leaks NO
secret values, is RBAC-gated on ``connector:read`` (403 for a role lacking it, allowed
for viewer), and 404s an unknown connector.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)

_TOUCHED = ["HIMMY_GITHUB_TOKEN"]


@pytest.fixture
def writable_backend(tmp_path: Path):
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in _TOUCHED:
        os.environ.pop(name, None)
    yield
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


def _open_client() -> TestClient:
    """No authenticator → offline default (RBAC bypassed)."""
    return TestClient(create_app(ApiContainer.build_default()))


def _auth_client(role: str) -> TestClient:
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["t"], roles=[role], auth_method="apikey"
            )
        }
    )
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


# ----------------------------------------------------------------- catalog shape


def test_catalog_lists_connectors(writable_backend: None) -> None:
    resp = _open_client().get("/v1/connectors")
    assert resp.status_code == 200
    body = resp.json()
    names = {c["name"] for c in body["connectors"]}
    assert {"slack", "discord", "github", "websearch", "webhook"} <= names
    # each row carries the descriptor + status fields, no secret values
    row = next(c for c in body["connectors"] if c["name"] == "github")
    assert row["kind"] == "outbound"
    assert "enabled" in row and "secrets" in row
    assert all("value" not in s for s in row["secrets"])  # only name/present


def test_catalog_never_returns_a_secret_value(writable_backend: None) -> None:
    from himmy.connectors.manage import ConnectorService

    ConnectorService().configure("github", {"HIMMY_GITHUB_TOKEN": "ghp_topsecret"})
    resp = _open_client().get("/v1/connectors")
    assert "ghp_topsecret" not in resp.text
    row = next(c for c in resp.json()["connectors"] if c["name"] == "github")
    assert row["configured"] is True
    token = next(s for s in row["secrets"] if s["name"] == "HIMMY_GITHUB_TOKEN")
    assert token["present"] is True


def test_get_single_connector(writable_backend: None) -> None:
    resp = _open_client().get("/v1/connectors/slack")
    assert resp.status_code == 200
    assert resp.json()["name"] == "slack"


def test_get_unknown_connector_404(writable_backend: None) -> None:
    resp = _open_client().get("/v1/connectors/nope")
    assert resp.status_code == 404


# --------------------------------------------------------------------- RBAC


def test_viewer_can_read_catalog(writable_backend: None) -> None:
    resp = _auth_client("viewer").get("/v1/connectors")
    assert resp.status_code == 200


def test_role_without_permission_is_denied(writable_backend: None) -> None:
    # data_subject holds only consent:read → no connector:read
    resp = _auth_client("data_subject").get("/v1/connectors")
    assert resp.status_code == 403


def test_get_single_denied_without_permission(writable_backend: None) -> None:
    resp = _auth_client("data_subject").get("/v1/connectors/slack")
    assert resp.status_code == 403


# ----------------------------------------------------- read-only (no config route)


def test_no_write_route_on_v1(writable_backend: None) -> None:
    # /v1 connector CONFIG is explicitly out of scope — only GETs exist.
    c = _open_client()
    assert c.post("/v1/connectors", json={}).status_code in (404, 405)
    assert c.put("/v1/connectors/github", json={}).status_code in (404, 405)
