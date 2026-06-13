"""T1.4(c) — the Studio Connections adapter over the connector-management surface.

The Studio Connections screen now lists the connector-managed connections
(Slack/GitHub/Discord/webhook) alongside the legacy email/telegram/web_search ones, each
carrying its ``kind``/``enableable``/``enabled``/``allowlist_field``/``mount_path``, and the
BFF exposes ``PUT /connections/{ctype}/enable|disable`` delegating to the manage service.
These tests cover the catalog shape, the enable/disable round-trip, the allow-list surfacing,
the read-only-backend 409, and the unknown/non-enableable 404 — all offline.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)

_TOUCHED = [
    "HIMMY_GITHUB_TOKEN",
    "HIMMY_GITHUB_ALLOWED_REPOS",
    "HIMMY_CONNECTOR_GITHUB_OUTBOUND_ENABLED",
    "HIMMY_WEBHOOK_SIGNING_SECRET",
    "HIMMY_CONNECTOR_WEBHOOK_INBOUND_ENABLED",
    "HIMMY_STUDIO_GUARD",
]


@pytest.fixture
def writable(tmp_path: Path):
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in _TOUCHED:
        os.environ.pop(name, None)
    os.environ["HIMMY_STUDIO_GUARD"] = "0"
    yield
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


@pytest.fixture
def env_only():
    """A read-only env-only secrets backend (enable/disable must 409)."""
    configure_secrets(EnvSecrets())
    os.environ["HIMMY_STUDIO_GUARD"] = "0"
    yield
    configure_secrets(None)
    os.environ.pop("HIMMY_STUDIO_GUARD", None)


def _client() -> TestClient:
    return TestClient(create_app(ApiContainer.build_default()))


# ------------------------------------------------------------------ catalog shape
def test_lists_the_four_connector_managed_connections(writable: None) -> None:
    rows = _client().get("/api/studio/connections").json()
    types = {r["type"] for r in rows}
    assert {"slack", "github", "discord", "webhook"} <= types
    gh = next(r for r in rows if r["type"] == "github")
    assert gh["kind"] == "outbound"
    assert gh["enableable"] is True
    assert gh["enabled"] is False
    assert gh["allowlist_field"] == "allowed_repos"
    wh = next(r for r in rows if r["type"] == "webhook")
    assert wh["kind"] == "inbound"
    assert wh["mount_path"] == "/v1/connectors/webhook"


def test_legacy_connections_stay_non_enableable(writable: None) -> None:
    rows = _client().get("/api/studio/connections").json()
    email = next(r for r in rows if r["type"] == "email")
    assert email["enableable"] is False
    assert email["enabled"] is False


# ------------------------------------------------------------ enable / disable
def test_enable_disable_round_trip(writable: None) -> None:
    c = _client()
    c.put("/api/studio/connections/github", json={"fields": {"token": "ghp_x"}})
    on = c.put("/api/studio/connections/github/enable")
    assert on.status_code == 200
    assert on.json()["enabled"] is True
    # round-trips through a fresh app (persisted to the writable backend)
    rows = _client().get("/api/studio/connections").json()
    assert next(r for r in rows if r["type"] == "github")["enabled"] is True
    off = c.put("/api/studio/connections/github/disable")
    assert off.status_code == 200
    assert off.json()["enabled"] is False


def test_allowlist_is_surfaced_via_the_field(writable: None) -> None:
    c = _client()
    c.put(
        "/api/studio/connections/github",
        json={"fields": {"token": "ghp_x", "allowed_repos": "o/a,o/b"}},
    )
    rows = c.get("/api/studio/connections").json()
    gh = next(r for r in rows if r["type"] == "github")
    allow_field = next(f for f in gh["fields"] if f["name"] == "allowed_repos")
    assert allow_field["present"] is True
    # the allow-list value is NOT a secret, so its hint is the value itself (not masked)
    assert allow_field["masked_hint"] == "o/a,o/b"


def test_enable_unknown_or_non_enableable_is_404(writable: None) -> None:
    c = _client()
    assert c.put("/api/studio/connections/nope/enable").status_code == 404
    # email is a valid connection but NOT enableable
    assert c.put("/api/studio/connections/email/enable").status_code == 404


def test_enable_on_readonly_backend_is_409(env_only: None) -> None:
    resp = _client().put("/api/studio/connections/github/enable")
    assert resp.status_code == 409


def test_disable_clears_enabled_flag(writable: None) -> None:
    from himmy.connectors.manage import ConnectorService

    ConnectorService().enable("github", "outbound")
    c = _client()
    rows = c.get("/api/studio/connections").json()
    assert next(r for r in rows if r["type"] == "github")["enabled"] is True
    c.put("/api/studio/connections/github/disable")
    rows2 = c.get("/api/studio/connections").json()
    assert next(r for r in rows2 if r["type"] == "github")["enabled"] is False
