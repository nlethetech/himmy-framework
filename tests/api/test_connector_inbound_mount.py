"""T1.4(b) — the BFF inbound-connector mount (create_app + mount_inbound_connectors).

An inbound connector (webhook/Slack/Discord intake) is mounted on the BFF ONLY when it is
enabled for the ``inbound`` surface AND its signing secret is present (capability OK) AND an
inbound ``agent.yaml`` is configured. These tests prove the gating end to end, offline:

* nothing is mounted with no inbound agent configured (the zero-config surface is unchanged);
* nothing is mounted when the connector is not enabled;
* nothing is mounted when the connector is enabled but its signing secret is absent
  (an unsigned public trigger is never exposed);
* WHEN enabled + secret present + agent configured, the route is mounted, a correctly
  signed delivery runs the agent (its reply comes back), and an unsigned one is 401.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.connector_inbound import INBOUND_AGENT_PATH_ENV
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)
from himmy.connectors.manage import ConnectorService

_SECRET = "whsec_aaaabbbbccccddddeeeeffff"

_TOUCHED = [
    "HIMMY_WEBHOOK_SIGNING_SECRET",
    "HIMMY_WEBHOOK_ALLOWED_SOURCES",
    "HIMMY_CONNECTOR_WEBHOOK_INBOUND_ENABLED",
    INBOUND_AGENT_PATH_ENV,
    "HIMMY_STUDIO_GUARD",
]


@pytest.fixture
def env(tmp_path: Path):
    """Writable secrets backend + an inbound agent.yaml; clean env around the test."""
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in _TOUCHED:
        os.environ.pop(name, None)
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        "name: inbound-bot\ndescription: inbound test agent\nprovider: stub\n"
    )
    # The rebinding guard would 403 a TestClient host; disable it for these mount tests
    # (the connector's OWN HMAC + allowlist gate is what we are exercising).
    os.environ["HIMMY_STUDIO_GUARD"] = "0"
    yield tmp_path
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


def _mounted_paths(app: object) -> set[str]:
    return {getattr(r, "path", "") for r in app.routes}


def _app() -> object:
    return create_app(ApiContainer.build_default())


def test_no_mount_without_inbound_agent(env: Path) -> None:
    svc = ConnectorService()
    svc.configure("webhook", {"HIMMY_WEBHOOK_SIGNING_SECRET": _SECRET})
    svc.enable("webhook", "inbound")
    # no INBOUND_AGENT_PATH set → nothing mounts
    assert "/v1/connectors/webhook" not in _mounted_paths(_app())


def test_no_mount_when_disabled(env: Path) -> None:
    svc = ConnectorService()
    svc.configure("webhook", {"HIMMY_WEBHOOK_SIGNING_SECRET": _SECRET})
    # enabled flag NOT set
    os.environ[INBOUND_AGENT_PATH_ENV] = str(env / "agent.yaml")
    assert "/v1/connectors/webhook" not in _mounted_paths(_app())


def test_no_mount_when_secret_absent(env: Path) -> None:
    svc = ConnectorService()
    svc.enable("webhook", "inbound")  # enabled but NO signing secret configured
    os.environ[INBOUND_AGENT_PATH_ENV] = str(env / "agent.yaml")
    # capability fails (no signing secret) → an unsigned public trigger is never mounted
    assert "/v1/connectors/webhook" not in _mounted_paths(_app())


def test_mounts_when_enabled_and_secret_present(env: Path) -> None:
    svc = ConnectorService()
    svc.configure(
        "webhook",
        {
            "HIMMY_WEBHOOK_SIGNING_SECRET": _SECRET,
            "HIMMY_WEBHOOK_ALLOWED_SOURCES": "ci",
        },
    )
    svc.enable("webhook", "inbound")
    os.environ[INBOUND_AGENT_PATH_ENV] = str(env / "agent.yaml")
    assert "/v1/connectors/webhook" in _mounted_paths(_app())


def test_mounted_signed_delivery_runs_agent_unsigned_is_401(env: Path) -> None:
    svc = ConnectorService()
    svc.configure(
        "webhook",
        {
            "HIMMY_WEBHOOK_SIGNING_SECRET": _SECRET,
            "HIMMY_WEBHOOK_ALLOWED_SOURCES": "ci",
        },
    )
    svc.enable("webhook", "inbound")
    os.environ[INBOUND_AGENT_PATH_ENV] = str(env / "agent.yaml")

    app = _app()
    body = json.dumps({"source": "ci", "text": "ping the agent", "id": "e1"}).encode()
    sig = "sha256=" + hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    with TestClient(app) as client:
        ok = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={"X-Himmy-Signature": sig},
        )
        assert ok.status_code == 200
        data = ok.json()
        assert data["ok"] is True
        assert data["handled"] is True
        # the stub agent actually ran and echoed the prompt text into its reply
        assert "ping the agent" in data["reply"]

        unsigned = client.post("/v1/connectors/webhook", content=body)
        assert unsigned.status_code == 401


def test_durable_container_mounts_webhook_with_durable_dedup(env: Path) -> None:
    """Q4 regression: under a DURABLE container the webhook router mounts (NOT skipped).

    Before the fix ``make_webhook_connector`` had no ``idempotency`` parameter, so the durable
    store injected through the ``build_inbound(**kwargs) -> idempotency=`` seam raised a
    ``TypeError`` that ``mount_inbound_connectors`` swallowed as 'failed to mount ... skipping' —
    silently removing the endpoint on every SQLite/Postgres deployment AND defeating durable
    dedup. With a durable (SQLite) container injected, ``_durable_idempotency_store`` returns a
    real ``DurableIdempotencyStore``; this test asserts (a) the route IS mounted (the seam no
    longer drops it) and (b) durable dedup actually engages — a repeated delivery id is deduped.
    """
    from himmy.api.deps import ApiContainer
    from himmy.services.storage.sqlite import SqliteStorageService

    svc = ConnectorService()
    svc.configure(
        "webhook",
        {
            "HIMMY_WEBHOOK_SIGNING_SECRET": _SECRET,
            "HIMMY_WEBHOOK_ALLOWED_SOURCES": "ci",
        },
    )
    svc.enable("webhook", "inbound")
    os.environ[INBOUND_AGENT_PATH_ENV] = str(env / "agent.yaml")

    # An INJECTED durable container (file-backed SQLite) — so the mount, which runs at
    # create_app time, sees a durable backend and injects the durable dedup store. (The
    # in-memory default returns None, which is why the offline path stays on the built-in
    # store.) Injected ⇒ the lifespan does not upgrade it; the durable backend is served as-is.
    durable = ApiContainer._assemble(
        SqliteStorageService(str(env / "storage.db"))
    )
    app = create_app(durable)
    assert "/v1/connectors/webhook" in _mounted_paths(app)

    body = json.dumps({"source": "ci", "text": "build it", "id": "dup-1"}).encode()
    sig = "sha256=" + hmac.new(_SECRET.encode(), body, hashlib.sha256).hexdigest()
    with TestClient(app) as client:
        first = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={"X-Himmy-Signature": sig},
        )
        assert first.status_code == 200
        assert first.json()["handled"] is True
        # The SAME delivery id again is deduped by the DURABLE store (not re-run).
        second = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={"X-Himmy-Signature": sig},
        )
        assert second.status_code == 200
        body2 = second.json()
        assert body2["handled"] is False and body2.get("deduplicated") is True
