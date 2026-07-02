"""agent-aware-service (P0 #2/#3): ``himmy serve -f`` exposes the agent as a signed endpoint.

``himmy serve -f agent.yaml`` (and the reusable summary helper) is the one-command front door
over the EXISTING inbound machinery — it must reuse :mod:`himmy.api.connector_inbound` without
relaxing its default-deny + HMAC posture. These tests prove, in-process and offline:

* the serve wiring (:func:`_enable_inbound_webhook`) points the inbound agent at the file and
  enables the webhook so ``mount_inbound_connectors`` mounts POST ``/v1/connectors/webhook``;
* a correctly-signed delivery runs the agent (its reply comes back) and an unsigned one is 401
  — the connector's own gate, untouched;
* the boxed summary helper renders the endpoint and a signature that ACTUALLY verifies against
  this build's connector (and never prints the raw signing secret);
* the ``serve`` / ``worker`` parsers accept ``-f`` (and ``--agent``).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import create_app
from himmy.api.connector_inbound import INBOUND_AGENT_PATH_ENV
from himmy.api.route_introspection import collect_route_paths
from himmy.cli import commands
from himmy.cli.__main__ import build_parser
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)
from himmy.connectors.manage import _enabled_flag_name
from himmy.connectors.webhook import (
    DEFAULT_SIGNATURE_HEADER,
    WEBHOOK_SIGNING_SECRET,
    WebhookInboundConnector,
)

_TOUCHED = [
    WEBHOOK_SIGNING_SECRET,
    "HIMMY_WEBHOOK_ALLOWED_SOURCES",
    _enabled_flag_name("webhook", "inbound"),
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


def _serve_args(agent_path: str) -> argparse.Namespace:
    return argparse.Namespace(file=agent_path, agent=None)


def test_service_agent_path_prefers_flag_then_discovers(env: Path) -> None:
    yaml = str(env / "agent.yaml")
    assert commands._service_agent_path(_serve_args(yaml)) == yaml
    # --agent is an alias for -f.
    assert (
        commands._service_agent_path(argparse.Namespace(file=None, agent=yaml)) == yaml
    )


def test_enable_wires_the_same_env_mount_inbound_reads(env: Path) -> None:
    yaml = str(env / "agent.yaml")
    secret = commands._enable_inbound_webhook(yaml)
    assert secret  # a signing secret was ensured
    assert os.environ[INBOUND_AGENT_PATH_ENV] == yaml
    assert os.environ[_enabled_flag_name("webhook", "inbound")] == "1"
    # the sample source is allow-listed so the ready-to-paste curl is not default-denied
    assert os.environ["HIMMY_WEBHOOK_ALLOWED_SOURCES"] == commands._SERVICE_SAMPLE_SOURCE


def test_enable_never_clobbers_an_operator_secret_or_allowlist(env: Path) -> None:
    os.environ[WEBHOOK_SIGNING_SECRET] = "whsec_operator_chosen"
    os.environ["HIMMY_WEBHOOK_ALLOWED_SOURCES"] = "ci,prod"
    secret = commands._enable_inbound_webhook(str(env / "agent.yaml"))
    assert secret == "whsec_operator_chosen"  # honoured, not regenerated
    assert os.environ["HIMMY_WEBHOOK_ALLOWED_SOURCES"] == "ci,prod"  # untouched


def test_serve_wiring_mounts_webhook_signed_ok_unsigned_401(env: Path) -> None:
    """The serve front door mounts the agent endpoint; signed runs, unsigned is denied."""
    yaml = str(env / "agent.yaml")
    secret = commands._enable_inbound_webhook(yaml)  # what cmd_serve does before create_app
    app = create_app(bind_host="127.0.0.1")
    assert "/v1/connectors/webhook" in collect_route_paths(app)

    # Sign the EXACT sample payload the summary advertises, over the raw bytes.
    from himmy.connectors.webhook import sign_webhook_body

    body = json.dumps(
        {"source": commands._SERVICE_SAMPLE_SOURCE, "text": "hello"},
        separators=(",", ":"),
    ).encode()
    sig = sign_webhook_body(secret=secret, body=body)
    with TestClient(app) as client:
        ok = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={DEFAULT_SIGNATURE_HEADER: sig},
        )
        assert ok.status_code == 200
        data = ok.json()
        assert data["ok"] is True and data["handled"] is True
        assert "hello" in data["reply"]  # the stub agent actually ran

        unsigned = client.post("/v1/connectors/webhook", content=body)
        assert unsigned.status_code == 401


def test_summary_endpoint_signature_actually_verifies(env: Path) -> None:
    """The summary's ready-to-paste curl carries a signature the connector accepts."""
    secret = "whsec_" + "a" * 48
    summary = commands.render_service_summary(
        host="127.0.0.1",
        port=8000,
        agent_path=str(env / "agent.yaml"),
        signing_secret=secret,
        store_path=".himmy/storage.db",
    )
    # The raw secret is NEVER printed; only a valid signature for the sample body is.
    assert secret not in summary
    assert "/v1/connectors/webhook" in summary
    assert "/docs" in summary and "/readyz" in summary and "/metrics" in summary
    assert ".himmy/storage.db" in summary

    # Pull the signature out of the rendered curl and prove it verifies over the sample body.
    sig = next(
        line.split(f"{DEFAULT_SIGNATURE_HEADER}: ", 1)[1].split("'", 1)[0]
        for line in summary.splitlines()
        if DEFAULT_SIGNATURE_HEADER in line
    )
    body = json.dumps(
        {"source": commands._SERVICE_SAMPLE_SOURCE, "text": "hello"},
        separators=(",", ":"),
    ).encode()
    connector = WebhookInboundConnector(
        lambda sender, text: text,  # unused: we only exercise verify_webhook
        signing_secret=secret,
        allowed_sources=[commands._SERVICE_SAMPLE_SOURCE],
    )
    assert connector.verify_webhook(body, {DEFAULT_SIGNATURE_HEADER: sig}) is True


def test_worker_summary_has_store_no_http_and_hides_secret(env: Path) -> None:
    summary = commands.render_service_summary(
        host=None,
        port=None,
        agent_path=str(env / "agent.yaml"),
        signing_secret=None,
        store_path=".himmy/storage.db",
    )
    assert "no HTTP endpoint" in summary
    assert ".himmy/storage.db" in summary
    assert "http://" not in summary  # a worker exposes no URL


def test_serve_and_worker_parsers_accept_agent_flag() -> None:
    parser = build_parser()
    serve = parser.parse_args(["serve", "-f", "agent.yaml"])
    assert serve.file == "agent.yaml"
    serve_alias = parser.parse_args(["serve", "--agent", "a.yaml", "--port", "9001"])
    assert serve_alias.agent == "a.yaml" and serve_alias.port == 9001
    worker = parser.parse_args(["worker", "-f", "agent.yaml", "--no-scheduler"])
    assert worker.file == "agent.yaml" and worker.no_scheduler is True
