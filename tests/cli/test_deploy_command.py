"""``himmy deploy`` — the one-command front door over serve + worker.

``himmy deploy`` is a THIN, delightful wrapper over the EXISTING inbound machinery + worker
substrate; it must not relax their security posture. These tests prove, in-process and
offline:

* deploy wires the SAME signed webhook endpoint (a correctly-signed delivery runs the agent;
  an unsigned one is denied) — the connector's own default-deny gate, untouched;
* provider stamping is idempotent AND never clobbers an explicit ``provider``/``model``;
* the missing-credential preflight lists exactly the env keys a pack needs;
* the supervised serve+worker group shuts down cleanly (no leaked task) when either half
  finishes — no orphan worker;
* ``--docker`` emits a runnable Dockerfile; the deploy parser accepts the new flags.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import create_app
from himmy.api.connector_inbound import INBOUND_AGENT_PATH_ENV, INBOUND_PROVIDER_ENV
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
    sign_webhook_body,
)

_TOUCHED = [
    WEBHOOK_SIGNING_SECRET,
    "HIMMY_WEBHOOK_ALLOWED_SOURCES",
    _enabled_flag_name("webhook", "inbound"),
    INBOUND_AGENT_PATH_ENV,
    INBOUND_PROVIDER_ENV,
    "HIMMY_STUDIO_GUARD",
    "HIMMY_SMTP_HOST",
    "HIMMY_TELEGRAM_BOT_TOKEN",
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
    os.environ["HIMMY_STUDIO_GUARD"] = "0"
    yield tmp_path
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


# ------------------------------------------------------------------ endpoint wiring


def test_deploy_wiring_mounts_signed_endpoint_unsigned_denied(env: Path) -> None:
    """The deploy front door wires the agent endpoint: signed runs, unsigned is 401."""
    yaml = str(env / "agent.yaml")
    # What cmd_deploy does before create_app: wire the signed webhook at the agent.
    secret = commands._enable_inbound_webhook(yaml)
    assert secret
    app = create_app(bind_host="127.0.0.1")
    assert "/v1/connectors/webhook" in collect_route_paths(app)

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
        assert "hello" in data["reply"]

        unsigned = client.post("/v1/connectors/webhook", content=body)
        assert unsigned.status_code == 401


# ------------------------------------------------------------- provider stamping


def test_provider_stamp_is_idempotent_and_does_not_clobber_explicit(env: Path) -> None:
    """An explicit provider/model is never overwritten by the stub-stamp path."""
    yaml = str(env / "agent.yaml")  # declares provider: stub explicitly
    provider, model, hint = commands._deploy_resolve_provider(yaml)
    # provider: stub is EXPLICIT → left exactly as written, no install hint, no rewrite.
    assert provider == "stub"
    assert hint is None
    assert "provider: stub" in Path(yaml).read_text()


def test_provider_stamp_writes_detected_backend_when_default(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A default spec on the stub gets the best detected backend stamped in (idempotently)."""
    yaml = env / "agent.yaml"
    yaml.write_text("name: bot\ndescription: d\nmodel: default\n")  # no provider

    from himmy.cli.wizard import ProviderChoice

    fake = ProviderChoice(key="ollama", label="ollama", model="llama3.2")
    monkeypatch.setattr(
        "himmy.cli.wizard.detect_provider_choices", lambda: [fake]
    )
    provider, model, hint = commands._deploy_resolve_provider(str(yaml))
    assert provider == "ollama" and model == "llama3.2" and hint is None
    text = yaml.read_text()
    assert "provider: ollama" in text and "llama3.2" in text

    # Idempotent: a second resolve sees the now-explicit provider and does not re-stamp.
    provider2, _model2, hint2 = commands._deploy_resolve_provider(str(yaml))
    assert provider2 == "ollama" and hint2 is None


def test_provider_stamp_no_backend_returns_install_hint(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With nothing detected, the spec stays on the stub and an install line is returned."""
    yaml = env / "agent.yaml"
    yaml.write_text("name: bot\ndescription: d\nmodel: default\n")

    from himmy.cli.wizard import ProviderChoice

    stub_only = ProviderChoice(key="stub", label="stub")
    monkeypatch.setattr(
        "himmy.cli.wizard.detect_provider_choices", lambda: [stub_only]
    )
    provider, _model, hint = commands._deploy_resolve_provider(str(yaml))
    assert provider is None
    assert hint is not None and "stub" in hint
    # No backend was written — the file stays provider-less.
    assert "provider:" not in yaml.read_text()


# ---------------------------------------------------------------- cred preflight


def test_preflight_lists_missing_pack_credentials(env: Path) -> None:
    """A comms+telegram agent with no creds set reports exactly the missing env keys."""
    from himmy.runtime import from_spec

    yaml = env / "agent.yaml"
    yaml.write_text(
        "name: bot\ndescription: d\nprovider: stub\ntool_packs: [comms, telegram, web]\n"
    )
    spec = from_spec.load_spec_file(str(yaml))
    missing = commands._preflight_pack_credentials(spec)
    assert "HIMMY_SMTP_HOST" in missing
    assert "HIMMY_TELEGRAM_BOT_TOKEN" in missing
    # 'web' is keyless — it never contributes a required key.
    assert all("WEB" not in k for k in missing)


def test_preflight_satisfied_when_key_present(env: Path) -> None:
    """A configured credential drops out of the missing list."""
    from himmy.runtime import from_spec

    os.environ["HIMMY_SMTP_HOST"] = "smtp.example.com"
    yaml = env / "agent.yaml"
    yaml.write_text("name: bot\ndescription: d\nprovider: stub\ntool_packs: [comms]\n")
    spec = from_spec.load_spec_file(str(yaml))
    assert commands._preflight_pack_credentials(spec) == []


# ------------------------------------------------------- supervised shutdown


def test_serve_and_worker_group_shuts_down_cleanly_no_leak(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When one half finishes, the group cancels the other — no orphaned task survives."""

    async def _fake_server_serve(self: object) -> None:  # noqa: ANN001
        # Simulate the server exiting quickly (e.g. told to stop) so the group tears down.
        return None

    async def _fake_worker(**_kwargs: object) -> None:
        # A worker that would block forever if not cancelled by the group shutdown.
        await asyncio.Event().wait()

    monkeypatch.setattr("uvicorn.Server.serve", _fake_server_serve)
    monkeypatch.setattr(commands, "_run_worker", _fake_worker)

    class _App:
        pass

    async def _drive() -> set[asyncio.Task]:
        await commands._serve_and_worker(
            _App(), "127.0.0.1", 8000, run_scheduler=True, run_dispatcher=True
        )
        # No himmy-deploy-* task should still be alive after the group returns.
        return {
            t
            for t in asyncio.all_tasks()
            if (t.get_name() or "").startswith("himmy-deploy-")
            and t is not asyncio.current_task()
        }

    leaked = asyncio.run(_drive())
    assert leaked == set()


def test_serve_and_worker_reraises_server_boot_error(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A server that fails to bind surfaces its error to the caller (→ port-in-use message)."""

    async def _boom_serve(self: object) -> None:  # noqa: ANN001
        raise OSError(48, "address already in use")

    async def _fast_worker(**_kwargs: object) -> None:
        await asyncio.Event().wait()

    monkeypatch.setattr("uvicorn.Server.serve", _boom_serve)
    monkeypatch.setattr(commands, "_run_worker", _fast_worker)

    class _App:
        pass

    with pytest.raises(OSError):
        asyncio.run(
            commands._serve_and_worker(
                _App(), "127.0.0.1", 8000, run_scheduler=True, run_dispatcher=True
            )
        )


# ------------------------------------------------------------------- docker + parser


def test_docker_emits_runnable_dockerfile(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = argparse.Namespace(
        file=str(env / "agent.yaml"), agent=None, docker=True, channel="http"
    )
    assert commands.cmd_deploy(args) == 0
    out = capsys.readouterr().out
    assert "FROM python:3.12-slim" in out
    assert "pip install" in out and "himmy[api]" in out
    assert 'CMD ["himmy", "deploy"' in out
    assert "agent.yaml" in out


def test_deploy_parser_accepts_new_flags() -> None:
    parser = build_parser()
    ns = parser.parse_args(
        ["deploy", "-f", "agent.yaml", "--port", "9100", "--channel", "telegram"]
    )
    assert ns.file == "agent.yaml"
    assert ns.port == 9100
    assert ns.channel == "telegram"
    assert ns.host == "127.0.0.1"

    shared = parser.parse_args(["deploy", "--share", "--host", "0.0.0.0"])  # noqa: S104
    assert shared.share is True and shared.host == "0.0.0.0"  # noqa: S104
