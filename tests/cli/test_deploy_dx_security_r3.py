"""deploy-DX security regression suite (red-team round 3).

Each test pins one confirmed defect in the one-command deploy surface so it can never
silently regress:

* the Studio DNS-rebind/CSRF guard used to 403 EVERY genuine signed webhook delivery
  (public Host, no Origin/Referer) BEFORE signature verification — steering operators to
  disable the whole guard. Signed ``/v1/connectors/*`` paths are now carved out of the
  browser-origin half of the guard (still HMAC + timestamp + allowlist authenticated);
* the ``render_service_summary`` "try it" curl (no Origin/Referer) now succeeds against the
  DEFAULT (guard-ON) posture — proven via the carve-out above;
* a non-share off-loopback deploy/serve now defaults a separate ``HIMMY_METRICS_TOKEN`` so
  the single shared apikey can't scrape ``/metrics`` (parity with ``--share``);
* a uvicorn port-in-use failure (real uvicorn raises ``SystemExit``, not ``OSError``) is
  normalised so the worker is not orphaned and a ``--share`` operator key is revoked;
* the provider auto-stamp is line-anchored so a comment containing ``model: default`` is
  never rewritten in place of the real field.
"""

from __future__ import annotations

import argparse
import json
import os
import time as _time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import create_app
from himmy.api.connector_inbound import INBOUND_AGENT_PATH_ENV, INBOUND_PROVIDER_ENV
from himmy.api.route_introspection import collect_route_paths
from himmy.cli import commands
from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)
from himmy.connectors.manage import _enabled_flag_name
from himmy.connectors.webhook import (
    DEFAULT_SIGNATURE_HEADER,
    DEFAULT_TIMESTAMP_HEADER,
    WEBHOOK_SIGNING_SECRET,
    sign_webhook_body,
)

_TOUCHED = [
    WEBHOOK_SIGNING_SECRET,
    "HIMMY_WEBHOOK_ALLOWED_SOURCES",
    "HIMMY_WEBHOOK_REQUIRE_TIMESTAMP",
    _enabled_flag_name("webhook", "inbound"),
    INBOUND_AGENT_PATH_ENV,
    INBOUND_PROVIDER_ENV,
    "HIMMY_STUDIO_GUARD",
    "HIMMY_STUDIO_ALLOW_HOSTS",
    "HIMMY_METRICS_TOKEN",
]


@pytest.fixture
def env(tmp_path: Path):
    """Writable secrets backend + an inbound agent.yaml; clean env around the test.

    NOTE: unlike the sibling mount suite, this fixture DELIBERATELY does NOT set
    ``HIMMY_STUDIO_GUARD=0`` — the whole point of the round-3 fix is that the signed webhook
    works with the guard at its DEFAULT (ON) posture.
    """
    configure_secrets(
        ChainSecretProvider([FileSecrets(tmp_path / "secrets"), EnvSecrets()])
    )
    for name in _TOUCHED:
        os.environ.pop(name, None)
    agent_yaml = tmp_path / "agent.yaml"
    agent_yaml.write_text(
        "name: inbound-bot\ndescription: inbound test agent\nprovider: stub\n"
    )
    yield tmp_path
    configure_secrets(None)
    for name in _TOUCHED:
        os.environ.pop(name, None)


def _signed(secret: str):
    body = json.dumps(
        {"source": commands._SERVICE_SAMPLE_SOURCE, "text": "hello"},
        separators=(",", ":"),
    ).encode()
    ts = str(int(_time.time()))
    sig = sign_webhook_body(secret=secret, body=body, timestamp=ts)
    return body, {DEFAULT_SIGNATURE_HEADER: sig, DEFAULT_TIMESTAMP_HEADER: ts}


# --------------------------------------------------------------- guard carve-out (bug 1/2)


def test_signed_webhook_passes_guard_with_public_host_and_no_origin(env: Path) -> None:
    """A genuine external delivery (public Host, NO Origin/Referer) reaches the connector.

    This is the round-3 regression: before the carve-out the Studio guard (default ON) 403'd
    every such request with 'host not allowed' / 'cross-origin blocked' BEFORE signature
    verification. Now the signed delivery runs the agent even though the Host is public and
    there is no Origin header — because ``/v1/connectors/*`` is exempt from the browser-origin
    half of the guard (its HMAC + timestamp + allowlist gate still authorizes it).
    """
    yaml = str(env / "agent.yaml")
    secret = commands._enable_inbound_webhook(yaml)
    # Guard at DEFAULT posture (fixture does NOT disable it) — assert it really is on.
    assert "HIMMY_STUDIO_GUARD" not in os.environ
    app = create_app(bind_host="127.0.0.1")
    assert "/v1/connectors/webhook" in collect_route_paths(app)

    body, headers = _signed(secret)
    with TestClient(app) as client:
        # Public Host (like a k8s ingress / tunnel), NO Origin, NO Referer.
        resp = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={**headers, "host": "agent.example.com"},
        )
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["ok"] is True and data["handled"] is True

        # And a loopback POST with no Origin (the exact `render_service_summary` curl shape)
        # also passes now instead of 403 'cross-origin blocked'.
        body2, headers2 = _signed(secret)
        loopback = client.post(
            "/v1/connectors/webhook",
            content=body2,
            headers={**headers2, "host": "127.0.0.1"},
        )
        assert loopback.status_code == 200, loopback.text


def test_guard_still_protects_non_connector_v1_with_public_host(env: Path) -> None:
    """The carve-out is narrow: a NON-connector /v1 route is still guarded (default deny).

    A public Host on e.g. ``/v1/runs`` must STILL be 403'd — the carve-out must not leak the
    DNS-rebinding/CSRF protection off the rest of the tenant API.
    """
    app = create_app(bind_host="127.0.0.1")
    with TestClient(app) as client:
        resp = client.get("/v1/runs", headers={"host": "agent.example.com"})
        assert resp.status_code == 403
        assert resp.json()["detail"] == "host not allowed"


def test_summary_curl_shape_verifies_under_default_guard(env: Path) -> None:
    """The advertised 'try it' curl (timestamp + signature headers, NO Origin) works now.

    Proves finding #2 is fixed end-to-end: the summary teaches a signed curl that succeeds
    against the shipped (guard-ON) posture, so users are never pushed to disable the guard.
    """
    yaml = str(env / "agent.yaml")
    secret = commands._enable_inbound_webhook(yaml)
    app = create_app(bind_host="127.0.0.1")
    body, headers = _signed(secret)  # exactly the headers the rendered curl sets
    with TestClient(app) as client:
        resp = client.post("/v1/connectors/webhook", content=body, headers=headers)
        assert resp.status_code == 200, resp.text


# --------------------------------------------------------------- metrics token (bug 3)


def test_off_loopback_deploy_defaults_a_metrics_token(env: Path) -> None:
    """An off-loopback bind provisions a separate HIMMY_METRICS_TOKEN (parity with --share)."""
    assert "HIMMY_METRICS_TOKEN" not in os.environ
    commands._deploy_provision_metrics_token("0.0.0.0")  # noqa: S104
    token = os.environ.get("HIMMY_METRICS_TOKEN")
    assert token and len(token) >= 20


def test_loopback_deploy_does_not_set_a_metrics_token(env: Path) -> None:
    """On a loopback bind only the operator reaches /metrics — no token minted (no churn)."""
    commands._deploy_provision_metrics_token("127.0.0.1")
    assert "HIMMY_METRICS_TOKEN" not in os.environ


def test_metrics_token_never_clobbers_operator_value(env: Path) -> None:
    """An operator/template-set token stands (setdefault, not overwrite)."""
    os.environ["HIMMY_METRICS_TOKEN"] = "operator-chosen"
    commands._deploy_provision_metrics_token("0.0.0.0")  # noqa: S104
    assert os.environ["HIMMY_METRICS_TOKEN"] == "operator-chosen"


# --------------------------------------------------------------- SystemExit port-in-use (bug 4)


def test_serve_and_worker_normalises_systemexit_to_oserror() -> None:
    """A uvicorn bind failure (SystemExit) becomes OSError(EADDRINUSE) — not a raw SystemExit.

    Real uvicorn calls ``sys.exit(1)`` inside ``Server.startup`` when the port is in use, which
    raises ``SystemExit`` (a BaseException matching NEITHER ``except OSError`` NOR
    ``except Exception`` in cmd_deploy). ``_serve_and_worker`` must (a) not let SystemExit
    escape the cleanup suppress and orphan the worker, and (b) re-surface it as an
    OSError(EADDRINUSE) so the port-in-use branch fires and the share key is revoked.
    """
    import asyncio
    import errno

    async def _boom_server() -> None:
        raise SystemExit(1)

    async def _idle_worker() -> None:
        # Behaves like the real worker: runs until cancelled at shutdown.
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    orig_uvicorn = __import__("uvicorn")

    class _FakeServer:
        def __init__(self, config: object) -> None:
            self.should_exit = False
            self.install_signal_handlers = lambda: None

        async def serve(self) -> None:
            await _boom_server()

    class _FakeConfig:
        def __init__(self, *a: object, **k: object) -> None:
            pass

    # Patch uvicorn.Server/Config used inside _serve_and_worker, and _run_worker to an idler.
    orig_server = orig_uvicorn.Server
    orig_config = orig_uvicorn.Config
    orig_run_worker = commands._run_worker
    orig_uvicorn.Server = _FakeServer  # type: ignore[assignment,misc]
    orig_uvicorn.Config = _FakeConfig  # type: ignore[assignment,misc]
    commands._run_worker = lambda **_k: _idle_worker()  # type: ignore[assignment]
    try:
        with pytest.raises(OSError) as excinfo:
            asyncio.run(
                commands._serve_and_worker(
                    object(), "0.0.0.0", 8000, run_scheduler=True, run_dispatcher=True  # noqa: S104
                )
            )
        assert excinfo.value.errno == errno.EADDRINUSE
    finally:
        orig_uvicorn.Server = orig_server  # type: ignore[assignment,misc]
        orig_uvicorn.Config = orig_config  # type: ignore[assignment,misc]
        commands._run_worker = orig_run_worker  # type: ignore[assignment]


def test_cmd_deploy_revokes_share_key_on_systemexit(monkeypatch, tmp_path: Path) -> None:
    """A --share deploy that fails to bind (SystemExit path) revokes the minted operator key.

    Drives cmd_deploy with a stubbed _serve_and_worker that raises SystemExit (belt-and-
    suspenders path) and asserts the share key is revoked + a non-zero exit is returned, so a
    failed --share never leaves a live 7-day operator credential on disk.
    """
    revoked: list[str | None] = []
    monkeypatch.setattr(commands, "_revoke_share_key", lambda s: revoked.append(s))

    async def _raise_systemexit(*a: object, **k: object) -> None:
        raise SystemExit(1)

    monkeypatch.setattr(commands, "_serve_and_worker", _raise_systemexit)
    # Neutralise the pre-boot machinery so we reach the asyncio.run branch deterministically.
    monkeypatch.setattr(commands, "_service_agent_path", lambda a: str(tmp_path / "a.yaml"))
    (tmp_path / "a.yaml").write_text("name: x\ndescription: y\nprovider: stub\n")
    # cmd_deploy does `from himmy.cli.agents import _findings_for` locally — patch the source.
    from himmy.cli import agents as _agents

    monkeypatch.setattr(_agents, "_findings_for", lambda p: [])
    monkeypatch.setattr(commands, "_preflight_pack_credentials", lambda s: [])
    monkeypatch.setattr(commands, "_deploy_resolve_provider", lambda p: ("stub", "default", None))
    monkeypatch.setattr(
        commands, "_deploy_configure_share_auth", lambda host: (True, "himmy_sharekey")
    )
    monkeypatch.setattr(commands, "_stamp_inbound_provider", lambda a: None)
    monkeypatch.setattr(commands, "_enable_inbound_webhook", lambda p: "whsec_x")
    monkeypatch.setattr(commands, "_materialize_api_keys_file", lambda: None)
    monkeypatch.setattr(commands, "_deploy_provision_metrics_token", lambda h: None)
    # cmd_deploy does `from himmy.api.app import create_app` locally, so patch the source module.
    monkeypatch.setattr("himmy.api.app.create_app", lambda bind_host: object())
    monkeypatch.setattr(commands, "render_service_summary", lambda **k: "")
    monkeypatch.setattr(commands, "render_share_tunnel", lambda **k: "", raising=False)
    monkeypatch.setattr(commands, "_durable_store_path", lambda: ".himmy/storage.db")

    args = argparse.Namespace(
        channel="http",
        docker=False,
        file=str(tmp_path / "a.yaml"),
        agent=None,
        host="0.0.0.0",  # noqa: S104
        port=8000,
        share=True,
        provider=None,
    )
    rc = commands.cmd_deploy(args)
    assert rc == 1
    assert revoked == ["himmy_sharekey"]


# --------------------------------------------------------------- provider stamp anchor (bug 5)


def test_textual_stamp_is_line_anchored_not_first_substring() -> None:
    """A comment containing 'model: default' is NOT rewritten in place of the real field."""
    text = (
        "name: bot\n"
        "# e.g. model: default runs the offline stub\n"
        "# provider: claude-cli\n"
        "model: default\n"
    )
    stamped = commands._textual_stamp_provider(text, provider="ollama", model="llama3.2")
    assert stamped is not None
    # The teaching comment is preserved verbatim (NOT mangled).
    assert "# e.g. model: default runs the offline stub" in stamped
    # The REAL top-level field is the one that got stamped.
    assert "\nmodel: llama3.2\n" in stamped
    assert "provider: ollama" in stamped
    # And there is no leftover real `model: default` field.
    assert "\nmodel: default\n" not in stamped
