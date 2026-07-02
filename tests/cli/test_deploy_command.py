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
    "HIMMY_WEBHOOK_REQUIRE_TIMESTAMP",
    _enabled_flag_name("webhook", "inbound"),
    INBOUND_AGENT_PATH_ENV,
    INBOUND_PROVIDER_ENV,
    "HIMMY_STUDIO_GUARD",
    "HIMMY_SMTP_HOST",
    "HIMMY_TELEGRAM_BOT_TOKEN",
    "HIMMY_AUTH_MODE",
    "HIMMY_API_KEYS_FILE",
    "HIMMY_METRICS_TOKEN",
    commands.API_KEYS_JSON_ENV,
    commands.API_KEY_ENV,
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

    import time as _time

    from himmy.connectors.webhook import DEFAULT_TIMESTAMP_HEADER

    body = json.dumps(
        {"source": commands._SERVICE_SAMPLE_SOURCE, "text": "hello"},
        separators=(",", ":"),
    ).encode()
    ts = str(int(_time.time()))
    sig = sign_webhook_body(secret=secret, body=body, timestamp=ts)
    with TestClient(app) as client:
        ok = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={DEFAULT_SIGNATURE_HEADER: sig, DEFAULT_TIMESTAMP_HEADER: ts},
        )
        assert ok.status_code == 200
        data = ok.json()
        assert data["ok"] is True and data["handled"] is True
        assert "hello" in data["reply"]

        unsigned = client.post("/v1/connectors/webhook", content=body)
        assert unsigned.status_code == 401

        # A timestamp-less body-only signature is refused by the front door (replay guard on).
        body_only_sig = sign_webhook_body(secret=secret, body=body)
        replay = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={DEFAULT_SIGNATURE_HEADER: body_only_sig},
        )
        assert replay.status_code == 401


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
    # 3-line agent Dockerfile: FROM the published runtime image (no build-time pip install),
    # COPY the spec, CMD `himmy deploy`.
    assert f"FROM {commands.GHCR_IMAGE}:" in out
    assert "pip install" not in out
    assert 'CMD ["himmy", "deploy"' in out
    assert "COPY agent.yaml /app/agent.yaml" in out


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


# ---------------------------------------------------- off-loopback fail-closed DX


def test_deploy_off_loopback_no_share_prints_share_hint_exit_2(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`himmy deploy --host 0.0.0.0` with no --share fails closed WITH the friendly hint.

    create_app raises HimmyError (subclasses Exception, NOT RuntimeError) — the except must
    catch HimmyError or the guidance never fires and a raw traceback leaks. Assert exit 2 and
    the `himmy deploy --share` hint, not a traceback.
    """
    args = argparse.Namespace(
        file=str(env / "agent.yaml"),
        agent=None,
        docker=False,
        channel="http",
        host="0.0.0.0",  # noqa: S104
        port=8000,
        share=False,
    )
    assert commands.cmd_deploy(args) == 2
    err = capsys.readouterr().err
    assert "himmy deploy --share" in err
    assert "off-loopback" in err
    # The server never bound and no traceback surfaced.
    assert "Traceback" not in err


def test_deploy_off_loopback_share_is_accepted(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--share` off-loopback mints a key + enables auth so create_app is accepted (no refusal).

    We stop before actually binding (monkeypatch the supervised group) — the point is that
    the fail-closed posture passes because --share minted an apikey record + set apikey mode.
    """
    ran: dict[str, object] = {}

    async def _fake_group(app: object, host: str, port: int, **kwargs: object) -> None:
        ran["host"] = host
        ran["port"] = port

    monkeypatch.setattr(commands, "_serve_and_worker", _fake_group)
    monkeypatch.chdir(env)  # keys file + store land in the tmp dir

    args = argparse.Namespace(
        file=str(env / "agent.yaml"),
        agent=None,
        docker=False,
        channel="http",
        host="0.0.0.0",  # noqa: S104
        port=8000,
        share=True,
    )
    assert commands.cmd_deploy(args) == 0
    assert ran == {"host": "0.0.0.0", "port": 8000}  # noqa: S104
    out = capsys.readouterr()
    # --share minted a key + enabled auth; the summary curl carries the apikey header.
    assert os.environ.get("HIMMY_AUTH_MODE") == "apikey"
    assert "x-himmy-internal-key" in (out.err + out.out)


def test_share_summary_curl_carries_apikey_and_succeeds(env: Path) -> None:
    """In --share (apikey) mode the printed sample curl works: signature + apikey both pass.

    The finding: apikey mode 403s any request without the key, but the sample curl only
    carried the signature. The summary must thread the apikey (x-himmy-internal-key) so a
    paste succeeds. We build the apikey-mode app and replay the summary's headers.
    """
    import secrets as _secrets

    from himmy.api.auth.apikey import DEFAULT_HEADER as APIKEY_HEADER
    from himmy.cli.apikey_cmd import _fingerprint, _load_keys, _write_json_0600

    yaml = str(env / "agent.yaml")
    secret = commands._enable_inbound_webhook(yaml)

    # Mint an apikey record + turn on apikey mode (what --share does).
    keys_path = env / "api_keys.json"
    apikey = f"himmy_{_secrets.token_urlsafe(16)}"
    keys = _load_keys(keys_path)
    keys[apikey] = {
        "subject": f"apikey:{_fingerprint(apikey)}",
        "tenant_ids": [],
        "roles": [],
        "all_tenants": True,
        "disabled": False,
    }
    _write_json_0600(keys_path, keys)
    os.environ["HIMMY_API_KEYS_FILE"] = str(keys_path)
    os.environ["HIMMY_AUTH_MODE"] = "apikey"
    try:
        import time as _time

        from himmy.connectors.webhook import DEFAULT_TIMESTAMP_HEADER

        app = create_app(bind_host="127.0.0.1")
        body = json.dumps(
            {"source": commands._SERVICE_SAMPLE_SOURCE, "text": "hello"},
            separators=(",", ":"),
        ).encode()
        ts = str(int(_time.time()))
        sig = sign_webhook_body(secret=secret, body=body, timestamp=ts)
        with TestClient(app) as client:
            # signature alone (what the OLD curl sent) is rejected in apikey mode.
            denied = client.post(
                "/v1/connectors/webhook",
                content=body,
                headers={DEFAULT_SIGNATURE_HEADER: sig, DEFAULT_TIMESTAMP_HEADER: ts},
            )
            assert denied.status_code in (401, 403)
            # signature + timestamp + apikey (what the NEW summary prints) succeeds.
            ok = client.post(
                "/v1/connectors/webhook",
                content=body,
                headers={
                    DEFAULT_SIGNATURE_HEADER: sig,
                    DEFAULT_TIMESTAMP_HEADER: ts,
                    APIKEY_HEADER: apikey,
                },
            )
            assert ok.status_code == 200

        # The rendered summary references the apikey via $HIMMY_SHARE_KEY (NOT the raw key —
        # baking a live admin credential into a copy-paste command would leak it into shell
        # history). The header name is present; the literal key is NOT.
        summary = commands.render_service_summary(
            host="0.0.0.0",  # noqa: S104
            port=8000,
            agent_path=yaml,
            signing_secret=secret,
            store_path="x",
            apikey=apikey,
        )
        assert APIKEY_HEADER in summary
        assert "$HIMMY_SHARE_KEY" in summary
        assert apikey not in summary
    finally:
        os.environ.pop("HIMMY_API_KEYS_FILE", None)
        os.environ.pop("HIMMY_AUTH_MODE", None)


def test_docker_cmd_is_fail_closed_not_open_by_default(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The emitted Dockerfile is fail-closed: the ACTIVE CMD binds loopback, never 0.0.0.0.

    `himmy init` now drops this recipe unprompted, so it must not ship an open, unauthenticated
    admin surface by default. The live CMD binds 127.0.0.1 (reachable to the in-container
    healthcheck, NOT through `-p`), and the unauthenticated-proxy opt-in
    (HIMMY_ALLOW_UNAUTHENTICATED + `--host 0.0.0.0`) appears only as COMMENTED guidance.
    """
    args = argparse.Namespace(
        file=str(env / "agent.yaml"), agent=None, docker=True, channel="http"
    )
    assert commands.cmd_deploy(args) == 0
    out = capsys.readouterr().out
    active = "\n".join(ln for ln in out.splitlines() if ln and not ln.startswith("#"))
    # Live CMD binds loopback; no active line binds all interfaces or bakes the unauth opt-in.
    assert '"--host", "127.0.0.1"' in active
    assert "0.0.0.0" not in active  # noqa: S104 - asserting absence, not binding
    assert "HIMMY_ALLOW_UNAUTHENTICATED" not in active
    # The opt-in still exists as documented (commented) trusted-proxy guidance.
    assert "trusted proxy" in out
    assert "HIMMY_ALLOW_UNAUTHENTICATED" in out


# --------------------------------------------------------- --share tunnel (WP #9)


def test_share_provisions_auth_before_printing_any_public_command(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--share` (even on the DEFAULT loopback bind) mints auth BEFORE printing the tunnel.

    A tunnel makes the loopback port public, so the security-critical ordering is: auth ON
    first, THEN the copy-paste `cloudflared`/`ngrok` command. We assert both — the apikey was
    provisioned AND the printed tunnel command references the local port — never a public
    command for an unauthenticated endpoint.
    """
    ran: dict[str, object] = {}

    async def _fake_group(app: object, host: str, port: int, **kwargs: object) -> None:
        # By the time we reach the supervised group, auth MUST already be configured.
        ran["auth_mode"] = os.environ.get("HIMMY_AUTH_MODE")

    monkeypatch.setattr(commands, "_serve_and_worker", _fake_group)
    monkeypatch.chdir(env)

    args = argparse.Namespace(
        file=str(env / "agent.yaml"),
        agent=None,
        docker=False,
        channel="http",
        host="127.0.0.1",  # the DEFAULT loopback bind — tunnel is the exposure
        port=8000,
        share=True,
    )
    assert commands.cmd_deploy(args) == 0
    # Auth was minted + enabled BEFORE the process group booted (and stays on after).
    assert ran["auth_mode"] == "apikey"
    assert os.environ.get("HIMMY_AUTH_MODE") == "apikey"
    err = capsys.readouterr().err
    # The public tunnel command is printed, targeting the local port.
    assert "cloudflared tunnel --url http://127.0.0.1:8000" in err
    assert "ngrok http 8000" in err
    # And the minted key was shown once (the credential the friend needs).
    assert "shown ONCE" in err
    # The tunnel block explicitly states auth is ON before offering the public command.
    assert "auth is ON" in err


def test_share_refuses_when_auth_cannot_be_provisioned(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If auth can't be minted, `--share` REFUSES — it never exposes an unauthenticated port.

    We force the keys-file write to fail; `--share` must abort with exit 2, print a refusal,
    and NOT boot the service or print any tunnel command.
    """
    booted = {"ran": False}

    async def _fake_group(*a: object, **k: object) -> None:  # pragma: no cover - must not run
        booted["ran"] = True

    def _boom(*a: object, **k: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(commands, "_serve_and_worker", _fake_group)
    monkeypatch.setattr("himmy.cli.apikey_cmd._write_json_0600", _boom)
    monkeypatch.chdir(env)

    args = argparse.Namespace(
        file=str(env / "agent.yaml"),
        agent=None,
        docker=False,
        channel="http",
        host="127.0.0.1",
        port=8000,
        share=True,
    )
    assert commands.cmd_deploy(args) == 2
    assert booted["ran"] is False
    out = capsys.readouterr()
    combined = out.err + out.out
    assert "refuses to expose an unauthenticated endpoint" in combined
    # No public tunnel command was printed.
    assert "cloudflared" not in combined
    assert "ngrok" not in combined


def test_share_tunnel_block_only_after_auth() -> None:
    """`render_share_tunnel` states auth is required + targets the exact host:port."""
    block = commands.render_share_tunnel(host="127.0.0.1", port=9123)
    assert "cloudflared tunnel --url http://127.0.0.1:9123" in block
    assert "ngrok http 9123" in block
    assert "needs the API key" in block


# --------------------------------------------------------- cloud templates (WP #9)


def test_cloud_templates_exist_and_reference_image_and_agent() -> None:
    """The deploy/cloud/ templates are valid + wire the ghcr image + an agent path.

    fly.toml, render.yaml, railway.json must each be parseable AND reference the published
    runtime image (directly or via the thin agent Dockerfile that builds FROM it) plus the
    agent.yaml path — proving they deploy the USER's agent, not empty Studio.
    """
    import tomllib

    import yaml as _yaml

    root = Path(commands.__file__).resolve().parents[2]
    cloud = root / "deploy" / "cloud"
    fly = cloud / "fly.toml"
    render = cloud / "render.yaml"
    railway = cloud / "railway.json"
    assert fly.is_file() and render.is_file() and railway.is_file()

    # fly.toml: valid TOML, builds the agent Dockerfile, runs api + worker on the agent.
    fly_data = tomllib.loads(fly.read_text())
    assert fly_data["build"]["dockerfile"] == "Dockerfile"
    procs = fly_data["processes"]
    assert "/app/agent.yaml" in procs["api"]
    assert "/app/agent.yaml" in procs["worker"]
    # Auth is enabled before the public service accepts traffic.
    assert fly_data["env"]["HIMMY_AUTH_MODE"] == "apikey"

    # render.yaml: valid YAML, api + worker services on the agent, image via Dockerfile.
    render_data = _yaml.safe_load(render.read_text())
    svcs = {s["name"]: s for s in render_data["services"]}
    assert "himmy-agent-api" in svcs and "himmy-agent-worker" in svcs
    assert "/app/agent.yaml" in svcs["himmy-agent-api"]["dockerCommand"]
    assert svcs["himmy-agent-api"]["dockerfilePath"] == "./Dockerfile"
    auth = {
        e["key"]: e.get("value")
        for e in svcs["himmy-agent-api"]["envVars"]
    }
    assert auth["HIMMY_AUTH_MODE"] == "apikey"

    # railway.json: valid JSON, builds the Dockerfile, serves the agent.
    railway_data = json.loads(railway.read_text())
    assert railway_data["build"]["builder"] == "DOCKERFILE"
    assert "/app/agent.yaml" in railway_data["deploy"]["startCommand"]
    # The published image + agent path are referenced in the guidance comment.
    comment = "\n".join(railway_data["_comment"])
    assert "ghcr.io/nlethetech/himmy" in comment
    assert "agent.yaml" in comment


# ----------------------------------------------------- cloud boot-time key seeding (WP #9)


def test_cloud_template_env_boots_authenticated_on_0000_after_seeding(env: Path) -> None:
    """The cloud-template env (apikey mode, NO keys file) crash-loops until the seed secret.

    This is the real happy path the templates promise, proven end to end:
      1. under the template env alone (HIMMY_AUTH_MODE=apikey + a keys-file path that does not
         exist yet), building the app on 0.0.0.0 FAILS — the file is genuinely required;
      2. setting the documented one platform secret (HIMMY_API_KEY) + running the boot seeder
         materializes the keys file, so create_app(bind_host='0.0.0.0') now BOOTS authenticated.
    Asserting the YAML parses (the weaker sibling test) never catches the crash-loop; this does.
    """
    keys_file = env / ".himmy" / "api_keys.json"
    os.environ["HIMMY_AUTH_MODE"] = "apikey"
    os.environ["HIMMY_API_KEYS_FILE"] = str(keys_file)

    # 1) template env with no seed + no file: create_app on 0.0.0.0 fail-closes (file required).
    assert not keys_file.exists()
    with pytest.raises(Exception):  # noqa: B017 - FileNotFoundError/HimmyError, both are refusals
        create_app(bind_host="0.0.0.0")  # noqa: S104

    # 2) the documented one-secret step: set HIMMY_API_KEY, seed on boot, now it boots.
    apikey = "himmy_seeded_boot_key"
    os.environ[commands.API_KEY_ENV] = apikey
    commands._materialize_api_keys_file()
    assert keys_file.exists()
    # File is 0600 and holds exactly the seeded record (the raw key is the map key).
    seeded = json.loads(keys_file.read_text())
    assert apikey in seeded and seeded[apikey]["all_tenants"] is True

    import time as _time

    from himmy.connectors.webhook import DEFAULT_TIMESTAMP_HEADER

    body = json.dumps(
        {"source": commands._SERVICE_SAMPLE_SOURCE, "text": "hi"}, separators=(",", ":")
    ).encode()
    # Wire the signed webhook so we can prove the seeded key gates a real request.
    secret = commands._enable_inbound_webhook(str(env / "agent.yaml"))
    ts = str(int(_time.time()))
    sig = sign_webhook_body(secret=secret, body=body, timestamp=ts)
    from himmy.api.auth.apikey import DEFAULT_HEADER as APIKEY_HEADER

    # create_app now BOOTS on 0.0.0.0 (was a refusal before seeding) — proven authenticated.
    with TestClient(create_app(bind_host="0.0.0.0")) as client:  # noqa: S104
        # signature alone (no apikey) is refused; signature + seeded key succeeds.
        denied = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={DEFAULT_SIGNATURE_HEADER: sig, DEFAULT_TIMESTAMP_HEADER: ts},
        )
        assert denied.status_code in (401, 403)
        ok = client.post(
            "/v1/connectors/webhook",
            content=body,
            headers={
                DEFAULT_SIGNATURE_HEADER: sig,
                DEFAULT_TIMESTAMP_HEADER: ts,
                APIKEY_HEADER: apikey,
            },
        )
        assert ok.status_code == 200


def test_seed_from_full_json_secret_materializes_verbatim(env: Path) -> None:
    """HIMMY_API_KEYS_JSON (literal keys-file contents) is written verbatim + boots on 0.0.0.0."""
    keys_file = env / ".himmy" / "api_keys.json"
    os.environ["HIMMY_AUTH_MODE"] = "apikey"
    os.environ["HIMMY_API_KEYS_FILE"] = str(keys_file)
    record = {"himmy_json_key": {"subject": "ops", "all_tenants": True, "roles": ["admin"]}}
    os.environ[commands.API_KEYS_JSON_ENV] = json.dumps(record)

    commands._materialize_api_keys_file()
    assert json.loads(keys_file.read_text()) == record
    # The multi-tenant apikey posture boots on 0.0.0.0 with the seeded per-tenant record.
    create_app(bind_host="0.0.0.0")  # noqa: S104 - must not raise


def test_seed_is_idempotent_and_does_not_clobber_existing_keys_file(env: Path) -> None:
    """An existing keys file is never overwritten by the boot seeder (respects `apikey mint`)."""
    from himmy.cli.apikey_cmd import _write_json_0600

    keys_file = env / ".himmy" / "api_keys.json"
    existing = {"himmy_preexisting": {"all_tenants": True}}
    _write_json_0600(keys_file, existing)
    os.environ["HIMMY_AUTH_MODE"] = "apikey"
    os.environ["HIMMY_API_KEYS_FILE"] = str(keys_file)
    os.environ[commands.API_KEY_ENV] = "himmy_would_be_seeded"

    commands._materialize_api_keys_file()
    # Untouched — the pre-existing file wins over the seed secret.
    assert json.loads(keys_file.read_text()) == existing


def test_seed_noop_when_not_apikey_mode(env: Path) -> None:
    """The seeder does nothing outside apikey mode (offline default stays byte-unchanged)."""
    keys_file = env / ".himmy" / "api_keys.json"
    os.environ.pop("HIMMY_AUTH_MODE", None)  # offline default: no auth mode
    os.environ["HIMMY_API_KEYS_FILE"] = str(keys_file)
    os.environ[commands.API_KEY_ENV] = "himmy_should_not_seed"

    commands._materialize_api_keys_file()
    assert not keys_file.exists()


def test_seed_invalid_json_secret_does_not_write_and_warns(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A malformed HIMMY_API_KEYS_JSON is rejected (no half-written file, clear error)."""
    keys_file = env / ".himmy" / "api_keys.json"
    os.environ["HIMMY_AUTH_MODE"] = "apikey"
    os.environ["HIMMY_API_KEYS_FILE"] = str(keys_file)
    os.environ[commands.API_KEYS_JSON_ENV] = "{not: valid json"

    commands._materialize_api_keys_file()
    assert not keys_file.exists()
    assert "not valid JSON" in capsys.readouterr().err


def test_seed_never_prints_raw_key_material(
    env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The seeder logs that it seeded + from which env — NEVER the raw secret value."""
    keys_file = env / ".himmy" / "api_keys.json"
    os.environ["HIMMY_AUTH_MODE"] = "apikey"
    os.environ["HIMMY_API_KEYS_FILE"] = str(keys_file)
    raw = "himmy_super_secret_value_do_not_log"
    os.environ[commands.API_KEY_ENV] = raw

    commands._materialize_api_keys_file()
    out = capsys.readouterr()
    combined = out.err + out.out
    assert commands.API_KEY_ENV in combined  # names the env it seeded from
    assert raw not in combined  # but never the secret itself


# ------------------------------------------- sec-r1 regressions (deploy-DX red-team round 1)


def test_generated_webhook_secret_persists_across_restarts(env: Path) -> None:
    """A generated signing secret is PERSISTED, so a restart reuses it (signed senders survive).

    Regression: the secret used to be written to process env only, so every restart minted a
    fresh secret and broke every previously-configured signed sender. With a writable secrets
    backend (the ``env`` fixture wires a FileSecrets provider), the generated secret must be
    stored through it so a second ``_enable_inbound_webhook`` (a fresh process reading the same
    backend) resolves the SAME secret.
    """
    from himmy.config.secrets import get_secret

    yaml = str(env / "agent.yaml")
    # No operator secret set → a fresh one is generated and persisted through the provider.
    os.environ.pop(WEBHOOK_SIGNING_SECRET, None)
    first = commands._enable_inbound_webhook(yaml)
    assert first
    # It landed in the (writable) secrets backend, not merely process env.
    assert get_secret(WEBHOOK_SIGNING_SECRET) == first
    # Simulate a restart: clear the process env, re-run — the SAME secret is resolved.
    os.environ.pop(WEBHOOK_SIGNING_SECRET, None)
    second = commands._enable_inbound_webhook(yaml)
    assert second == first  # not regenerated — the persisted secret is reused


def test_enable_inbound_webhook_turns_on_the_replay_guard(env: Path) -> None:
    """The front door sets HIMMY_WEBHOOK_REQUIRE_TIMESTAMP (replay guard on by default)."""
    commands._enable_inbound_webhook(str(env / "agent.yaml"))
    from himmy.config.flags import truthy

    assert truthy(os.environ.get("HIMMY_WEBHOOK_REQUIRE_TIMESTAMP"))


def test_provider_stamp_preserves_comments_and_writes_atomically(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp is a comment-preserving textual edit, not a safe_dump that drops comments.

    Regression: ``himmy deploy`` used to yaml.safe_load + safe_dump the user's spec, destroying
    every teaching comment. Now it edits only the ``model:``/``provider:`` lines textually, so
    comments survive; and it writes atomically (no truncation window).
    """
    yaml = env / "agent.yaml"
    yaml.write_text(
        "name: bot\n"
        "description: d\n"
        "model: default\n"
        "# provider: claude-cli      # keep this teaching comment\n"
        "# tool_packs: [web]         # and this one\n"
    )
    from himmy.cli.wizard import ProviderChoice

    fake = ProviderChoice(key="ollama", label="ollama", model="llama3.2")
    monkeypatch.setattr("himmy.cli.wizard.detect_provider_choices", lambda: [fake])

    provider, model, hint = commands._deploy_resolve_provider(str(yaml))
    assert provider == "ollama" and model == "llama3.2" and hint is None
    text = yaml.read_text()
    assert "provider: ollama" in text
    assert "model: llama3.2" in text
    # The user's teaching comments survived the stamp (the whole point of the fix).
    assert "# keep this teaching comment" in text
    assert "# and this one" in text


def test_atomic_write_leaves_no_temp_file(env: Path) -> None:
    """The atomic writer replaces the target and leaves no stray tmp file behind."""
    target = env / "spec.yaml"
    target.write_text("old\n")
    assert commands._atomic_write_text(target, "new content\n") is True
    assert target.read_text() == "new content\n"
    strays = [p.name for p in env.iterdir() if p.name.startswith(".stamp-")]
    assert strays == []


def test_restart_supervisor_fails_fast_on_permanent_config_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A permanent config error (exit 2) is NOT retried forever — the supervisor gives up.

    Regression: ``himmy deploy --channel telegram`` with no token returned exit 2 on every
    attempt and the supervisor looped forever. Now exit 2 (the fatal-config code) is
    propagated immediately instead of thrashing.
    """
    calls = {"n": 0}

    def _always_fatal(_args: object) -> int:
        calls["n"] += 1
        return commands._DEPLOY_FATAL_EXIT  # a missing-token-style permanent failure

    code = commands._deploy_channel_with_restart(_always_fatal, argparse.Namespace(), "telegram")
    assert code == commands._DEPLOY_FATAL_EXIT
    assert calls["n"] == 1  # tried exactly once, did not loop
    assert "permanent" in capsys.readouterr().err.lower()


def test_restart_supervisor_still_retries_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient non-zero exit (not the fatal code) is still retried, then succeeds."""
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    codes = iter([1, 1, 0])  # two transient failures, then a clean stop

    def _flaky(_args: object) -> int:
        return next(codes)

    code = commands._deploy_channel_with_restart(_flaky, argparse.Namespace(), "studio")
    assert code == 0
    assert len(sleeps) == 2  # backed off between the two retries, then exited on 0


def test_share_key_is_revoked_when_deploy_aborts(
    env: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failed ``--share`` deploy revokes the just-minted key (no orphaned live credential).

    Regression: the share key was persisted BEFORE the server bound, so an abort (fail-closed
    refusal / port-in-use) left a never-expiring all-tenants key on disk. The abort paths now
    revoke exactly that key.
    """
    from himmy.cli.apikey_cmd import _load_keys

    monkeypatch.chdir(env)

    # Force the server bind to fail as if the port were in use.
    async def _boom(*a: object, **k: object) -> None:
        raise OSError(48, "address already in use")

    monkeypatch.setattr(commands, "_serve_and_worker", _boom)

    args = argparse.Namespace(
        file=str(env / "agent.yaml"),
        agent=None,
        docker=False,
        channel="http",
        host="127.0.0.1",
        port=8000,
        share=True,
    )
    rc = commands.cmd_deploy(args)
    assert rc == 1  # port-in-use failure

    # The minted share key must NOT linger in the keys file.
    keys_path = Path(os.environ["HIMMY_API_KEYS_FILE"])
    remaining = _load_keys(keys_path) if keys_path.exists() else {}
    # Any himmy_ all-tenants key that was minted for --share is gone (file empty or no share key).
    assert not any(k.startswith("himmy_") for k in remaining)


def test_share_provisions_a_metrics_token_off_the_shared_key(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--share`` sets HIMMY_METRICS_TOKEN so the shared key can't scrape /metrics.

    Regression: /metrics authenticates any valid principal, so without its own token the
    shared 'friend' key could read the deployment-wide observability map. --share now
    provisions a SEPARATE metrics token (not the shared key, not printed).
    """
    monkeypatch.chdir(env)
    os.environ.pop("HIMMY_METRICS_TOKEN", None)
    ok, apikey = commands._deploy_configure_share_auth("127.0.0.1")
    assert ok and apikey
    token = os.environ.get("HIMMY_METRICS_TOKEN")
    assert token  # a metrics token was provisioned
    assert token != apikey  # and it is NOT the shared friend key
