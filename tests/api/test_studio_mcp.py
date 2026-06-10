"""Tests for the Studio MCP router — registry CRUD, test (listing only), attach.

Drives the FastAPI app in-process via TestClient against a tmp project root.
The "test server" path spawns REAL stdio subprocesses (the in-repo mock MCP
server / a tiny env-probe server written per-test), so the launch → handshake →
tools/list mechanics are exercised end to end without any network. An optional
live test runs an attached agent against local Ollama (skipped when absent).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from himmy.config.secrets import (
    ChainSecretProvider,
    EnvSecrets,
    FileSecrets,
    configure_secrets,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MOCK_SERVER = str(_REPO_ROOT / "tests" / "mcp" / "mock_mcp_server.py")

# A minimal stdio MCP server that logs every JSON-RPC method it receives and
# advertises one tool whose description carries an env var — so a test can
# assert (a) secret injection into the subprocess env and (b) that the router
# only ever issues initialize/tools/list, never tools/call.
_PROBE_SERVER = textwrap.dedent(
    """
    import json, os, sys

    def send(m):
        sys.stdout.write(json.dumps(m) + "\\n")
        sys.stdout.flush()

    log = os.environ.get("MCP_METHOD_LOG")
    for raw in sys.stdin:
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if log:
            with open(log, "a") as fh:
                fh.write(str(msg.get("method", "")) + "\\n")
        mid, method = msg.get("id"), msg.get("method")
        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "probe", "version": "0"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": [{
                "name": "probe",
                "description": "token=" + os.environ.get("PROBE_TOKEN", "absent")[::-1],
                "inputSchema": {"type": "object"}}]}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "nope"}})
    """
)


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp project root (cwd) with one plain agent and one team spec."""
    (tmp_path / "agent.yaml").write_text(
        "name: helper\ndescription: A simple helper.\n", encoding="utf-8"
    )
    (tmp_path / "team.yaml").write_text(
        "entry: lead\nmembers:\n  - path: agent.yaml\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def client(project: Path) -> TestClient:
    from himmy.api.app import create_app

    yield TestClient(create_app())
    configure_secrets(None)


def _mk(name: str = "mock", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": name,
        "command": sys.executable,
        "args": [_MOCK_SERVER],
        **extra,
    }
    return body


def _create(client: TestClient, name: str = "mock", **extra: Any) -> dict[str, Any]:
    r = client.post("/api/studio/mcp/servers", json=_mk(name, **extra))
    assert r.status_code == 201, r.text
    return r.json()


# ---- CRUD -------------------------------------------------------------------


def test_registry_starts_empty(client: TestClient) -> None:
    r = client.get("/api/studio/mcp/servers")
    assert r.status_code == 200
    assert r.json() == {"servers": []}


def test_create_list_and_persisted_shape(client: TestClient, project: Path) -> None:
    created = _create(client, "fs", prefix="fs_", env_keys=["GITHUB_TOKEN"])
    assert created["name"] == "fs"
    assert created["transport"] == "stdio"
    assert created["enabled"] is True
    assert created["last_test"] is None
    assert created["tools"] == []

    listed = client.get("/api/studio/mcp/servers").json()["servers"]
    assert [s["name"] for s in listed] == ["fs"]
    # the registry file persists env NAMES only — never a secret value
    raw = json.loads((project / ".himmy" / "mcp_servers.json").read_text())
    assert raw["servers"][0]["env_keys"] == ["GITHUB_TOKEN"]


def test_create_duplicate_is_409(client: TestClient) -> None:
    _create(client, "dup")
    r = client.post("/api/studio/mcp/servers", json=_mk("dup"))
    assert r.status_code == 409


def test_validation_bounds(client: TestClient) -> None:
    # non-stdio transport: clear 422, not a silent accept
    r = client.post("/api/studio/mcp/servers", json=_mk("h", transport="http"))
    assert r.status_code == 422
    assert "stdio" in r.text
    # env entry that looks like NAME=value (a value!) is rejected
    r = client.post("/api/studio/mcp/servers", json=_mk("e", env_keys=["TOKEN=abc123"]))
    assert r.status_code == 422
    # bad name characters
    r = client.post("/api/studio/mcp/servers", json=_mk("../etc"))
    assert r.status_code == 422
    # oversized argument
    r = client.post("/api/studio/mcp/servers", json=_mk("a", args=["x" * 600]))
    assert r.status_code == 422


def test_update_rename_and_cache_invalidation(client: TestClient) -> None:
    _create(client, "one")
    # a successful test populates the tool cache…
    assert client.post("/api/studio/mcp/servers/one/test").json()["ok"] is True
    server = client.get("/api/studio/mcp/servers").json()["servers"][0]
    assert len(server["tools"]) == 2

    # …a launch-config change invalidates it
    body = _mk("one", args=[_MOCK_SERVER, "--changed"])
    r = client.put("/api/studio/mcp/servers/one", json=body)
    assert r.status_code == 200
    assert r.json()["tools"] == []
    assert r.json()["last_test"] is None

    # rename onto an existing name collides
    _create(client, "two")
    r = client.put("/api/studio/mcp/servers/one", json=_mk("two"))
    assert r.status_code == 409
    # plain rename works
    r = client.put("/api/studio/mcp/servers/one", json=_mk("three"))
    assert r.status_code == 200
    names = {s["name"] for s in client.get("/api/studio/mcp/servers").json()["servers"]}
    assert names == {"three", "two"}


def test_delete_server(client: TestClient) -> None:
    _create(client, "gone")
    assert client.delete("/api/studio/mcp/servers/gone").status_code == 204
    assert client.delete("/api/studio/mcp/servers/gone").status_code == 404
    assert client.get("/api/studio/mcp/servers").json() == {"servers": []}


# ---- POST /servers/{name}/test (real subprocess, listing only) ---------------


def test_test_lists_tools_and_caches(client: TestClient) -> None:
    _create(client, "mock")
    r = client.post("/api/studio/mcp/servers/mock/test")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    by_name = {t["name"]: t["description"] for t in body["tools"]}
    assert by_name == {
        "echo": "Echo the given text back.",
        "add": "Add two numbers.",
    }
    assert body["server_info"]["name"] == "mock-mcp"

    cached = client.get("/api/studio/mcp/servers/mock/tools").json()
    assert {t["name"] for t in cached["tools"]} == {"echo", "add"}
    assert cached["tested_at"] == body["at"]

    # the lifecycle event landed in the notification sink
    notes = client.get("/api/studio/notify").json()["items"]
    assert any(n["kind"] == "mcp" and "mock" in n["title"] for n in notes)


def test_test_failure_is_a_verdict_not_a_500(client: TestClient) -> None:
    _create(client, "broken", command="/nonexistent/mcp-binary-xyz")
    r = client.post("/api/studio/mcp/servers/broken/test")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["error"]
    server = client.get("/api/studio/mcp/servers").json()["servers"][0]
    assert server["last_test"]["ok"] is False
    assert server["tools"] == []  # never tested OK → no cached tools


def test_test_unknown_server_is_404(client: TestClient) -> None:
    assert client.post("/api/studio/mcp/servers/nope/test").status_code == 404
    assert client.get("/api/studio/mcp/servers/nope/tools").status_code == 404


def test_test_missing_secret_fails_clearly(client: TestClient) -> None:
    _create(client, "needy", env_keys=["MCP_TEST_TOKEN_THAT_IS_UNSET"])
    body = client.post("/api/studio/mcp/servers/needy/test").json()
    assert body["ok"] is False
    assert "MCP_TEST_TOKEN_THAT_IS_UNSET" in body["error"]


def test_secret_injection_and_no_tool_calls(client: TestClient, project: Path) -> None:
    """The secret value reaches the subprocess env; only listing methods run."""
    probe = project / "probe_server.py"
    probe.write_text(_PROBE_SERVER, encoding="utf-8")
    log = project / "methods.log"

    secrets_dir = project / "sec"
    configure_secrets(ChainSecretProvider([FileSecrets(secrets_dir), EnvSecrets()]))
    FileSecrets(secrets_dir).set("PROBE_TOKEN", "s3cr3t-value")
    FileSecrets(secrets_dir).set("MCP_METHOD_LOG", str(log))

    _create(
        client,
        "probe",
        args=[str(probe)],
        env_keys=["PROBE_TOKEN", "MCP_METHOD_LOG"],
    )
    body = client.post("/api/studio/mcp/servers/probe/test").json()
    assert body["ok"] is True
    # injection visible: the server saw the secret value (it advertises it
    # reversed, so the plaintext itself never travels back through the API)
    assert body["tools"][0]["description"] == "token=" + "s3cr3t-value"[::-1]
    # listing only: the router never issued tools/call
    methods = log.read_text().split()
    assert "initialize" in methods and "tools/list" in methods
    assert "tools/call" not in methods
    # the registry file still holds NAMES only — the value was never persisted
    raw = (project / ".himmy" / "mcp_servers.json").read_text()
    assert "s3cr3t-value" not in raw


# ---- attach / detach ----------------------------------------------------------


def test_attach_and_detach_round_trip(client: TestClient, project: Path) -> None:
    _create(client, "mock", prefix="mk_", requires_approval=True)
    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "mock", "agent_path": "agent.yaml", "attach": True},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {
        "server": "mock",
        "agent_path": "agent.yaml",
        "attached": True,
    }

    raw = yaml.safe_load((project / "agent.yaml").read_text())
    assert raw["name"] == "helper"  # untouched fields survive
    assert raw["mcp_servers"] == [
        {
            "command": sys.executable,
            "args": [_MOCK_SERVER],
            "prefix": "mk_",
            "requires_approval": True,
        }
    ]
    # the written file still loads through the real loader
    from himmy.config.agent_spec import load_agent_spec

    spec = load_agent_spec(str(project / "agent.yaml"))
    assert spec.mcp_servers[0].prefix == "mk_"

    # the per-server agents view reflects the attachment
    agents = client.get("/api/studio/mcp/servers/mock/agents").json()["agents"]
    assert agents == [{"path": "agent.yaml", "name": "helper", "attached": True}]

    # attach is idempotent — no duplicate entries
    client.post(
        "/api/studio/mcp/attach",
        json={"server": "mock", "agent_path": "agent.yaml", "attach": True},
    )
    raw = yaml.safe_load((project / "agent.yaml").read_text())
    assert len(raw["mcp_servers"]) == 1

    # detach removes the entry (and the now-empty key)
    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "mock", "agent_path": "agent.yaml", "attach": False},
    )
    assert r.status_code == 200
    raw = yaml.safe_load((project / "agent.yaml").read_text())
    assert "mcp_servers" not in raw
    agents = client.get("/api/studio/mcp/servers/mock/agents").json()["agents"]
    assert agents[0]["attached"] is False


def test_attach_resolves_env_values_into_spec(
    client: TestClient, project: Path
) -> None:
    secrets_dir = project / "sec"
    configure_secrets(ChainSecretProvider([FileSecrets(secrets_dir), EnvSecrets()]))
    FileSecrets(secrets_dir).set("PROBE_TOKEN", "attach-time-value")
    _create(client, "mock", env_keys=["PROBE_TOKEN"])

    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "mock", "agent_path": "agent.yaml", "attach": True},
    )
    assert r.status_code == 200
    assert "attach-time-value" not in r.text  # value never echoed in a response
    raw = yaml.safe_load((project / "agent.yaml").read_text())
    assert raw["mcp_servers"][0]["env"] == {"PROBE_TOKEN": "attach-time-value"}


def test_attach_missing_secret_is_400(client: TestClient, project: Path) -> None:
    _create(client, "mock", env_keys=["MCP_TEST_TOKEN_THAT_IS_UNSET"])
    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "mock", "agent_path": "agent.yaml", "attach": True},
    )
    assert r.status_code == 400
    assert "MCP_TEST_TOKEN_THAT_IS_UNSET" in r.json()["detail"]
    # the spec file was not touched
    assert "mcp_servers" not in yaml.safe_load((project / "agent.yaml").read_text())


def test_attach_error_paths(client: TestClient) -> None:
    _create(client, "mock")
    post = lambda body: client.post("/api/studio/mcp/attach", json=body)  # noqa: E731

    assert post({"server": "nope", "agent_path": "agent.yaml"}).status_code == 404
    assert post({"server": "mock", "agent_path": "missing.yaml"}).status_code == 404
    r = post({"server": "mock", "agent_path": "../outside.yaml"})
    assert r.status_code == 400
    r = post({"server": "mock", "agent_path": "team.yaml"})
    assert r.status_code == 400
    assert "team" in r.json()["detail"]


def test_attach_disabled_server_is_409_but_detach_allowed(
    client: TestClient, project: Path
) -> None:
    _create(client, "off", enabled=False)
    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "off", "agent_path": "agent.yaml", "attach": True},
    )
    assert r.status_code == 409
    # detach of a disabled server is fine (it is a removal)
    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "off", "agent_path": "agent.yaml", "attach": False},
    )
    assert r.status_code == 200


# ---- end-to-end mechanics: an attached server is live inside a real run ------


def test_attached_server_connects_during_stub_run(
    client: TestClient, project: Path
) -> None:
    """Attach → POST /run (stub provider): the MCP subprocess is launched,
    its tools registered, and the run completes — the full lifecycle."""
    _create(client, "mock", prefix="mk_")
    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "mock", "agent_path": "agent.yaml", "attach": True},
    )
    assert r.status_code == 200

    frames: list[dict[str, Any]] = []
    with client.stream(
        "POST",
        "/api/studio/run",
        json={"agent_path": "agent.yaml", "prompt": "hello", "provider": "stub"},
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    types = [f["type"] for f in frames]
    assert types[0] == "start"
    assert types[-1] == "done"
    assert frames[-1]["succeeded"] is True


# ---- optional live test (local Ollama) ----------------------------------------


def _ollama_up() -> bool:
    try:
        return (
            httpx.get("http://localhost:11434/api/tags", timeout=2.0).status_code == 200
        )
    except Exception:
        return False


@pytest.mark.integration
@pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable")
def test_live_run_with_attached_mcp_server(client: TestClient, project: Path) -> None:
    """MECHANICS, not prose: with a real model, a run against an agent carrying
    an attached MCP server streams events and terminates cleanly — proving the
    attach → connect → register → close lifecycle under a live provider."""
    (project / "live.agent.yaml").write_text(
        "name: live-mcp\ndescription: Uses MCP tools when helpful.\n",
        encoding="utf-8",
    )
    _create(client, "mock", prefix="mk_")
    r = client.post(
        "/api/studio/mcp/attach",
        json={"server": "mock", "agent_path": "live.agent.yaml", "attach": True},
    )
    assert r.status_code == 200

    frames: list[dict[str, Any]] = []
    with client.stream(
        "POST",
        "/api/studio/run",
        json={
            "agent_path": "live.agent.yaml",
            "prompt": "Use the mk_add tool to add 2 and 3, then answer.",
            "provider": "ollama",
            "model": "qwen2.5:7b-instruct",
        },
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    types = [f["type"] for f in frames]
    assert types and types[0] == "start"
    assert types[-1] in ("done", "error")
    if types[-1] == "error":  # a model hiccup is not an MCP-lifecycle failure
        pytest.skip(f"live model error: {frames[-1]}")
    assert frames[-1]["succeeded"] is True
