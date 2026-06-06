"""Tests for Himmy Studio chat: agent discovery + SSE run (Phase 1)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp project dir (cwd) holding a couple of agent specs."""
    (tmp_path / "agent.yaml").write_text(
        "name: helper\ndescription: A simple helper.\n"
    )
    (tmp_path / "tooly.agent.yaml").write_text(
        "name: tooly\ndescription: A tool user.\ntool_packs: [utils]\n"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def client(project: Path) -> TestClient:
    return TestClient(create_app())


def _frames(client: TestClient, body: dict) -> list[dict]:
    """Collect SSE frames from POST /api/studio/run."""
    out: list[dict] = []
    with client.stream("POST", "/api/studio/run", json=body) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def test_list_agents_discovers_specs(client: TestClient) -> None:
    agents = {a["name"]: a for a in client.get("/api/studio/agents").json()}
    assert "helper" in agents and "tooly" in agents
    assert agents["helper"]["has_tools"] is False
    assert agents["tooly"]["has_tools"] is True


def test_run_streams_tokens_no_tools(client: TestClient) -> None:
    frames = _frames(
        client, {"agent_path": "agent.yaml", "prompt": "hi", "provider": "stub"}
    )
    types = [f["type"] for f in frames]
    assert types[0] == "start"
    assert "token" in types
    assert types[-1] == "done"
    done = frames[-1]
    assert done["output_text"]
    assert done["succeeded"] is True


def test_run_tools_agent_uses_loop(client: TestClient) -> None:
    frames = _frames(
        client,
        {"agent_path": "tooly.agent.yaml", "prompt": "2+2?", "provider": "stub"},
    )
    types = [f["type"] for f in frames]
    assert types[0] == "start"
    assert "message" in types  # tool agents return a full message, not tokens
    assert types[-1] == "done"


def test_run_accepts_multi_turn_history(client: TestClient) -> None:
    frames = _frames(
        client,
        {
            "agent_path": "agent.yaml",
            "prompt": "and again",
            "provider": "stub",
            "history": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
            ],
        },
    )
    assert frames[-1]["type"] == "done"


def test_run_unknown_agent_is_404(client: TestClient) -> None:
    r = client.post("/api/studio/run", json={"agent_path": "nope.yaml", "prompt": "x"})
    assert r.status_code == 404


def test_run_path_traversal_rejected(client: TestClient) -> None:
    r = client.post(
        "/api/studio/run",
        json={"agent_path": "../../etc/passwd", "prompt": "x"},
    )
    assert r.status_code in (400, 404)
