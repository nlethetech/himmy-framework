"""Tests for Himmy Studio team running (discovery + SSE + persistence).

Uses an all-stub team so the streaming/persistence machinery is exercised offline
(real multi-provider manager/worker runs are verified manually).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.studio_runs import reset_run_store

_STUB_TEAM = """\
entry: lead
members:
  - name: lead
    description: Answer the request directly.
    provider: stub
  - name: helper
    description: A delegate worker.
    provider: stub
    delegates: []
"""


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    (tmp_path / "demo.team.yaml").write_text(_STUB_TEAM)
    (tmp_path / "solo.agent.yaml").write_text("name: solo\ndescription: a single agent\n")
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    return TestClient(create_app())


def _frames(client: TestClient, body: dict) -> list[dict]:
    out: list[dict] = []
    with client.stream("POST", "/api/studio/run-team", json=body) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def test_team_discovery_excludes_agents(client: TestClient) -> None:
    teams = client.get("/api/studio/teams").json()
    agents = client.get("/api/studio/agents").json()
    assert [t["name"] for t in teams] == ["demo"]
    assert teams[0]["entry"] == "lead"
    # the single agent is in /agents, not /teams; the team is not in /agents
    assert "solo" in {a["name"] for a in agents}
    assert "demo" not in {a["name"] for a in agents}


def test_team_run_streams_and_persists(client: TestClient) -> None:
    frames = _frames(
        client, {"team_path": "demo.team.yaml", "prompt": "hello team"}
    )
    types = [f["type"] for f in frames]
    assert types[0] == "start"
    assert frames[0]["team"] is True
    assert types[-1] == "done"
    assert any(t == "message" for t in types)
    run_id = frames[-1]["run_id"]

    # persisted under Runs with provider=team
    detail = client.get(f"/api/studio/runs/{run_id}").json()
    assert detail["provider"] == "team"
    assert detail["messages"][0]["content"] == "hello team"
    assert detail["timeline"][0]["type"] == "run_started"


def test_run_unknown_team_404(client: TestClient) -> None:
    r = client.post(
        "/api/studio/run-team", json={"team_path": "nope.team.yaml", "prompt": "x"}
    )
    assert r.status_code == 404
