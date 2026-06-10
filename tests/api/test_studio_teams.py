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
    (tmp_path / "solo.agent.yaml").write_text(
        "name: solo\ndescription: a single agent\n"
    )
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
    frames = _frames(client, {"team_path": "demo.team.yaml", "prompt": "hello team"})
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


# ---- Team builder: validate / save / detail ------------------------------

_BUILD_BODY = {
    "name": "duo",
    "entry": "boss",
    "members": [
        {
            "name": "boss",
            "description": "Coordinate the work.",
            "role": "manager",
            "delegates": ["scout"],
        },
        {
            "name": "scout",
            "description": "Investigate details.",
            "provider": "stub",
        },
    ],
}


def test_validate_team_ok(client: TestClient) -> None:
    r = client.post("/api/studio/teams/validate", json=_BUILD_BODY)
    assert r.status_code == 200
    assert r.json() == {"ok": True, "errors": []}


def test_validate_team_collects_errors(client: TestClient) -> None:
    body = {
        "name": "bad",
        "entry": "ghost",
        "members": [
            {"name": "a", "delegates": ["a", "nobody"], "tool_packs": ["nopack"]},
            {"name": "a"},
        ],
    }
    r = client.post("/api/studio/teams/validate", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["ok"] is False
    joined = " ".join(out["errors"])
    assert "Duplicate member name" in joined
    assert "Entry member" in joined
    assert "cannot delegate to itself" in joined
    assert "unknown delegate target" in joined
    assert "unknown tool pack" in joined


def test_validate_team_bad_name(client: TestClient) -> None:
    body = dict(_BUILD_BODY, name="../evil")
    r = client.post("/api/studio/teams/validate", json=body)
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "Team name" in r.json()["errors"][0]


def test_save_team_writes_loadable_yaml(client: TestClient, tmp_path: Path) -> None:
    r = client.post("/api/studio/teams", json=_BUILD_BODY)
    assert r.status_code == 200
    summary = r.json()
    assert summary["name"] == "duo"
    assert summary["path"] == "duo.team.yaml"
    assert summary["entry"] == "boss"
    assert [m["name"] for m in summary["members"]] == ["boss", "scout"]

    # The file is real, loadable by the team loader, and discoverable.
    from himmy.config.team_spec import load_team_spec

    spec = load_team_spec(tmp_path / "duo.team.yaml")
    assert spec.entry == "boss"
    assert spec.members[0].delegates == ["scout"]
    teams = client.get("/api/studio/teams").json()
    assert "duo" in {t["name"] for t in teams}


def test_save_team_invalid_spec_422(client: TestClient) -> None:
    body = dict(_BUILD_BODY, entry="ghost")
    r = client.post("/api/studio/teams", json=body)
    assert r.status_code == 422
    assert "Entry member" in r.json()["detail"]


def test_save_team_conflict_then_overwrite_preserves_extras(
    client: TestClient, tmp_path: Path
) -> None:
    import yaml

    # demo.team.yaml exists (fixture); decorate it with a key the form
    # doesn't expose, then prove an overwrite keeps it.
    p = tmp_path / "demo.team.yaml"
    raw = yaml.safe_load(p.read_text())
    raw["mcp_servers"] = []
    raw["x_custom"] = 7
    p.write_text(yaml.safe_dump(raw, sort_keys=False))

    body = dict(_BUILD_BODY, name="demo")
    r = client.post("/api/studio/teams", json=body)
    assert r.status_code == 409

    r = client.post("/api/studio/teams", json=dict(body, overwrite=True))
    assert r.status_code == 200
    saved = yaml.safe_load(p.read_text())
    assert saved["entry"] == "boss"
    assert saved["x_custom"] == 7  # advanced/unknown keys survive the edit


def test_save_team_explicit_path_guarded(client: TestClient) -> None:
    body = dict(_BUILD_BODY, path="../outside.team.yaml")
    assert client.post("/api/studio/teams", json=body).status_code == 400
    body = dict(_BUILD_BODY, path="notes.txt")
    assert client.post("/api/studio/teams", json=body).status_code == 400


def test_save_team_explicit_path_roundtrip(client: TestClient, tmp_path: Path) -> None:
    # Editing the discovered team writes back to ITS file, not a new one.
    body = dict(_BUILD_BODY, path="demo.team.yaml", overwrite=True)
    r = client.post("/api/studio/teams", json=body)
    assert r.status_code == 200
    assert r.json()["path"] == "demo.team.yaml"
    detail = client.get(
        "/api/studio/teams/detail", params={"path": "demo.team.yaml"}
    ).json()
    assert detail["spec"]["entry"] == "boss"


def test_team_detail_parses_spec(client: TestClient) -> None:
    r = client.get("/api/studio/teams/detail", params={"path": "demo.team.yaml"})
    assert r.status_code == 200
    detail = r.json()
    assert detail["name"] == "demo"
    assert detail["spec"]["entry"] == "lead"
    assert [m["name"] for m in detail["spec"]["members"]] == ["lead", "helper"]
    assert detail["has_advanced"] is False


def test_team_detail_unknown_404(client: TestClient) -> None:
    r = client.get("/api/studio/teams/detail", params={"path": "nope.team.yaml"})
    assert r.status_code == 404
    r = client.get("/api/studio/teams/detail", params={"path": "../etc.yaml"})
    assert r.status_code == 400


# ---- Workflow runner: streaming node-level events -------------------------

_STUB_WORKFLOW = """\
name: brief
description: two stub steps
steps:
  - name: find
    subtask: "Say something about cats"
    output_key: found
  - name: write
    subtask: "Summarize: {found}"
"""


def _wf_frames(client: TestClient, body: dict) -> list[dict]:
    out: list[dict] = []
    with client.stream("POST", "/api/studio/workflows/run-stream", json=body) as r:
        assert r.status_code == 200
        assert "text/event-stream" in r.headers["content-type"]
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def test_workflow_stream_node_events(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "brief.workflow.yaml").write_text(_STUB_WORKFLOW)
    frames = _wf_frames(
        client,
        {
            "workflow_path": "brief.workflow.yaml",
            "agent_path": "solo.agent.yaml",
            "provider": "stub",
        },
    )
    assert frames[0]["type"] == "workflow_start"
    assert frames[0]["workflow"] == "brief"
    assert frames[0]["steps"] == ["find", "write"]

    nodes = [(f["name"], f["state"]) for f in frames if f["type"] == "node"]
    assert nodes == [
        ("find", "running"),
        ("find", "completed"),
        ("write", "running"),
        ("write", "completed"),
    ]

    done = frames[-1]
    assert done["type"] == "done"
    assert done["status"] == "completed"
    assert [r["step_name"] for r in done["step_results"]] == ["find", "write"]
    assert all(r["status"] == "completed" for r in done["step_results"])
    assert "found" in done["final_state"]


def test_workflow_stream_failed_step(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "broken.workflow.yaml").write_text(
        "name: broken\nsteps:\n  - name: boom\n    subtask: 'Use {missing_key}'\n"
    )
    frames = _wf_frames(
        client,
        {
            "workflow_path": "broken.workflow.yaml",
            "agent_path": "solo.agent.yaml",
            "provider": "stub",
        },
    )
    nodes = [(f["name"], f["state"]) for f in frames if f["type"] == "node"]
    assert ("boom", "failed") in nodes
    failed = next(f for f in frames if f["type"] == "node" and f["state"] == "failed")
    assert "missing_key" in (failed.get("error") or "")
    done = frames[-1]
    assert done["type"] == "done"
    assert done["status"] == "failed"


def test_workflow_stream_unknown_paths(client: TestClient, tmp_path: Path) -> None:
    r = client.post(
        "/api/studio/workflows/run-stream",
        json={"workflow_path": "nope.workflow.yaml", "agent_path": "solo.agent.yaml"},
    )
    assert r.status_code == 404
    (tmp_path / "empty.workflow.yaml").write_text("name: empty\nsteps: []\n")
    r = client.post(
        "/api/studio/workflows/run-stream",
        json={"workflow_path": "empty.workflow.yaml", "agent_path": "solo.agent.yaml"},
    )
    assert r.status_code == 400
    r = client.post(
        "/api/studio/workflows/run-stream",
        json={"workflow_path": "../escape.yaml", "agent_path": "solo.agent.yaml"},
    )
    assert r.status_code == 400
