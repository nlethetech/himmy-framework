"""Tests for Himmy Studio run persistence + browsing (Phase 2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.studio_runs import reset_run_store


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    (tmp_path / "tooly.agent.yaml").write_text(
        "name: tooly\ndescription: Tools.\ntool_packs: [utils]\n"
    )
    monkeypatch.chdir(tmp_path)
    reset_run_store()  # the store is keyed on cwd; start fresh per test
    return TestClient(create_app())


def _run(client: TestClient, body: dict) -> list[dict]:
    frames: list[dict] = []
    with client.stream("POST", "/api/studio/run", json=body) as r:
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    return frames


def test_run_is_persisted_and_listed(client: TestClient) -> None:
    frames = _run(
        client, {"agent_path": "agent.yaml", "prompt": "hello", "provider": "stub"}
    )
    done = frames[-1]
    assert done["type"] == "done"
    run_id = done["run_id"]

    lst = client.get("/api/studio/runs").json()
    assert lst["total"] == 1
    assert lst["items"][0]["id"] == run_id
    assert lst["items"][0]["agent_name"] == "helper"
    assert lst["items"][0]["status"] == "ok"


def test_run_detail_has_transcript_and_timeline(client: TestClient) -> None:
    frames = _run(
        client, {"agent_path": "agent.yaml", "prompt": "hello", "provider": "stub"}
    )
    run_id = frames[-1]["run_id"]
    det = client.get(f"/api/studio/runs/{run_id}").json()

    assert det["agent_path"] == "agent.yaml"
    assert [m["role"] for m in det["messages"]] == ["user", "assistant"]
    assert det["messages"][0]["content"] == "hello"
    kinds = [s["type"] for s in det["timeline"]]
    assert kinds[0] == "run_started"
    assert kinds[-1] == "run_completed"


def test_tool_run_records_tools_and_richer_timeline(client: TestClient) -> None:
    frames = _run(
        client,
        {"agent_path": "tooly.agent.yaml", "prompt": "2+2", "provider": "stub"},
    )
    run_id = frames[-1]["run_id"]
    det = client.get(f"/api/studio/runs/{run_id}").json()
    assert det["tool_count"] >= 1
    assert len(det["tools"]) >= 1
    # tool runs surface runtime events between the bookends
    assert len(det["timeline"]) > 2


def test_multi_turn_history_in_transcript(client: TestClient) -> None:
    frames = _run(
        client,
        {
            "agent_path": "agent.yaml",
            "prompt": "third",
            "provider": "stub",
            "history": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
            ],
        },
    )
    det = client.get(f"/api/studio/runs/{frames[-1]['run_id']}").json()
    roles = [m["role"] for m in det["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert det["messages"][2]["content"] == "third"


def test_runs_pagination(client: TestClient) -> None:
    for i in range(3):
        _run(
            client,
            {"agent_path": "agent.yaml", "prompt": f"p{i}", "provider": "stub"},
        )
    page = client.get("/api/studio/runs?limit=2&offset=0").json()
    assert page["total"] == 3
    assert len(page["items"]) == 2
    assert page["next_offset"] == 2
    page2 = client.get("/api/studio/runs?limit=2&offset=2").json()
    assert len(page2["items"]) == 1
    assert page2["next_offset"] is None


def test_unknown_run_404(client: TestClient) -> None:
    assert client.get("/api/studio/runs/does-not-exist").status_code == 404
