"""Tests for the Studio Deep Research SSE endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.studio_runs import reset_run_store


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    return TestClient(create_app())


def _frames(client: TestClient, body: dict) -> list[dict]:
    out: list[dict] = []
    with client.stream("POST", "/api/studio/research", json=body) as r:
        for line in r.iter_lines():
            if line and line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


def test_research_streams_a_start_and_a_terminal_frame(client: TestClient) -> None:
    frames = _frames(client, {"query": "what is permaculture?", "provider": "stub"})
    assert frames, "expected at least one SSE frame"
    assert frames[0]["type"] == "start"
    assert frames[0]["agent"] == "Deep Research"
    # The research agent has web tools, so it runs the tool loop and ends with a
    # final message (or a clean error frame) — never silence.
    assert frames[-1]["type"] in {"message", "done", "error"}


def test_research_requires_a_query(client: TestClient) -> None:
    r = client.post("/api/studio/research", json={"query": ""})
    assert r.status_code == 422
