"""Tests for Himmy Studio agent authoring (Phase 3)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

from himmy.api.app import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.chdir(tmp_path)
    return TestClient(create_app())


def test_list_tools_and_skills(client: TestClient) -> None:
    packs = {p["name"] for p in client.get("/api/studio/tools").json()}
    assert {"web", "utils", "knowledge"} <= packs
    skills = {s["name"] for s in client.get("/api/studio/skills").json()}
    assert "web_research" in skills


def test_validate_reports_errors(client: TestClient) -> None:
    r = client.post(
        "/api/studio/agents/validate",
        json={"name": "", "tool_packs": ["nope"], "skills": ["ghost"]},
    ).json()
    assert r["ok"] is False
    assert any("Name is required" in e for e in r["errors"])
    assert any("tool pack" in e for e in r["errors"])
    assert any("skill" in e for e in r["errors"])


def test_validate_passes_for_good_spec(client: TestClient) -> None:
    r = client.post(
        "/api/studio/agents/validate",
        json={
            "name": "ok",
            "description": "fine",
            "tool_packs": ["utils"],
            "skills": ["web_research"],
        },
    ).json()
    assert r["ok"] is True
    assert r["errors"] == []


def test_save_writes_tidy_yaml_and_lists(client: TestClient, tmp_path: Path) -> None:
    body = {
        "path": "support.agent.yaml",
        "spec": {
            "name": "support",
            "description": "Helpdesk.",
            "instructions": ["Be kind."],
            "tool_packs": ["utils"],
            "model": "default",  # default → dropped on write
            "language": "en",  # default → dropped
        },
    }
    r = client.put("/api/studio/agents", json=body)
    assert r.status_code == 200
    assert r.json()["name"] == "support"

    written = yaml.safe_load((tmp_path / "support.agent.yaml").read_text())
    assert written["name"] == "support"
    assert written["tool_packs"] == ["utils"]
    assert "model" not in written  # default stripped
    assert "language" not in written

    names = {a["name"] for a in client.get("/api/studio/agents").json()}
    assert "support" in names


def test_load_agent_detail_roundtrip(client: TestClient) -> None:
    client.put(
        "/api/studio/agents",
        json={
            "path": "a.agent.yaml",
            "spec": {"name": "a", "description": "d", "skills": ["summarize"]},
        },
    )
    d = client.get("/api/studio/agent", params={"path": "a.agent.yaml"}).json()
    assert d["path"] == "a.agent.yaml"
    assert d["spec"]["name"] == "a"
    assert d["spec"]["skills"] == ["summarize"]
    assert d["has_advanced"] is False


def test_edit_preserves_advanced_fields(client: TestClient, tmp_path: Path) -> None:
    (tmp_path / "api.agent.yaml").write_text(
        "name: api\ndescription: orders\n"
        "http_tools:\n"
        "  - name: get_order\n"
        "    base_url_env_var: U\n"
        "    path: /o/{id}\n"
    )
    # The form never sees http_tools…
    d = client.get("/api/studio/agent", params={"path": "api.agent.yaml"}).json()
    assert d["has_advanced"] is True
    assert "http_tools" not in d["spec"]
    # …and editing (overwrite=True) keeps it.
    client.put(
        "/api/studio/agents",
        json={
            "path": "api.agent.yaml",
            "spec": {"name": "api", "description": "EDITED"},
            "overwrite": True,
        },
    )
    saved = yaml.safe_load((tmp_path / "api.agent.yaml").read_text())
    assert saved["description"] == "EDITED"
    assert saved["http_tools"][0]["name"] == "get_order"


def test_new_agent_will_not_silently_overwrite(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "agent.yaml").write_text("name: existing\ndescription: keep me\n")
    # Creating a new agent at an existing path (overwrite omitted) → 409, file intact.
    r = client.put(
        "/api/studio/agents",
        json={"path": "agent.yaml", "spec": {"name": "new", "description": "nope"}},
    )
    assert r.status_code == 409
    assert yaml.safe_load((tmp_path / "agent.yaml").read_text())["name"] == "existing"
    # With overwrite=True it goes through.
    r2 = client.put(
        "/api/studio/agents",
        json={
            "path": "agent.yaml",
            "spec": {"name": "new", "description": "yes"},
            "overwrite": True,
        },
    )
    assert r2.status_code == 200
    assert yaml.safe_load((tmp_path / "agent.yaml").read_text())["name"] == "new"


def test_invalid_save_is_422_with_errors(client: TestClient) -> None:
    r = client.put(
        "/api/studio/agents",
        json={"path": "x.yaml", "spec": {"name": "", "tool_packs": ["bogus"]}},
    )
    assert r.status_code == 422
    assert r.json()["detail"]["errors"]


def test_path_traversal_and_extension_rejected(client: TestClient) -> None:
    assert (
        client.put(
            "/api/studio/agents",
            json={"path": "../evil.yaml", "spec": {"name": "a", "description": "b"}},
        ).status_code
        == 400
    )
    assert (
        client.put(
            "/api/studio/agents",
            json={"path": "notyaml.txt", "spec": {"name": "a", "description": "b"}},
        ).status_code
        == 400
    )
