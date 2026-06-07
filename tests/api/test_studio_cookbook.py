"""Tests for Studio Cookbook (recipes) + the Models endpoint."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_cookbook as cb
from himmy.api.app import create_app


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    cb.reset_cookbook_store()
    yield
    cb.reset_cookbook_store()


def test_cookbook_store_and_routes(store: None) -> None:
    c = TestClient(create_app())
    saved = c.put(
        "/api/studio/cookbook",
        json={"name": "Daily brief", "agent_path": "a.yaml", "prompt": "brief me"},
    ).json()
    assert c.get("/api/studio/cookbook").json()[0]["name"] == "Daily brief"
    # edit by id
    c.put(
        "/api/studio/cookbook",
        json={
            "id": saved["id"],
            "name": "Daily brief v2",
            "agent_path": "a.yaml",
            "prompt": "x",
        },
    )
    assert c.get("/api/studio/cookbook").json()[0]["name"] == "Daily brief v2"
    assert c.delete(f"/api/studio/cookbook/{saved['id']}").json()["ok"]
    assert c.get("/api/studio/cookbook").json() == []


def test_cookbook_name_required_422(store: None) -> None:
    c = TestClient(create_app())
    assert c.put("/api/studio/cookbook", json={"name": ""}).status_code == 422


def test_models_endpoint_shape(store: None) -> None:
    c = TestClient(create_app())
    body = c.get("/api/studio/models").json()
    providers = {p["provider"] for p in body}
    assert {"ollama", "claude-cli"} <= providers
    for p in body:
        assert "available" in p and isinstance(p["models"], list)
