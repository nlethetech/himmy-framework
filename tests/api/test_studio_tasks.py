"""Tests for Studio Tasks: store, routes, and the agentic tasks tool pack."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_tasks as st
from himmy.api.app import create_app


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIMMY_TASKS_PATH", str(tmp_path / "tasks.db"))
    st.reset_tasks_store()
    yield
    st.reset_tasks_store()


def test_store_add_complete_delete(store: None) -> None:
    s = st.get_tasks_store()
    t = s.add("Water the orchard", due="2026-06-10")
    assert s.list()[0].title == "Water the orchard"
    assert s.list()[0].done is False
    assert s.set_done(t.id, True) is True
    assert s.list()[0].done is True
    assert s.set_done(t.id, False) is True
    # complete by title (the agent path)
    assert s.complete_by_title("Water the orchard") is True
    assert s.list()[0].done is True
    assert s.complete_by_title("nope") is False
    assert s.delete(t.id) is True
    assert s.list() == []


def test_tasks_routes(store: None) -> None:
    c = TestClient(create_app())
    saved = c.post("/api/studio/tasks", json={"title": "T", "due": "2026-06-09"}).json()
    assert c.get("/api/studio/tasks").json()[0]["title"] == "T"
    assert c.patch(f"/api/studio/tasks/{saved['id']}", json={"done": True}).json()["ok"]
    assert c.get("/api/studio/tasks").json()[0]["done"] is True
    assert c.delete(f"/api/studio/tasks/{saved['id']}").json()["ok"]
    assert c.get("/api/studio/tasks").json() == []


def test_tasks_pack_registers_agent_tools(store: None) -> None:
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit.config import ToolkitConfig
    from himmy.toolkit.pack import BUILTIN_PACKS, register_packs

    assert "tasks" in BUILTIN_PACKS
    reg = ToolRegistry()
    register_packs(reg, ["tasks"], ToolkitConfig.from_env())
    names = {d.name for d in reg.list()}
    assert {"list_tasks", "add_task", "complete_task"} <= names


def test_tasks_pack_shares_the_gui_store(store: None) -> None:
    # Prove the store an agent writes to is the one the GUI reads.
    st.get_tasks_store().add("FromAgent")
    c = TestClient(create_app())
    titles = [t["title"] for t in c.get("/api/studio/tasks").json()]
    assert "FromAgent" in titles
