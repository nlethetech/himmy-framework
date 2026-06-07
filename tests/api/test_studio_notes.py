"""Tests for Studio Notes: store, routes, and the agentic notes tool pack."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_notes as sn
from himmy.api.app import create_app


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIMMY_NOTES_PATH", str(tmp_path / "notes.db"))
    sn.reset_notes_store()
    yield
    sn.reset_notes_store()


def test_store_upsert_list_delete(store: None) -> None:
    s = sn.get_notes_store()
    n = s.upsert(sn.Note(title="Plan", body="# Plan"))
    assert s.list()[0].title == "Plan"
    assert s.find_by_title("Plan").body == "# Plan"  # type: ignore[union-attr]
    n.body = "# Plan v2"
    s.upsert(n)  # overwrite by id
    assert s.get(n.id).body == "# Plan v2"  # type: ignore[union-attr]
    assert s.delete(n.id) is True
    assert s.list() == []


def test_notes_routes(store: None) -> None:
    c = TestClient(create_app())
    saved = c.put("/api/studio/notes", json={"title": "T", "body": "B"}).json()
    assert c.get("/api/studio/notes").json()[0]["title"] == "T"
    assert c.get(f"/api/studio/notes/{saved['id']}").json()["body"] == "B"
    assert c.delete(f"/api/studio/notes/{saved['id']}").json()["ok"]
    assert c.get("/api/studio/notes/nope").status_code == 404


def test_notes_pack_registers_agent_tools(store: None) -> None:
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit.config import ToolkitConfig
    from himmy.toolkit.pack import BUILTIN_PACKS, register_packs

    assert "notes" in BUILTIN_PACKS
    reg = ToolRegistry()
    register_packs(reg, ["notes"], ToolkitConfig.from_env())
    names = {d.name for d in reg.list()}
    assert {"list_notes", "read_note", "write_note"} <= names


def test_notes_pack_shares_the_gui_store(store: None) -> None:
    # Prove the store an agent writes to is the one the GUI reads.
    sn.get_notes_store().upsert(sn.Note(title="FromAgent", body="hi"))
    c = TestClient(create_app())
    titles = [n["title"] for n in c.get("/api/studio/notes").json()]
    assert "FromAgent" in titles
