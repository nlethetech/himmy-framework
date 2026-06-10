"""Tests for Studio Projects: grouped chats + default KB/agent, in the chats store."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api import studio_chats as sc
from himmy.api.app import create_app

# The pre-projects chat_sessions schema (no project_id column), used to prove
# the additive migration upgrades old databases in place.
_OLD_SCHEMA = """
CREATE TABLE chat_sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    agent_path  TEXT,
    provider    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE chat_messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    created_at  TEXT NOT NULL
);
"""


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HIMMY_CHATS_PATH", str(tmp_path / "chats.db"))
    sc.reset_chats_store()
    yield
    sc.reset_chats_store()


def _save_chat(client: TestClient, text: str) -> dict:
    return client.post(
        "/api/studio/chats",
        json={"messages": [{"role": "user", "text": text}]},
    ).json()


def test_projects_crud_routes(store: None) -> None:
    c = TestClient(create_app())

    assert c.get("/api/studio/projects").json() == []

    created = c.post(
        "/api/studio/projects",
        json={"name": "Farm research", "description": "soil + ducks"},
    ).json()
    assert created["name"] == "Farm research"
    assert created["description"] == "soil + ducks"
    assert created["chat_count"] == 0
    pid = created["id"]

    listed = c.get("/api/studio/projects").json()
    assert [p["id"] for p in listed] == [pid]

    detail = c.get(f"/api/studio/projects/{pid}").json()
    assert detail["chats"] == []
    assert detail["kb_name"] is None
    assert detail["agent_name"] is None

    renamed = c.patch(
        f"/api/studio/projects/{pid}",
        json={"name": "Farm research v2", "agent_path": "agents/farm.yaml"},
    ).json()
    assert renamed["name"] == "Farm research v2"
    assert renamed["agent_path"] == "agents/farm.yaml"
    # explicit null clears an attachment; omitted fields stay untouched
    cleared = c.patch(f"/api/studio/projects/{pid}", json={"agent_path": None}).json()
    assert cleared["agent_path"] is None
    assert cleared["name"] == "Farm research v2"

    assert c.delete(f"/api/studio/projects/{pid}").json()["ok"] is True
    assert c.get("/api/studio/projects").json() == []
    assert c.delete(f"/api/studio/projects/{pid}").json()["ok"] is False


def test_projects_validation_and_404s(store: None) -> None:
    c = TestClient(create_app())
    assert c.post("/api/studio/projects", json={}).status_code == 422
    assert c.post("/api/studio/projects", json={"name": "x" * 121}).status_code == 422
    assert c.get("/api/studio/projects/nope").status_code == 404
    assert c.patch("/api/studio/projects/nope", json={"name": "y"}).status_code == 404
    assert (
        c.post("/api/studio/projects/nope/assign", json={"chat_id": "z"}).status_code
        == 404
    )


def test_assign_unassign_and_detail_ordering(store: None) -> None:
    c = TestClient(create_app())
    pid = c.post("/api/studio/projects", json={"name": "Ledger"}).json()["id"]
    first = _save_chat(c, "First chat")
    second = _save_chat(c, "Second chat")

    for chat in (first, second):
        assert c.post(
            f"/api/studio/projects/{pid}/assign", json={"chat_id": chat["id"]}
        ).json()["ok"]
    # unknown chat → 404, project intact
    assert (
        c.post(
            f"/api/studio/projects/{pid}/assign", json={"chat_id": "ghost"}
        ).status_code
        == 404
    )

    detail = c.get(f"/api/studio/projects/{pid}").json()
    assert detail["chat_count"] == 2
    ids = [ch["id"] for ch in detail["chats"]]
    assert set(ids) == {first["id"], second["id"]}
    stamps = [ch["updated_at"] for ch in detail["chats"]]
    assert stamps == sorted(stamps, reverse=True)  # newest first

    # the chats list/get surfaces carry the assignment
    assert c.get(f"/api/studio/chats/{first['id']}").json()["project_id"] == pid

    assert c.post(
        f"/api/studio/projects/{pid}/unassign", json={"chat_id": first["id"]}
    ).json()["ok"]
    assert c.get(f"/api/studio/chats/{first['id']}").json()["project_id"] is None
    # unassigning twice is a quiet no-op
    assert (
        c.post(
            f"/api/studio/projects/{pid}/unassign", json={"chat_id": first["id"]}
        ).json()["ok"]
        is False
    )
    assert c.get(f"/api/studio/projects/{pid}").json()["chat_count"] == 1


def test_delete_project_keeps_chats(store: None) -> None:
    c = TestClient(create_app())
    pid = c.post("/api/studio/projects", json={"name": "Doomed"}).json()["id"]
    chat = _save_chat(c, "Survivor")
    c.post(f"/api/studio/projects/{pid}/assign", json={"chat_id": chat["id"]})

    assert c.delete(f"/api/studio/projects/{pid}").json()["ok"]
    survivor = c.get(f"/api/studio/chats/{chat['id']}").json()
    assert survivor["project_id"] is None
    assert survivor["messages"][0]["text"] == "Survivor"


def test_save_preserves_project_assignment(store: None) -> None:
    """A plain resave (no project_id) must not pull a chat out of its project."""
    s = sc.get_chats_store()
    project = s.create_project(name="Sticky")
    saved = s.save(
        session_id=None,
        title=None,
        agent_path=None,
        provider=None,
        messages=[sc.ChatMessage(role="user", text="hello")],
        project_id=project.id,
    )
    assert saved.project_id == project.id

    resaved = s.save(
        session_id=saved.id,
        title=saved.title,
        agent_path=None,
        provider=None,
        messages=[
            sc.ChatMessage(role="user", text="hello"),
            sc.ChatMessage(role="agent", text="hi"),
        ],
    )
    assert resaved.project_id == project.id
    assert s.project_chats(project.id)[0].id == saved.id


def test_migration_is_idempotent_on_old_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening a pre-projects DB adds project_id once; old rows are untouched."""
    db = tmp_path / "old-chats.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO chat_sessions VALUES ('s1', 'Old chat', NULL, NULL, "
        "'2026-01-01T00:00:00+00:00', '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HIMMY_CHATS_PATH", str(db))
    sc.reset_chats_store()
    try:
        for _ in range(2):  # open twice — the ALTER must not re-run
            store = sc.ChatsStore(str(db))
            cols = [
                r[1]
                for r in store._conn.execute(
                    "PRAGMA table_info(chat_sessions)"
                ).fetchall()
            ]
            assert cols.count("project_id") == 1
            store.close()

        s = sc.get_chats_store()
        sessions = s.list()
        assert [x.id for x in sessions] == ["s1"]
        assert sessions[0].title == "Old chat"
        assert sessions[0].project_id is None
        # and the upgraded store fully supports projects
        p = s.create_project(name="After upgrade")
        assert s.assign_chat(p.id, "s1") is True
        assert s.get("s1").project_id == p.id  # type: ignore[union-attr]
    finally:
        sc.reset_chats_store()


def test_project_lifecycle_notifies(store: None) -> None:
    from himmy.api.routers import studio_notify

    c = TestClient(create_app())
    pid = c.post("/api/studio/projects", json={"name": "Noisy"}).json()["id"]
    c.delete(f"/api/studio/projects/{pid}")
    kinds = [
        n["title"]
        for n in studio_notify._NOTIFICATIONS
        if n["kind"] == "project" and "Noisy" in n["title"]
    ]
    assert any("created" in t for t in kinds)
    assert any("deleted" in t for t in kinds)
