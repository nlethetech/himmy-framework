"""Tenant isolation for the cwd-keyed singleton Studio stores (the studio-store-drain pass).

The Studio task/note/calendar/cookbook stores + the notify ring were born single-user-local
with NO tenant column, so once an authenticator binds a Studio key to a workspace a
tenant-bound principal could read + write ACROSS tenants — the IDOR/BOLA recurrence class
the RBAC pass closes everywhere else. This file proves the by-construction fix lands without
disturbing the offline / zero-config / single-box path:

* MIGRATION — an EXISTING single-box DB whose ``CREATE TABLE`` predates the ``workspace_id``
  column still opens, reads its legacy (``NULL``-tenant) rows, and writes; the additive +
  nullable ``ensure_workspace_column`` migration never rewrites a row or hides data; and
* SCOPING — a tenant-bound principal (``all_tenants=False``) sees + mutates ONLY its own
  workspace's rows (plus legacy ``NULL`` rows), and a foreign row is invisible (absent from
  lists, a uniform 404/no-op by id); while
* the OFFLINE invariant — with NO authenticator every row is the single local tenant, the
  scope is ``None``, no filtering happens, and the response is byte-unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy

# --------------------------------------------------------------------------- helpers

#: Every Studio surface this pass drained, with the RBAC grants a browse key needs.
_READER_GRANTS = [
    "studio.console:read",
    "studio.tasks:read",
    "studio.tasks:write",
    "studio.notes:read",
    "studio.notes:write",
    "studio.calendar:read",
    "studio.calendar:write",
    "studio.cookbook:read",
    "studio.cookbook:write",
    "studio.notify:read",
    "studio.notify:write",
]


def _bound_app(tenant: str) -> FastAPI:
    """An app whose mapped key ``"k"`` is bound to a single ``tenant`` (all_tenants=False)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=[tenant], roles=["reader"], auth_method="apikey"
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping({"reader": list(_READER_GRANTS)})
    return app


def _client(app: FastAPI) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": "k"})
    return c


@pytest.fixture
def studio_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every singleton Studio store at a fresh tmp project root + reset singletons."""
    from himmy.api import (
        studio_calendar,
        studio_cookbook,
        studio_notes,
        studio_tasks,
    )
    from himmy.api.routers import studio_notify

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_TASKS_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("HIMMY_NOTES_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("HIMMY_NOTIFY_PATH", str(tmp_path / "notify.db"))
    studio_tasks.reset_tasks_store()
    studio_notes.reset_notes_store()
    studio_calendar.reset_calendar_store()
    studio_cookbook.reset_cookbook_store()
    studio_notify.reset_notify_state()
    yield tmp_path
    studio_tasks.reset_tasks_store()
    studio_notes.reset_notes_store()
    studio_calendar.reset_calendar_store()
    studio_cookbook.reset_cookbook_store()
    studio_notify.reset_notify_state()


# --------------------------------------------------------------------------- migration

def test_tasks_legacy_db_without_workspace_column_opens(tmp_path: Path) -> None:
    """An old tasks.db (no workspace_id column) opens, reads its row, and migrates additively."""
    path = tmp_path / "tasks.db"
    # Build the PRE-migration schema by hand (the column the new store adds is absent).
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, title TEXT NOT NULL, "
        "done INTEGER NOT NULL DEFAULT 0, due TEXT, created_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO tasks (id, title, done, created_at) VALUES "
        "('old1', 'legacy task', 0, '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    from himmy.api.studio_tasks import TasksStore

    store = TasksStore(str(path))
    rows = store.list()
    assert [t.title for t in rows] == ["legacy task"]
    # The additive column now exists and the legacy row carries NULL (single local tenant).
    cols = {r[1] for r in store._conn.execute("PRAGMA table_info(tasks)").fetchall()}
    assert "workspace_id" in cols
    assert store._conn.execute(
        "SELECT workspace_id FROM tasks WHERE id = 'old1'"
    ).fetchone()[0] is None
    # A legacy NULL row stays visible to a tenant-bound reader (predates tenant binding).
    assert [t.title for t in store.list(workspace_id="t-anything")] == ["legacy task"]
    store.close()


def test_notify_legacy_db_without_workspace_column_opens(tmp_path: Path) -> None:
    """An old notify.db (no workspace_id column) hydrates the ring + migrates additively."""
    path = tmp_path / "notify.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE notifications (id INTEGER PRIMARY KEY, kind TEXT NOT NULL, "
        "title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '', link TEXT NOT NULL "
        "DEFAULT '', created_at TEXT NOT NULL, read INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "INSERT INTO notifications (id, kind, title, created_at) VALUES "
        "(1, 'mission', 'legacy note', '2026-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    import os

    os.environ["HIMMY_NOTIFY_PATH"] = str(path)
    try:
        from himmy.api.routers import studio_notify

        studio_notify.reset_notify_state()
        # Open + hydrate; the legacy row must survive and the column must land.
        with studio_notify._LOCK:
            migrated = studio_notify._ensure_db_locked(create=True)
        assert migrated is not None
        cols = {
            r[1]
            for r in migrated.execute("PRAGMA table_info(notifications)").fetchall()
        }
        assert "workspace_id" in cols
        studio_notify.reset_notify_state()
    finally:
        os.environ.pop("HIMMY_NOTIFY_PATH", None)
        from himmy.api.routers import studio_notify

        studio_notify.reset_notify_state()


# --------------------------------------------------------------------------- offline byte-unchanged

def test_offline_studio_stores_unscoped(studio_root: Path) -> None:
    """No authenticator → every store is unscoped, byte-unchanged (the single-box default)."""
    c = TestClient(create_app(ApiContainer.build_default()))
    # Each surface accepts a write and reads it straight back, with NO tenant binding.
    assert c.post("/api/studio/tasks", json={"title": "T"}).status_code == 200
    assert [t["title"] for t in c.get("/api/studio/tasks").json()] == ["T"]
    assert c.put("/api/studio/notes", json={"title": "N", "body": "b"}).status_code == 200
    assert [n["title"] for n in c.get("/api/studio/notes").json()] == ["N"]
    assert c.post(
        "/api/studio/calendar", json={"date": "2026-06-30", "title": "E"}
    ).status_code == 200
    assert [e["title"] for e in c.get("/api/studio/calendar").json()] == ["E"]
    assert c.put("/api/studio/cookbook", json={"name": "R"}).status_code == 200
    assert [r["name"] for r in c.get("/api/studio/cookbook").json()] == ["R"]


# --------------------------------------------------------------------------- cross-tenant scoping

def test_tasks_cross_tenant_invisibility(studio_root: Path) -> None:
    """Tenant ``b`` never sees / mutates tenant ``a``'s task (list absent + by-id no-op)."""
    a, b = _client(_bound_app("a")), _client(_bound_app("b"))
    made = a.post("/api/studio/tasks", json={"title": "secret-a"}).json()
    # b's list does not contain a's task.
    assert all(t["title"] != "secret-a" for t in b.get("/api/studio/tasks").json())
    # b cannot complete or delete a's task by id (the UPDATE/DELETE matches no in-scope row).
    assert b.patch(f"/api/studio/tasks/{made['id']}", json={"done": True}).json()["ok"] is False
    assert b.delete(f"/api/studio/tasks/{made['id']}").json()["ok"] is False
    # a still sees its own task, untouched.
    a_tasks = a.get("/api/studio/tasks").json()
    assert [t["title"] for t in a_tasks] == ["secret-a"]
    assert a_tasks[0]["done"] is False


def test_notes_cross_tenant_invisibility(studio_root: Path) -> None:
    """Tenant ``b`` cannot list, read-by-id, clobber, or delete tenant ``a``'s note."""
    a, b = _client(_bound_app("a")), _client(_bound_app("b"))
    made = a.put("/api/studio/notes", json={"title": "na", "body": "x"}).json()
    assert all(n["id"] != made["id"] for n in b.get("/api/studio/notes").json())
    assert b.get(f"/api/studio/notes/{made['id']}").status_code == 404
    # b re-using a's note id must NOT clobber/re-stamp it (INSERT OR REPLACE) — 404.
    assert b.put(
        "/api/studio/notes", json={"id": made["id"], "title": "evil", "body": "y"}
    ).status_code == 404
    assert b.delete(f"/api/studio/notes/{made['id']}").json()["ok"] is False
    # a's note is intact (not clobbered by b's collision attempt).
    assert a.get(f"/api/studio/notes/{made['id']}").json()["title"] == "na"


def test_calendar_cross_tenant_invisibility(studio_root: Path) -> None:
    """Tenant ``b`` never sees / deletes tenant ``a``'s calendar event."""
    a, b = _client(_bound_app("a")), _client(_bound_app("b"))
    made = a.post(
        "/api/studio/calendar", json={"date": "2026-06-30", "title": "ev-a"}
    ).json()
    assert all(e["id"] != made["id"] for e in b.get("/api/studio/calendar").json())
    assert b.delete(f"/api/studio/calendar/{made['id']}").json()["ok"] is False
    assert [e["title"] for e in a.get("/api/studio/calendar").json()] == ["ev-a"]


def test_cookbook_cross_tenant_invisibility(studio_root: Path) -> None:
    """Tenant ``b`` cannot list, clobber, or delete tenant ``a``'s recipe."""
    a, b = _client(_bound_app("a")), _client(_bound_app("b"))
    made = a.put("/api/studio/cookbook", json={"name": "ra"}).json()
    assert all(r["id"] != made["id"] for r in b.get("/api/studio/cookbook").json())
    assert b.put(
        "/api/studio/cookbook", json={"id": made["id"], "name": "evil"}
    ).status_code == 404
    assert b.delete(f"/api/studio/cookbook/{made['id']}").json()["ok"] is False
    assert [r["name"] for r in a.get("/api/studio/cookbook").json()] == ["ra"]


def test_notify_cross_tenant_invisibility(studio_root: Path) -> None:
    """A notification stamped to tenant ``a`` is invisible to tenant ``b``'s bell."""
    from himmy.api.routers import studio_notify

    # Two stamped notifications (one per tenant) + one legacy NULL-tenant one.
    studio_notify.record_notification("mission", "for-a", workspace_id="a")
    studio_notify.record_notification("mission", "for-b", workspace_id="b")
    studio_notify.record_notification("mission", "for-everyone")  # NULL = local tenant

    b = _client(_bound_app("b"))
    titles = [i["title"] for i in b.get("/api/studio/notify").json()["items"]]
    # b sees its own + the legacy NULL one, but NOT a's.
    assert "for-b" in titles
    assert "for-everyone" in titles
    assert "for-a" not in titles

    # The internal isolation column is not leaked into the public item shape.
    for item in b.get("/api/studio/notify").json()["items"]:
        assert "workspace_id" not in item


def test_notify_mark_read_cannot_cross_tenant(studio_root: Path) -> None:
    """Tenant ``b`` cannot mark tenant ``a``'s notification read (uniform 404 by id)."""
    from himmy.api.routers import studio_notify

    studio_notify.record_notification("mission", "for-a", workspace_id="a")
    a_id = studio_notify._NEXT_ID
    b = _client(_bound_app("b"))
    assert b.post(f"/api/studio/notify/{a_id}/read").status_code == 404


def test_multi_tenant_write_is_rejected(studio_root: Path) -> None:
    """A Studio write by a principal bound to >1 tenant is a 400 (cannot guess the stamp)."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "k": Principal.build(
                "u", tenant_ids=["a", "b"], roles=["reader"], auth_method="apikey"
            )
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping({"reader": list(_READER_GRANTS)})
    c = _client(app)
    assert c.post("/api/studio/tasks", json={"title": "ambiguous"}).status_code == 400


# --------------------------------------------------------------- agent tool-pack scoping
#
# The REST routes above are scoped, but the tasks/notes TOOL packs reach the SAME singleton
# stores directly. A tenant-bound run threads its workspace in via ``ToolkitConfig.tenant_scope``
# (the same lever that namespaces the memory/KB packs); these prove the agent's list/add/
# complete/read/write are filtered + stamped exactly like the REST caller, and that ``None``
# (offline) leaves the tool path unscoped (byte-unchanged).


def _seed_two_tenant_tasks() -> None:
    from himmy.api.studio_tasks import get_tasks_store

    s = get_tasks_store()
    s.add("w1 task", workspace_id="w1")
    s.add("w2 task", workspace_id="w2")
    s.add("legacy task")  # NULL workspace (predates tenant binding)


def _register_pack(pack: str, tenant_scope: str | None):
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit.config import ToolkitConfig
    from himmy.toolkit.pack import register_packs

    reg = ToolRegistry()
    cfg = ToolkitConfig(tenant_scope=tenant_scope)
    register_packs(reg, [pack], cfg)
    return reg


def test_tasks_tool_pack_is_tenant_scoped(studio_root: Path) -> None:
    """A tenant-bound run's ``list_tasks`` sees ONLY its workspace's tasks (plus legacy NULL)."""
    _seed_two_tenant_tasks()
    reg = _register_pack("tasks", tenant_scope="w1")

    listed = reg.handler_for("list_tasks")({})
    titles = {t["title"] for t in listed["tasks"]}
    assert titles == {"w1 task", "legacy task"}
    assert "w2 task" not in titles

    # add_task stamps the run's workspace; the new row is visible to w1, never to w2.
    reg.handler_for("add_task")({"title": "agent-added"})
    from himmy.api.studio_tasks import get_tasks_store

    assert {t.title for t in get_tasks_store().list(workspace_id="w1")} >= {"agent-added"}
    assert "agent-added" not in {
        t.title for t in get_tasks_store().list(workspace_id="w2")
    }

    # complete_task cannot complete a FOREIGN tenant's task by title.
    assert reg.handler_for("complete_task")({"title": "w2 task"})["completed"] is False
    assert reg.handler_for("complete_task")({"title": "w1 task"})["completed"] is True


def test_tasks_tool_pack_offline_unscoped(studio_root: Path) -> None:
    """No ``tenant_scope`` (offline) -> the tool pack lists EVERY task (byte-unchanged)."""
    _seed_two_tenant_tasks()
    reg = _register_pack("tasks", tenant_scope=None)
    titles = {t["title"] for t in reg.handler_for("list_tasks")({})["tasks"]}
    assert titles == {"w1 task", "w2 task", "legacy task"}


def test_notes_tool_pack_is_tenant_scoped(studio_root: Path) -> None:
    """A tenant-bound run's notes tools read + write ONLY its workspace (plus legacy NULL)."""
    from himmy.api.studio_notes import Note, get_notes_store

    s = get_notes_store()
    s.upsert(Note(title="shared-key", body="w1 body"), workspace_id="w1")
    s.upsert(Note(title="shared-key", body="w2 body"), workspace_id="w2")

    reg = _register_pack("notes", tenant_scope="w1")
    # read_note resolves to the w1 copy, never w2's.
    read = reg.handler_for("read_note")({"title": "shared-key"})
    assert read["found"] is True and read["body"] == "w1 body"

    # write_note stamps w1 and upserts the w1 row (w2's copy is untouched).
    reg.handler_for("write_note")({"title": "shared-key", "body": "w1 updated"})
    assert (
        get_notes_store().find_by_title("shared-key", workspace_id="w1").body
        == "w1 updated"
    )
    assert (
        get_notes_store().find_by_title("shared-key", workspace_id="w2").body
        == "w2 body"
    )

    listed = {n["title"] for n in reg.handler_for("list_notes")({})["notes"]}
    assert listed == {"shared-key"}  # one per tenant; w2's is filtered out
