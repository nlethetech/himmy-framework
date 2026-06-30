"""Within-tenant cross-USER BOLA for the Studio singleton/conversation stores (rbac r5).

The cwd-keyed singleton Studio stores (tasks/notes/calendar/cookbook/chats/projects) historically
scoped their HTTP routes by the TENANT axis ONLY (``studio_tenant_filter`` /
``studio_write_workspace``), never the SUBJECT axis. So in a ``subject_scoped`` deployment — the
documented per-user model where many users share ONE tenant — user *alice* could create a task /
note / event / recipe / chat / project and user *bob* (same tenant, different subject) could READ
and MUTATE it: the exact within-tenant cross-USER BOLA ``subject_scoped`` is meant to close (and
that the runs/memory/KB surfaces already defend).

The fix routes these stores through the subject-axis-aware
:func:`himmy.api.auth.context.singleton_write_workspace` /
:func:`~himmy.api.auth.context.singleton_read_filter`, which fold the within-tenant subject into the
SAME combined ``t:<tenant>:s:<subject>`` owner token the agent tool packs use
(:meth:`ToolkitConfig.scoped_pack_workspace`) — so alice's HTTP write lands in her own partition,
bob cannot see/mutate it, AND alice's own agent-created rows are visible to her GUI (no split-brain).

These tests FAIL on the tenant-only routes (bob reads/deletes alice's rows) and PASS once the subject
axis is enforced. The offline / non-subject-scoped path is asserted byte-unchanged elsewhere
(:mod:`tests.api.test_studio_store_tenant_scope`); here we additionally pin the two-subject isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.api.auth.apikey import ApiKeyAuthenticator
from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy

_GRANTS = [
    "studio.console:read",
    "studio.tasks:read",
    "studio.tasks:write",
    "studio.notes:read",
    "studio.notes:write",
    "studio.calendar:read",
    "studio.calendar:write",
    "studio.cookbook:read",
    "studio.cookbook:write",
    "studio.chats:read",
    "studio.chats:write",
    "studio.projects:read",
    "studio.projects:write",
]


@pytest.fixture
def studio_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every singleton + conversation Studio store at a fresh tmp root + reset singletons."""
    from himmy.api import (
        studio_calendar,
        studio_chats,
        studio_cookbook,
        studio_notes,
        studio_tasks,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_TASKS_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("HIMMY_NOTES_PATH", str(tmp_path / "notes.db"))
    monkeypatch.setenv("HIMMY_CHATS_PATH", str(tmp_path / "chats.db"))
    studio_tasks.reset_tasks_store()
    studio_notes.reset_notes_store()
    studio_calendar.reset_calendar_store()
    studio_cookbook.reset_cookbook_store()
    studio_chats.reset_chats_store()
    yield tmp_path
    studio_tasks.reset_tasks_store()
    studio_notes.reset_notes_store()
    studio_calendar.reset_calendar_store()
    studio_cookbook.reset_cookbook_store()
    studio_chats.reset_chats_store()


def _subject_scoped_app() -> FastAPI:
    """One tenant ("acme"), two subject-scoped users (alice, bob) sharing it."""
    app = create_app(ApiContainer.build_default())
    app.state.authenticator = ApiKeyAuthenticator(
        key_principals={
            "key-alice": Principal.build(
                "alice",
                tenant_ids=["acme"],
                roles=["reader"],
                auth_method="apikey",
                subject_scoped=True,
            ),
            "key-bob": Principal.build(
                "bob",
                tenant_ids=["acme"],
                roles=["reader"],
                auth_method="apikey",
                subject_scoped=True,
            ),
        }
    )
    app.state.access_policy = AccessPolicy.from_mapping({"reader": list(_GRANTS)})
    return app


def _client(app: FastAPI, key: str) -> TestClient:
    c = TestClient(app)
    c.headers.update({"x-himmy-internal-key": key})
    return c


# --------------------------------------------------------------------------- tasks


def test_subject_scoped_peer_cannot_read_or_delete_tasks(studio_root: Path) -> None:
    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    bob = _client(app, "key-bob")

    created = alice.post("/api/studio/tasks", json={"title": "alice-secret"})
    assert created.status_code == 200
    task_id = created.json()["id"]

    # Alice sees her own task.
    assert "alice-secret" in [t["title"] for t in alice.get("/api/studio/tasks").json()]
    # Bob (same tenant, different subject) must NOT see it.
    assert "alice-secret" not in [
        t["title"] for t in bob.get("/api/studio/tasks").json()
    ]
    # Bob cannot complete or delete it (the write filter must miss alice's owner token).
    assert bob.patch(
        f"/api/studio/tasks/{task_id}", json={"done": True}
    ).json() == {"ok": False}
    assert bob.delete(f"/api/studio/tasks/{task_id}").json() == {"ok": False}
    # Alice's row is intact.
    assert "alice-secret" in [t["title"] for t in alice.get("/api/studio/tasks").json()]


# --------------------------------------------------------------------------- notes


def test_subject_scoped_peer_cannot_read_or_overwrite_notes(studio_root: Path) -> None:
    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    bob = _client(app, "key-bob")

    created = alice.put(
        "/api/studio/notes", json={"title": "alice", "body": "ssn 123"}
    )
    assert created.status_code == 200
    note_id = created.json()["id"]

    assert "alice" not in [n["title"] for n in bob.get("/api/studio/notes").json()]
    # Bob cannot read it by id (uniform 404).
    assert bob.get(f"/api/studio/notes/{note_id}").status_code == 404
    # Bob re-using alice's id to overwrite is folded to 404 (no clobber/steal).
    assert (
        bob.put(
            "/api/studio/notes", json={"id": note_id, "title": "bob", "body": "x"}
        ).status_code
        == 404
    )
    # Alice's note body is unchanged.
    assert alice.get(f"/api/studio/notes/{note_id}").json()["body"] == "ssn 123"


# --------------------------------------------------------------------------- calendar


def test_subject_scoped_peer_cannot_read_or_delete_calendar(studio_root: Path) -> None:
    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    bob = _client(app, "key-bob")

    created = alice.post(
        "/api/studio/calendar", json={"date": "2026-07-01", "title": "alice-appt"}
    )
    assert created.status_code == 200
    ev_id = created.json()["id"]

    assert "alice-appt" not in [
        e["title"] for e in bob.get("/api/studio/calendar").json()
    ]
    assert bob.delete(f"/api/studio/calendar/{ev_id}").json() == {"ok": False}
    assert "alice-appt" in [e["title"] for e in alice.get("/api/studio/calendar").json()]


# --------------------------------------------------------------------------- cookbook


def test_subject_scoped_peer_cannot_read_or_delete_cookbook(studio_root: Path) -> None:
    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    bob = _client(app, "key-bob")

    created = alice.put("/api/studio/cookbook", json={"name": "alice-recipe"})
    assert created.status_code == 200
    recipe_id = created.json()["id"]

    assert "alice-recipe" not in [
        r["name"] for r in bob.get("/api/studio/cookbook").json()
    ]
    assert bob.delete(f"/api/studio/cookbook/{recipe_id}").json() == {"ok": False}
    assert "alice-recipe" in [
        r["name"] for r in alice.get("/api/studio/cookbook").json()
    ]


# --------------------------------------------------------------------------- chats


def test_subject_scoped_peer_cannot_read_or_delete_chats(studio_root: Path) -> None:
    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    bob = _client(app, "key-bob")

    created = alice.post(
        "/api/studio/chats",
        json={
            "title": "alice-chat",
            "messages": [{"role": "user", "text": "secret transcript"}],
        },
    )
    assert created.status_code == 200
    sid = created.json()["id"]

    # Bob cannot list or read alice's saved chat transcript.
    assert "alice-chat" not in [c["title"] for c in bob.get("/api/studio/chats").json()]
    assert bob.get(f"/api/studio/chats/{sid}").status_code == 404
    assert bob.delete(f"/api/studio/chats/{sid}").json() == {"ok": False}
    # Alice still reads her own full transcript.
    detail = alice.get(f"/api/studio/chats/{sid}").json()
    assert any(m["text"] == "secret transcript" for m in detail["messages"])


# --------------------------------------------------------------------------- projects


def test_subject_scoped_peer_cannot_read_or_mutate_projects(studio_root: Path) -> None:
    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    bob = _client(app, "key-bob")

    created = alice.post("/api/studio/projects", json={"name": "alice-project"})
    assert created.status_code == 200
    pid = created.json()["id"]

    assert "alice-project" not in [
        p["name"] for p in bob.get("/api/studio/projects").json()
    ]
    assert bob.get(f"/api/studio/projects/{pid}").status_code == 404
    assert bob.patch(
        f"/api/studio/projects/{pid}", json={"name": "stolen"}
    ).status_code == 404
    assert alice.get(f"/api/studio/projects/{pid}").json()["name"] == "alice-project"


def test_project_chat_count_excludes_foreign_subject_conversations(
    studio_root: Path,
) -> None:
    """A peer attaching its own chat to alice's project id must not inflate her scoped count.

    Regression for the unscoped ``project_chat_count`` aggregate (finding 3): the PATCH response's
    ``chat_count`` is derived from a now-workspace-scoped count, so a conversation a different owner
    attached to the same project id is excluded.
    """
    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    bob = _client(app, "key-bob")

    pid = alice.post("/api/studio/projects", json={"name": "p"}).json()["id"]
    # Bob saves a conversation in HIS partition but referencing alice's project id.
    bob.post(
        "/api/studio/chats",
        json={
            "project_id": pid,
            "title": "bob-chat",
            "messages": [{"role": "user", "text": "hi"}],
        },
    )
    # Alice's PATCH response carries chat_count from the SCOPED count → bob's chat is excluded.
    patched = alice.patch(
        f"/api/studio/projects/{pid}", json={"description": "touch"}
    )
    assert patched.status_code == 200
    assert patched.json()["chat_count"] == 0


# --------------------------------------------------------------------------- agent/GUI parity


def test_http_owner_token_matches_agent_pack_scope(studio_root: Path) -> None:
    """The HTTP singleton write stamps the SAME owner token the agent tasks/notes packs use.

    Closes the split-brain (finding 2): a subject-scoped user's agent-created rows
    (``t:<tenant>:s:<subject>``) and HTTP-created rows must share ONE owner so the GUI surfaces the
    agent's rows and a co-tenant peer sees neither.
    """
    from himmy.toolkit.config import ToolkitConfig

    app = _subject_scoped_app()
    alice = _client(app, "key-alice")
    # The HTTP write stamps the combined owner token.
    created = alice.post("/api/studio/tasks", json={"title": "x"})
    from himmy.api.studio_tasks import get_tasks_store

    row = (
        get_tasks_store()
        ._conn.execute(
            "SELECT workspace_id FROM tasks WHERE id = ?", (created.json()["id"],)
        )
        .fetchone()
    )
    stamped = row[0]
    # The agent pack's scope for the same (tenant, subject) yields the identical owner.
    pack_owner = ToolkitConfig(
        tenant_scope="acme", subject_scope="alice"
    ).scoped_pack_workspace()
    assert stamped == pack_owner == "t:acme:s:alice"


def test_offline_singleton_stores_remain_unscoped(studio_root: Path) -> None:
    """No authenticator → singleton stores stamp NULL (byte-unchanged single-box path)."""
    c = TestClient(create_app(ApiContainer.build_default()))
    created = c.post("/api/studio/tasks", json={"title": "T"})
    assert created.status_code == 200
    from himmy.api.studio_tasks import get_tasks_store

    row = (
        get_tasks_store()
        ._conn.execute(
            "SELECT workspace_id FROM tasks WHERE id = ?", (created.json()["id"],)
        )
        .fetchone()
    )
    assert row[0] is None
    # And it reads straight back.
    assert "T" in [t["title"] for t in c.get("/api/studio/tasks").json()]
