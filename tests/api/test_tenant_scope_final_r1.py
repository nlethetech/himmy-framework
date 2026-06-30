"""Final-round tenancy regressions: the residual leaks BELOW/BESIDE the centralized gates.

Each test FAILS before the final-r1 data-model fix and PASSES after. Grouped by the confirmed
finding it closes:

* NULL-row WRITE immutability (``scope_clause_write``): a tenant-bound principal can READ a
  legacy ``workspace_id IS NULL`` row but can NEVER mutate / delete / re-stamp (steal) it
  across the tasks / notes / cookbook / calendar / conversation stores.
* Tool-store SUBJECT axis: two ``subject_scoped`` users of ONE tenant get distinct
  memory / KB / tasks / notes namespaces (no cross-USER confused-deputy).
* Studio memory TENANT axis: the process-global memory store is namespaced by the principal's
  workspace, so a tenant-bound recall/list/browse cannot cross tenants.
* Studio chats + projects TENANT axis: the unified conversation store carries a nullable
  ``workspace_id`` and scopes every read/write — the CLI (``NULL``) path stays byte-unchanged.

The OFFLINE invariant (scope ``None`` → no filtering, legacy rows untouched) is asserted
throughout so the zero-config single-box path stays byte-for-byte unchanged.
"""

from __future__ import annotations

from pathlib import Path

from himmy.api.studio_tenant_scope import (
    row_writable_in_scope,
    scope_clause,
    scope_clause_write,
)

# ---------------------------------------------------------------- the write-scope primitive


def test_scope_clause_write_excludes_null_rows() -> None:
    """The WRITE clause never matches a legacy NULL row (unlike the READ clause)."""
    # READ matches the caller's workspace OR a legacy NULL row.
    rclause, rparams = scope_clause("t-a")
    assert "IS NULL" in rclause and rparams == ["t-a"]
    # WRITE matches ONLY the caller's workspace — no OR IS NULL.
    wclause, wparams = scope_clause_write("t-a")
    assert wclause == "workspace_id = ?" and "IS NULL" not in wclause
    assert wparams == ["t-a"]
    # None (offline / all_tenants) is an empty no-op fragment on both.
    assert scope_clause_write(None) == ("", [])
    # An empty frozenset is fail-closed on write (a bound principal binds no tenant).
    assert scope_clause_write(frozenset()) == ("1 = 0", [])


def test_row_writable_in_scope_rejects_null_and_foreign() -> None:
    assert row_writable_in_scope(None, None) is True  # offline: anything writable
    assert row_writable_in_scope("t-a", "t-a") is True  # own row
    assert row_writable_in_scope(None, "t-a") is False  # legacy NULL row: immutable
    assert row_writable_in_scope("t-b", "t-a") is False  # foreign row: immutable


# ---------------------------------------------------------------- tasks: NULL-row immutable


def test_tasks_legacy_null_row_not_mutable_by_bound_tenant(tmp_path: Path) -> None:
    """A bound tenant can SEE a legacy NULL task but cannot complete / update / delete it."""
    from himmy.api.studio_tasks import TasksStore

    store = TasksStore(str(tmp_path / "tasks.db"))
    legacy = store.add("legacy", workspace_id=None)  # NULL-owned (offline / pre-migration)
    # Visible to a bound reader (read clause keeps legacy rows).
    assert [t.title for t in store.list(workspace_id="t-a")] == ["legacy"]
    # But NOT mutable by that tenant (the bug: these returned True before the fix).
    assert store.set_done(legacy.id, True, workspace_id="t-a") is False
    assert store.complete_by_title("legacy", workspace_id="t-a") is False
    assert store.update(legacy.id, priority=3, workspace_id="t-a") is None
    assert store.delete(legacy.id, workspace_id="t-a") is False
    # The legacy row is untouched (still open, still NULL-owned).
    row = store._conn.execute(
        "SELECT done, workspace_id FROM tasks WHERE id = ?", (legacy.id,)
    ).fetchone()
    assert row["done"] == 0 and row["workspace_id"] is None
    # Offline (None) still mutates it — byte-unchanged.
    assert store.delete(legacy.id, workspace_id=None) is True
    store.close()


# ---------------------------------------------------------------- notes: no clobber / steal


def test_notes_bound_tenant_cannot_clobber_or_steal_legacy_note(tmp_path: Path) -> None:
    """``write_note``-shaped flow: a bound tenant writes a FRESH note, never overwrites a NULL one."""
    from himmy.api.studio_notes import Note, NotesStore

    store = NotesStore(str(tmp_path / "notes.db"))
    legacy = Note(title="plan", body="original")
    store.upsert(legacy, workspace_id=None)  # NULL-owned shared note

    # for_write title lookup must NOT match the legacy NULL note for a bound tenant.
    assert store.find_by_title("plan", workspace_id="t-b", for_write=True) is None
    # ... so the tool path creates a NEW tenant-owned note instead of reusing the legacy id.
    fresh = Note(title="plan", body="tenant-b")
    store.upsert(fresh, workspace_id="t-b")
    assert fresh.id != legacy.id
    # The legacy note's body + owner are intact (not clobbered, not re-stamped).
    got = store.get(legacy.id, workspace_id=None)
    assert got is not None and got.body == "original"
    assert store._conn.execute(
        "SELECT workspace_id FROM notes WHERE id = ?", (legacy.id,)
    ).fetchone()["workspace_id"] is None
    store.close()


def test_notes_upsert_onto_foreign_id_preserves_owner(tmp_path: Path) -> None:
    """Re-stamp guard: upserting a bound tenant's body onto a FOREIGN id keeps the foreign owner."""
    from himmy.api.studio_notes import Note, NotesStore

    store = NotesStore(str(tmp_path / "notes.db"))
    a = Note(title="secret", body="A's")
    store.upsert(a, workspace_id="t-a")
    # Tenant B tries to capture A's row by id.
    store.upsert(Note(id=a.id, title="secret", body="hijack"), workspace_id="t-b")
    owner = store._conn.execute(
        "SELECT workspace_id FROM notes WHERE id = ?", (a.id,)
    ).fetchone()["workspace_id"]
    assert owner == "t-a"  # NOT re-stamped to t-b
    store.close()


# ---------------------------------------------------------------- cookbook + calendar writes


def test_cookbook_legacy_recipe_immutable_and_unstealable(tmp_path: Path) -> None:
    from himmy.api.studio_cookbook import CookbookStore, Recipe

    store = CookbookStore(str(tmp_path / "cookbook.db"))
    legacy = Recipe(name="R", prompt="p")
    store.upsert(legacy, workspace_id=None)
    # Bound tenant cannot delete the legacy recipe.
    assert store.delete(legacy.id, workspace_id="t-a") is False
    # Re-stamp via upsert onto the legacy id preserves the NULL owner.
    store.upsert(Recipe(id=legacy.id, name="R", prompt="steal"), workspace_id="t-a")
    assert store._conn.execute(
        "SELECT workspace_id FROM recipes WHERE id = ?", (legacy.id,)
    ).fetchone()["workspace_id"] is None
    store.close()


def test_calendar_legacy_event_not_deletable_by_bound_tenant(tmp_path: Path) -> None:
    from himmy.api.studio_calendar import CalendarEvent, CalendarStore

    store = CalendarStore(str(tmp_path / "cal.db"))
    ev = CalendarEvent(date="2026-07-01", title="legacy")
    store.add(ev, workspace_id=None)
    assert store.delete(ev.id, workspace_id="t-a") is False
    assert store.delete(ev.id, workspace_id=None) is True  # offline still works
    store.close()


# ---------------------------------------------------------------- tool-store SUBJECT axis


def test_toolkit_config_namespaces_by_tenant_and_subject() -> None:
    """memory/KB/pack scopes differ for two subject_scoped users of ONE tenant."""
    from himmy.toolkit.config import ToolkitConfig

    alice = ToolkitConfig(tenant_scope="acme", subject_scope="alice")
    bob = ToolkitConfig(tenant_scope="acme", subject_scope="bob")
    # Memory subject, KB scope, and pack workspace all differ between the two users.
    assert alice.scoped_memory_subject() != bob.scoped_memory_subject()
    assert alice.scoped_kb_keys() != bob.scoped_kb_keys()
    assert alice.scoped_pack_workspace() != bob.scoped_pack_workspace()

    # Tenant-only (no subject) is byte-unchanged: the historical static/tenant scope.
    tenant_only = ToolkitConfig(tenant_scope="acme")
    assert tenant_only.scoped_memory_subject() == "t:acme:default"
    assert tenant_only.scoped_kb_keys() == ("t:acme", "t:acme")
    # The pack workspace stays the bare tenant id (what the packs passed before).
    assert tenant_only.scoped_pack_workspace() == "acme"

    # Fully offline (no tenant, no subject) is the historical static scope.
    offline = ToolkitConfig()
    assert offline.scoped_memory_subject() == "default"
    assert offline.scoped_kb_keys() == ("local", "local")
    assert offline.scoped_pack_workspace() is None


def test_subject_scoped_users_get_isolated_task_partitions(tmp_path: Path) -> None:
    """Two users of one tenant never see each other's tasks via the pack workspace scope."""
    from himmy.api.studio_tasks import TasksStore
    from himmy.toolkit.config import ToolkitConfig

    store = TasksStore(str(tmp_path / "tasks.db"))
    alice_ws = ToolkitConfig(tenant_scope="acme", subject_scope="alice").scoped_pack_workspace()
    bob_ws = ToolkitConfig(tenant_scope="acme", subject_scope="bob").scoped_pack_workspace()
    store.add("alice-task", workspace_id=alice_ws)
    # Bob (same tenant, different subject) does not see alice's task.
    assert [t.title for t in store.list(workspace_id=bob_ws)] == []
    assert [t.title for t in store.list(workspace_id=alice_ws)] == ["alice-task"]
    store.close()


# ---------------------------------------------------------------- conversation store scope


def test_conversations_chats_tenant_scoped_and_cli_null_visible(tmp_path: Path) -> None:
    from himmy.services.storage.conversations import ConversationStore

    store = ConversationStore(str(tmp_path / "conv.db"))
    # CLI write (no workspace) → NULL-owned, visible to all (byte-unchanged).
    store.save_flat(
        conversation_id="cli1",
        title="cli chat",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "hi")],
    )
    # Two tenants' Studio chats.
    store.save_flat(
        conversation_id="a1",
        title="A chat",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "secret A")],
        workspace_id="t-a",
    )
    store.save_flat(
        conversation_id="b1",
        title="B chat",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "secret B")],
        workspace_id="t-b",
    )
    a_ids = {s.conversation_id for s in store.list_summaries(workspace_id=frozenset({"t-a"}))}
    # Tenant A sees its own + the legacy CLI row, NOT tenant B's.
    assert a_ids == {"a1", "cli1"}
    assert store.get_summary("b1", workspace_id=frozenset({"t-a"})) is None
    # Tenant A cannot delete tenant B's chat, nor the legacy CLI chat (write-strict).
    assert store.delete("b1", workspace_id=frozenset({"t-a"})) is False
    assert store.delete("cli1", workspace_id=frozenset({"t-a"})) is False
    # Offline (None) sees everything — byte-unchanged.
    assert len(store.list_summaries(workspace_id=None)) == 3
    store.close()


def test_conversations_projects_tenant_scoped(tmp_path: Path) -> None:
    from himmy.services.storage.conversations import ConversationStore

    store = ConversationStore(str(tmp_path / "conv.db"))
    pa = store.create_project(name="A-proj", workspace_id="t-a")
    pb = store.create_project(name="B-proj", workspace_id="t-b")
    a_ids = {r["id"] for r, _ in store.list_project_rows(workspace_id=frozenset({"t-a"}))}
    assert a_ids == {pa["id"]}
    assert store.get_project_row(pb["id"], workspace_id=frozenset({"t-a"})) is None
    # Tenant A cannot delete tenant B's project.
    assert store.delete_project(pb["id"], workspace_id=frozenset({"t-a"})) is False
    assert store.delete_project(pa["id"], workspace_id=frozenset({"t-a"})) is True
    store.close()


# ---------------------------------------------------------------- offline migration safety


def test_conversation_legacy_db_without_workspace_column_opens(tmp_path: Path) -> None:
    """A pre-migration conversations.db (no workspace_id) opens + migrates additively."""
    import sqlite3

    path = tmp_path / "conv.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE conversations (conversation_id TEXT PRIMARY KEY, thread TEXT NOT NULL, "
        "origin TEXT NOT NULL DEFAULT 'cli', title TEXT, agent_path TEXT, provider TEXT, "
        "project_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);"
        "CREATE TABLE conversation_messages (id TEXT PRIMARY KEY, conversation_id TEXT NOT "
        "NULL, role TEXT NOT NULL, text TEXT NOT NULL, seq INTEGER NOT NULL, created_at TEXT "
        "NOT NULL);"
        "CREATE TABLE projects (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT "
        "NULL DEFAULT '', kb_id TEXT, agent_path TEXT, created_at TEXT NOT NULL, updated_at "
        "TEXT NOT NULL);"
    )
    conn.execute(
        "INSERT INTO conversations (conversation_id, thread, origin, title, created_at, "
        "updated_at) VALUES ('old', '{}', 'cli', 'legacy', '2026-01-01', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    from himmy.services.storage.conversations import ConversationStore

    store = ConversationStore(str(path))
    cols = {
        r["name"]
        for r in store._conn.execute("PRAGMA table_info(conversations)").fetchall()
    }
    assert "workspace_id" in cols
    # The legacy row carries NULL and stays visible to a bound reader (predates binding).
    got = store.get_summary("old", workspace_id=frozenset({"t-x"}))
    assert got is not None and got.title == "legacy"
    store.close()


# ---------------------------------------------------------------- connector subject scope


def test_connector_runtime_namespaces_memory_off_static_default(monkeypatch) -> None:
    """The inbound connector threads its FIXED workspace as subject → off the static scope."""
    from himmy.api.auth.service_principal import connector_service_principal
    from himmy.toolkit.config import ToolkitConfig

    ws = connector_service_principal().default_tenant()
    assert ws is not None  # pinned to LOCAL_WORKSPACE, exactly one tenant
    cfg = ToolkitConfig(tenant_scope=ws)
    # NOT the static shared 'default' / ('local','local') the offline CLI shares.
    assert cfg.scoped_memory_subject() != "default"
    assert cfg.scoped_kb_keys() != ("local", "local")
