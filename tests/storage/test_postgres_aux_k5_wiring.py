"""K5 (offline): the Studio-CRUD + memory + notify Postgres mirrors' routing + plumbing.

No live Postgres runs in the unit lane, so these tests exercise the parts of the K5 mirrors
that do NOT need a real database:

* **Routing.** Under a ``postgres://`` ``HIMMY_DATABASE_URL`` every K5 ``get_*_store``
  resolves the Postgres mirror class, and with no DSN it stays on the durable-SQLite /
  in-memory path byte-for-byte. The notify sink (a module-level deque + best-effort mirror)
  routes its durability to Postgres without creating ``.himmy/notify.db``.
* **Sync-facade plumbing + tenant scoping.** Each mirror drives its async body on the shared
  aux loop and reuses the published pool; a recording fake pool proves the right
  tenant-scoped SQL is issued and the command-tag rowcount is parsed.

The authoritative SQL-semantics coverage (real round-trips, filters, ordering) lives in the
live-PG-gated ``test_postgres_aux_k5_live.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.services.storage.aux_store_factory import reset_aux_store_factory


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    configure_secrets(None)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    reset_aux_store_factory()
    yield
    reset_aux_store_factory()


# --------------------------------------------------------------- recording fake pool


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    """Records every SQL it is handed; returns canned rows / command tags by script."""

    def __init__(self, script: dict[str, Any]) -> None:
        self._script = script
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return self._script.get("execute", "UPDATE 1")

    async def executemany(self, sql: str, args_seq: Any) -> str:
        self.calls.append((sql, tuple(args_seq)))
        return self._script.get("execute", "UPDATE 1")

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        # Allow a per-SQL-fragment override so one fake can serve distinct queries
        # (e.g. notify's MAX(id) row vs its settings-lookup row).
        for fragment, value in self._script.get("fetchrow_by", {}).items():
            if fragment in sql:
                return value
        return self._script.get("fetchrow")

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.calls.append((sql, args))
        return self._script.get("fetch", [])

    async def fetchval(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
        return self._script.get("fetchval")


class _FakePool:
    def __init__(self, script: dict[str, Any] | None = None) -> None:
        self.conn = _FakeConn(script or {})

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)


def _install_pool(monkeypatch: pytest.MonkeyPatch, pool: _FakePool) -> None:
    """Short-circuit aux-pool resolution to the recording fake (loop-affinity fix).

    The aux stores open their OWN aux-loop-bound pool now (they no longer reuse the main-loop
    server pool); these wiring tests assert SQL, so we override ``_AuxPgPool._resolve_async``.
    """
    from himmy.services.storage.postgres_aux import _AuxPgPool

    async def _resolve(self: _AuxPgPool) -> Any:
        return pool

    monkeypatch.setattr(_AuxPgPool, "_resolve_async", _resolve, raising=True)


# --------------------------------------------------------------------- routing wiring


def test_calendar_cookbook_notes_tasks_memory_route_to_pg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://u@h/db")
    from himmy.api import (
        studio_calendar,
        studio_cookbook,
        studio_memory,
        studio_notes,
        studio_tasks,
    )
    from himmy.services.storage.postgres_aux import (
        PostgresCalendarStore,
        PostgresCookbookStore,
        PostgresMemoryStore,
        PostgresNotesStore,
        PostgresTasksStore,
    )

    studio_calendar.reset_calendar_store()
    studio_cookbook.reset_cookbook_store()
    studio_notes.reset_notes_store()
    studio_tasks.reset_tasks_store()
    studio_memory.reset_memory_service()
    try:
        assert isinstance(
            studio_calendar.get_calendar_store(), PostgresCalendarStore
        )
        assert isinstance(
            studio_cookbook.get_cookbook_store(), PostgresCookbookStore
        )
        assert isinstance(studio_notes.get_notes_store(), PostgresNotesStore)
        assert isinstance(studio_tasks.get_tasks_store(), PostgresTasksStore)
        # The memory service's underlying store is the PG mirror.
        studio_memory.get_memory_service()
        assert isinstance(studio_memory._store(), PostgresMemoryStore)
    finally:
        studio_calendar.reset_calendar_store()
        studio_cookbook.reset_cookbook_store()
        studio_notes.reset_notes_store()
        studio_tasks.reset_tasks_store()
        studio_memory.reset_memory_service()


def test_offline_path_stays_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DSN -> every K5 store stays on the SQLite path (byte-for-byte)."""
    from himmy.api import (
        studio_calendar,
        studio_cookbook,
        studio_memory,
        studio_notes,
        studio_tasks,
    )
    from himmy.api.studio_calendar import CalendarStore
    from himmy.api.studio_cookbook import CookbookStore
    from himmy.api.studio_notes import NotesStore
    from himmy.api.studio_tasks import TasksStore
    from himmy.services.memory.store import SqliteMemoryStore

    studio_calendar.reset_calendar_store()
    studio_cookbook.reset_cookbook_store()
    studio_notes.reset_notes_store()
    studio_tasks.reset_tasks_store()
    studio_memory.reset_memory_service()
    try:
        assert isinstance(studio_calendar.get_calendar_store(), CalendarStore)
        assert isinstance(studio_cookbook.get_cookbook_store(), CookbookStore)
        assert isinstance(studio_notes.get_notes_store(), NotesStore)
        assert isinstance(studio_tasks.get_tasks_store(), TasksStore)
        studio_memory.get_memory_service()
        assert isinstance(studio_memory._store(), SqliteMemoryStore)
    finally:
        studio_calendar.reset_calendar_store()
        studio_cookbook.reset_cookbook_store()
        studio_notes.reset_notes_store()
        studio_tasks.reset_tasks_store()
        studio_memory.reset_memory_service()


# --------------------------------------------------------- sync facade + tenant scoping


def test_calendar_emits_tenant_scoped_sql(monkeypatch: pytest.MonkeyPatch) -> None:
    from himmy.api.studio_calendar import CalendarEvent
    from himmy.services.storage.postgres_aux import PostgresCalendarStore

    pool = _FakePool({"fetch": []})
    _install_pool(monkeypatch, pool)

    store = PostgresCalendarStore(tenant="local")
    ev = CalendarEvent(date="2026-06-13", title="meeting")
    store.add(ev)
    store.list(month="2026-06")
    assert store.delete(ev.id) is True
    for sql, args in pool.conn.calls:
        assert "aux_calendar_events" in sql
        assert args[0] == "local"


def test_notes_find_by_title_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    from himmy.api.studio_notes import Note
    from himmy.services.storage.postgres_aux import PostgresNotesStore

    note = Note(title="t", body="b")
    pool = _FakePool(
        {
            "fetchrow": {
                "id": note.id,
                "title": "t",
                "body": "b",
                "updated_at": note.updated_at,
            }
        }
    )
    _install_pool(monkeypatch, pool)

    store = PostgresNotesStore(tenant="local")
    found = store.find_by_title("t")
    assert found is not None
    assert found.title == "t"
    sql, args = pool.conn.calls[0]
    assert "aux_notes" in sql and "title = $2" in sql
    assert args[0] == "local"


def test_tasks_complete_by_title_parses_rowcount(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from himmy.services.storage.postgres_aux import PostgresTasksStore

    won = _FakePool({"execute": "UPDATE 1"})
    _install_pool(monkeypatch, won)
    assert PostgresTasksStore(tenant="local").complete_by_title("x") is True

    lost = _FakePool({"execute": "UPDATE 0"})
    _install_pool(monkeypatch, lost)
    assert PostgresTasksStore(tenant="local").complete_by_title("x") is False


def test_memory_save_and_list_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    from himmy.services.memory.store import MemoryRecord
    from himmy.services.storage.postgres_aux import PostgresMemoryStore

    rec = MemoryRecord(subject_id="alice", text="likes tea")
    pool = _FakePool(
        {
            "fetch": [
                {
                    "memory_id": rec.memory_id,
                    "subject_id": "alice",
                    "kind": "semantic",
                    "text": "likes tea",
                    "metadata": {},
                    "created_at": rec.created_at,
                    "tier": "recall",
                    "valid_from": rec.valid_from,
                    "valid_to": None,
                    "superseded_by": None,
                    "confidence": 1.0,
                    "source": "user",
                    "stable_key": None,
                }
            ]
        }
    )
    _install_pool(monkeypatch, pool)

    store = PostgresMemoryStore(tenant="local")
    store.save(rec)
    out = store.list("alice", active_only=True)
    assert [r.text for r in out] == ["likes tea"]
    # The active_only filter widened the WHERE with valid_to IS NULL, all tenant-scoped.
    list_sql = [s for s, _ in pool.conn.calls if "SELECT" in s][0]
    assert "valid_to IS NULL" in list_sql
    for sql, args in pool.conn.calls:
        assert "aux_memor" in sql  # aux_memories / aux_memory_links
        assert args[0] == "local"


def test_aux_stores_accept_workspace_id_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION (studio-store-drain): the drained REST routes pass ``workspace_id=...`` to
    these singleton stores; under a Postgres DSN that landed on the PG mirror whose methods
    had NO ``workspace_id`` parameter, so every drained read AND write raised ``TypeError`` ->
    HTTP 500 on ANY Postgres deploy. This asserts each PG mirror method now tolerates the
    keyword (the column the routers thread) AND emits the tenant-scoped fragment.
    """
    from himmy.api.studio_calendar import CalendarEvent
    from himmy.api.studio_cookbook import Recipe
    from himmy.api.studio_notes import Note
    from himmy.services.storage.postgres_aux import (
        PostgresCalendarStore,
        PostgresCookbookStore,
        PostgresNotesStore,
        PostgresTasksStore,
    )

    pool = _FakePool({"fetch": [], "fetchrow": None, "execute": "UPDATE 1"})
    _install_pool(monkeypatch, pool)

    # tasks: every drained method must accept workspace_id and not crash.
    tasks = PostgresTasksStore(tenant="local")
    tasks.list(workspace_id="w1")
    tasks.add("buy milk", workspace_id="w1")
    tasks.set_done("id", True, workspace_id="w1")
    tasks.update("id", done=True, workspace_id="w1")
    tasks.complete_by_title("buy milk", workspace_id="w1")
    tasks.delete("id", workspace_id="w1")

    # notes
    notes = PostgresNotesStore(tenant="local")
    notes.list(workspace_id="w1")
    notes.get("id", workspace_id="w1")
    notes.find_by_title("t", workspace_id="w1")
    notes.upsert(Note(title="t", body="b"), workspace_id="w1")
    notes.delete("id", workspace_id="w1")

    # calendar
    cal = PostgresCalendarStore(tenant="local")
    cal.list(month="2026-06", workspace_id="w1")
    cal.add(CalendarEvent(date="2026-06-13", title="x"), workspace_id="w1")
    cal.delete("id", workspace_id="w1")

    # cookbook (incl. the get() the upsert route calls for the cross-tenant 404 check)
    cook = PostgresCookbookStore(tenant="local")
    cook.list(workspace_id="w1")
    cook.get("id", workspace_id="w1")
    cook.upsert(Recipe(name="r"), workspace_id="w1")
    cook.delete("id", workspace_id="w1")

    # A scoped read/mutation emitted the workspace predicate (NULL legacy rows stay visible).
    scoped = [sql for sql, _ in pool.conn.calls if "workspace_id" in sql]
    assert scoped, "no workspace_id-scoped SQL was emitted by the PG mirrors"
    assert any("workspace_id IS NULL" in sql for sql in scoped)
    # The bound workspace value rides as a bind param on at least one scoped call.
    assert any("w1" in args for _, args in pool.conn.calls)


def test_aux_stores_none_workspace_is_unscoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The offline / ``all_tenants`` path passes ``workspace_id=None`` -> NO filter clause, so
    the PG read is byte-unchanged from before the column existed (no ``workspace_id`` predicate
    and no extra bind param beyond tenant)."""
    from himmy.services.storage.postgres_aux import PostgresTasksStore

    pool = _FakePool({"fetch": []})
    _install_pool(monkeypatch, pool)

    PostgresTasksStore(tenant="local").list()  # workspace_id defaults to None
    sql, args = pool.conn.calls[0]
    assert "workspace_id" not in sql
    assert args == ("local",)


def test_memory_satisfies_store_protocol() -> None:
    """The PG mirror duck-types the MemoryStore protocol (drops into MemoryService)."""
    from himmy.services.memory.store import MemoryStore
    from himmy.services.storage.postgres_aux import PostgresMemoryStore

    store = PostgresMemoryStore(tenant="local")
    assert isinstance(store, MemoryStore)


# --------------------------------------------------------------------- notify routing


def test_notify_routes_durability_to_pg_no_sqlite_sidecar(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Under a Postgres DSN the notify sink persists to PG and creates no notify.db."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://u@h/db")
    from pathlib import Path

    from himmy.api.routers import studio_notify

    pool = _FakePool(
        {
            "fetch": [],
            "fetchrow_by": {"MAX(id)": {"m": 0}, "aux_notify_settings": None},
        }
    )
    _install_pool(monkeypatch, pool)
    studio_notify.reset_notify_state()
    try:
        studio_notify.record_notification("mission", "done", body="ok", link="/m")
        # The item was persisted via the PG mirror (an INSERT into aux_notifications).
        insert_calls = [
            (sql, args)
            for sql, args in pool.conn.calls
            if "INSERT INTO aux_notifications" in sql
        ]
        assert insert_calls, "notify item was not mirrored to Postgres"
        assert insert_calls[0][1][0] == "local"  # tenant-scoped
        # No .himmy/notify.db sidecar was created.
        assert not (Path(tmp_path) / ".himmy" / "notify.db").exists()
        # The in-memory deque still holds the live item (read truth unchanged).
        assert studio_notify._NEXT_ID == 1
    finally:
        studio_notify.reset_notify_state()


def test_notify_offline_still_uses_sqlite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """No DSN -> the notify sink keeps its durable SQLite mirror (.himmy/notify.db)."""
    monkeypatch.chdir(tmp_path)
    from pathlib import Path

    from himmy.api.routers import studio_notify

    studio_notify.reset_notify_state()
    try:
        studio_notify.record_notification("mission", "done")
        assert (Path(tmp_path) / ".himmy" / "notify.db").exists()
    finally:
        studio_notify.reset_notify_state()


# ----------------------------------------- rbac-harden(mopup-r1): K5 write scope clause


def _write_clauses(pool: _FakePool, table: str) -> list[str]:
    """The mutating SQL fragments (UPDATE/DELETE) issued against ``table``."""
    return [
        sql
        for sql, _args in pool.conn.calls
        if table in sql and ("UPDATE" in sql or "DELETE" in sql)
    ]


def test_k5_tasks_mutations_use_strict_write_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound-tenant task mutation must NOT match legacy NULL rows.

    Regression for the read-clause-on-write bug: set_done / update / complete_by_title /
    delete must emit the WRITE clause (no ``OR workspace_id IS NULL``) so a tenant-bound
    principal can mutate ONLY its own stamped rows — a legacy/shared NULL row stays
    READ-visible but IMMUTABLE (parity with the SQLite store's scope_clause_write).
    """
    from himmy.services.storage.postgres_aux import PostgresTasksStore

    pool = _FakePool({"execute": "UPDATE 1", "fetchrow": None})
    _install_pool(monkeypatch, pool)
    store = PostgresTasksStore(tenant="local")
    scope = frozenset({"A"})

    store.set_done("t1", True, workspace_id=scope)
    store.complete_by_title("Shared", workspace_id=scope)
    store.delete("t1", workspace_id=scope)
    store.update("t1", done=True, workspace_id=scope)

    clauses = _write_clauses(pool, "aux_tasks")
    assert clauses, "no task mutations were issued"
    for sql in clauses:
        assert "workspace_id IN" in sql, sql
        assert "IS NULL" not in sql, (
            "K5 task WRITE leaked the read clause (mutates legacy NULL-owned rows): " + sql
        )


def test_k5_calendar_notes_cookbook_delete_use_strict_write_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """calendar/notes/cookbook DELETE must use the write clause (no NULL match)."""
    from himmy.api.studio_calendar import CalendarEvent
    from himmy.api.studio_cookbook import Recipe
    from himmy.api.studio_notes import Note
    from himmy.services.storage.postgres_aux import (
        PostgresCalendarStore,
        PostgresCookbookStore,
        PostgresNotesStore,
    )

    scope = frozenset({"A"})

    for store_cls, table, make_id in (
        (PostgresCalendarStore, "aux_calendar_events", CalendarEvent(date="2026-06-13", title="x").id),
        (PostgresNotesStore, "aux_notes", Note(title="t", body="b").id),
        (PostgresCookbookStore, "aux_recipes", Recipe(name="r").id),
    ):
        pool = _FakePool({"execute": "DELETE 1"})
        _install_pool(monkeypatch, pool)
        store = store_cls(tenant="local")
        store.delete(make_id, workspace_id=scope)
        clauses = _write_clauses(pool, table)
        assert clauses, f"no delete issued for {table}"
        for sql in clauses:
            assert "IS NULL" not in sql, (
                f"K5 {table} DELETE leaked the read clause: {sql}"
            )


def test_k5_offline_delete_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """workspace_id=None (offline / all_tenants) issues NO scope fragment — byte-unchanged."""
    from himmy.services.storage.postgres_aux import PostgresTasksStore

    pool = _FakePool({"execute": "DELETE 1"})
    _install_pool(monkeypatch, pool)
    store = PostgresTasksStore(tenant="local")
    store.delete("t1", workspace_id=None)
    for sql in _write_clauses(pool, "aux_tasks"):
        assert "workspace_id" not in sql.split("WHERE", 1)[1], sql


# ------------------------------------ rbac-harden(mopup-r1): K5 upsert preserve-owner


def test_k5_notes_upsert_legacy_null_row_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A by-id upsert onto a legacy NULL-owned row must ABORT — no content write, no re-stamp.

    Regression (mopup-r4 content-tamper): the earlier fix preserved the owner column but the
    ON CONFLICT still REWROTE the row's content (title/body). The store now fails CLOSED for
    any row the caller cannot mutate — it issues NO ``INSERT INTO aux_notes`` and returns the
    existing row unchanged (parity with the SQLite store's abort, and the stated
    read-visible-but-IMMUTABLE invariant for legacy NULL rows).
    """
    from himmy.api.studio_notes import Note
    from himmy.services.storage.postgres_aux import PostgresNotesStore

    # Existing row is NULL-owned (legacy/shared) -> not writable by tenant 'A'. Both the
    # _existing_unwritable probe and the abort-path SELECT * read this canned full row.
    pool = _FakePool(
        {
            "fetchrow": {
                "id": "shared-note",
                "title": "orig-title",
                "body": "orig-body",
                "updated_at": "2026-01-01T00:00:00Z",
                "workspace_id": None,
            },
            "execute": "INSERT 1",
        }
    )
    _install_pool(monkeypatch, pool)
    store = PostgresNotesStore(tenant="local")
    returned = store.upsert(Note(id="shared-note", title="t", body="b"), workspace_id="A")

    inserts = [args for sql, args in pool.conn.calls if "INSERT INTO aux_notes" in sql]
    assert not inserts, (
        "K5 notes upsert clobbered a legacy NULL-owned row's content (should abort)"
    )
    # The caller gets the EXISTING row back, unchanged (not its attacker-supplied body).
    assert returned.body == "orig-body" and returned.title == "orig-title"


def test_k5_notes_upsert_stamps_own_new_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer DOES stamp its own workspace on a new / own row (no false preservation)."""
    from himmy.api.studio_notes import Note
    from himmy.services.storage.postgres_aux import PostgresNotesStore

    # Fresh row (no existing row -> SELECT returns None): the writer stamps its own 'A'.
    pool = _FakePool({"fetchrow": None, "execute": "INSERT 1"})
    _install_pool(monkeypatch, pool)
    store = PostgresNotesStore(tenant="local")
    store.upsert(Note(id="fresh-note", title="t", body="b"), workspace_id="A")
    inserts = [args for sql, args in pool.conn.calls if "INSERT INTO aux_notes" in sql]
    assert inserts and inserts[0][-1] == "A", "writer lost its stamp on a fresh row"

    # Own row (existing row already owned by 'A'): writer keeps its stamp.
    pool2 = _FakePool({"fetchrow": {"workspace_id": "A"}, "execute": "INSERT 1"})
    _install_pool(monkeypatch, pool2)
    store2 = PostgresNotesStore(tenant="local")
    store2.upsert(Note(id="own-note", title="t", body="b"), workspace_id="A")
    inserts2 = [
        args for sql, args in pool2.conn.calls if "INSERT INTO aux_notes" in sql
    ]
    assert inserts2 and inserts2[0][-1] == "A", (
        "writer lost its own stamp on a row it owns"
    )


def test_k5_notes_upsert_offline_no_select(monkeypatch: pytest.MonkeyPatch) -> None:
    """workspace_id=None (offline) issues NO preserve-owner SELECT — byte-unchanged."""
    from himmy.api.studio_notes import Note
    from himmy.services.storage.postgres_aux import PostgresNotesStore

    pool = _FakePool({"execute": "INSERT 1"})
    _install_pool(monkeypatch, pool)
    store = PostgresNotesStore(tenant="local")
    store.upsert(Note(id="n", title="t", body="b"), workspace_id=None)
    selects = [sql for sql, _ in pool.conn.calls if "SELECT workspace_id" in sql]
    assert not selects, "offline upsert issued a preserve-owner SELECT (should not)"
