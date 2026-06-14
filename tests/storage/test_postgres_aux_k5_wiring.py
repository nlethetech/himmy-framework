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


class _FakePool:
    def __init__(self, script: dict[str, Any] | None = None) -> None:
        self.conn = _FakeConn(script or {})

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)


def _install_pool(monkeypatch: pytest.MonkeyPatch, pool: _FakePool) -> None:
    import himmy.services.storage.aux_store_factory as factory

    monkeypatch.setattr(factory, "aux_pool", lambda: pool)


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
