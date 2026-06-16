"""T2e — the stored-agent (AgentDefRecord) persistence surface, across backends.

Pins the T2.2 "every storage method on all surfaces" rule for the new agent_def store:

* the in-memory facade and the durable SQLite backend round-trip + tenant-scope a
  stored :class:`AgentSpec`, idempotent-create on ``idempotency_key``, list per
  workspace, and tenant-scope delete;
* the SQLite forward migration (v4) lands ``agent_defs`` on a LEGACY pre-v4 file (the
  ``CREATE TABLE IF NOT EXISTS`` base is never reached by an existing db, so the new
  table must arrive via the migration chain);
* the AgentSpec round-trips through the stored JSON projection unchanged.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from himmy.config.agent_spec import AgentSpec
from himmy.services.storage import sqlite as sqlite_mod
from himmy.services.storage.models import AgentDefRecord
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _store_factories(tmp_path: Path) -> list:
    """Both concrete durable+volatile backends that must satisfy the surface."""
    return [
        StorageService(),
        SqliteStorageService(str(tmp_path / "storage.db")),
    ]


def _record(name: str = "analyst", *, workspace: str = "acme", **extra: object) -> AgentDefRecord:
    spec = AgentSpec(name=name, description="A stored analyst.", **extra)  # type: ignore[arg-type]
    return AgentDefRecord.from_spec(spec, workspace_id=workspace)


@pytest.mark.parametrize("backend_index", [0, 1])
def test_round_trip_and_tenant_scope(tmp_path: Path, backend_index: int) -> None:
    """save → get/list round-trips and reads are tenant-scoped on every backend."""
    store = _store_factories(tmp_path)[backend_index]

    async def _scenario() -> None:
        rec = _record()
        saved = await store.save_agent_def(rec)
        assert saved.agent_id == rec.agent_id

        # Same workspace: found. Other workspace: None (404 path).
        got = await store.get_agent_def(rec.agent_id, workspace_id="acme")
        assert got is not None and got.name == "analyst"
        assert await store.get_agent_def(rec.agent_id, workspace_id="globex") is None

        # The AgentSpec round-trips through the stored projection unchanged.
        assert got.agent_spec().name == "analyst"

        # List is workspace-scoped.
        assert [r.agent_id for r in await store.list_agent_defs(workspace_id="acme")] == [
            rec.agent_id
        ]
        assert await store.list_agent_defs(workspace_id="globex") == []

    run_async(_scenario())


@pytest.mark.parametrize("backend_index", [0, 1])
def test_idempotent_create(tmp_path: Path, backend_index: int) -> None:
    """save_agent_def_if_absent returns the prior record for a repeated key."""
    store = _store_factories(tmp_path)[backend_index]

    async def _scenario() -> None:
        first = AgentDefRecord.from_spec(
            AgentSpec(name="a"), workspace_id="acme", idempotency_key="once"
        )
        stored1, created1 = await store.save_agent_def_if_absent(first)
        assert created1 is True

        # A different record object, same (workspace, key): returns the prior, no dup.
        second = AgentDefRecord.from_spec(
            AgentSpec(name="a-changed"), workspace_id="acme", idempotency_key="once"
        )
        stored2, created2 = await store.save_agent_def_if_absent(second)
        assert created2 is False
        assert stored2.agent_id == stored1.agent_id
        assert stored2.name == "a"  # the prior, not the re-submit

        # Exactly one row.
        assert len(await store.list_agent_defs(workspace_id="acme")) == 1

    run_async(_scenario())


@pytest.mark.parametrize("backend_index", [0, 1])
def test_delete_is_tenant_scoped(tmp_path: Path, backend_index: int) -> None:
    """delete removes only within the owning workspace; a wrong-tenant delete is a no-op."""
    store = _store_factories(tmp_path)[backend_index]

    async def _scenario() -> None:
        rec = _record()
        await store.save_agent_def(rec)

        # Wrong workspace: nothing removed.
        assert await store.delete_agent_def(rec.agent_id, workspace_id="globex") is False
        assert await store.get_agent_def(rec.agent_id, workspace_id="acme") is not None

        # Owning workspace: removed; a re-delete is False.
        assert await store.delete_agent_def(rec.agent_id, workspace_id="acme") is True
        assert await store.get_agent_def(rec.agent_id, workspace_id="acme") is None
        assert await store.delete_agent_def(rec.agent_id, workspace_id="acme") is False

    run_async(_scenario())


def test_update_via_resave_keyed_by_agent_id(tmp_path: Path) -> None:
    """A second save with the same agent_id replaces the row (upsert), on SQLite."""
    store = SqliteStorageService(str(tmp_path / "storage.db"))

    async def _scenario() -> None:
        rec = _record(name="v1")
        await store.save_agent_def(rec)
        rec2 = AgentDefRecord.from_spec(
            AgentSpec(name="v2"), workspace_id="acme", agent_id=rec.agent_id
        )
        await store.save_agent_def(rec2)
        got = await store.get_agent_def(rec.agent_id, workspace_id="acme")
        assert got is not None and got.name == "v2"
        # Still exactly one row.
        assert len(await store.list_agent_defs(workspace_id="acme")) == 1

    run_async(_scenario())


def test_legacy_pre_v4_db_gets_agent_defs_via_migration(tmp_path: Path) -> None:
    """A pre-v4 file (no agent_defs table) gains it through the v4 forward migration.

    The ``CREATE TABLE IF NOT EXISTS`` base DDL is a no-op on an existing file, so the
    agent_defs table can only reach a legacy db via the migration chain. We lay down the
    v1 base + drop agent_defs to simulate a pre-v4 file at user_version 3, then open the
    real service and assert the table exists + a write/read round-trips.
    """
    path = str(tmp_path / "legacy.db")
    raw = sqlite3.connect(path)
    raw.executescript(sqlite_mod._SCHEMA)
    # A real v3 db has already been through v2 (run_events.event_type/tool_name columns);
    # add them so this fixture is a faithful v3 shape, not a v1 base mis-stamped at v3.
    raw.execute("ALTER TABLE run_events ADD COLUMN event_type TEXT")
    raw.execute("ALTER TABLE run_events ADD COLUMN tool_name TEXT")
    # Simulate a db provisioned BEFORE v4 (agent_defs did not exist; stamped at v3).
    raw.execute("PRAGMA user_version = 3")
    raw.commit()
    # Sanity: the legacy file genuinely lacks the new table.
    assert (
        raw.execute(
            "SELECT name FROM sqlite_master WHERE name='agent_defs'"
        ).fetchone()
        is None
    )
    raw.close()

    store = SqliteStorageService(path)
    assert store.schema_version >= 4
    # The migration created the table; the store round-trips a record on it.
    rec = _record()
    run_async(store.save_agent_def(rec))
    got = run_async(store.get_agent_def(rec.agent_id, workspace_id="acme"))
    assert got is not None and got.name == "analyst"
