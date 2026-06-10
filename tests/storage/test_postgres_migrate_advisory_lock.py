"""Connection-seam tests for the migrate() advisory lock (no database required).

The migration runner in :mod:`himmy.services.storage.postgres` must serialise
concurrent migrators across processes via a session-level ``pg_advisory_lock``:
two services booting simultaneously may both call ``migrate()``, and exactly one
may apply each migration. These tests drive ``migrate()`` against a fake
asyncpg-shaped pool whose connections share one in-memory "database" — each fake
call yields to the event loop, so WITHOUT the lock the two runners interleave and
both apply the base schema (the bug). The real-Postgres counterpart lives in
``test_postgres_storage_service.py`` (DSN-gated).
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from himmy.services.storage.postgres import (
    STORAGE_MIGRATIONS,
    PostgresStorageService,
)
from tests.conftest import run_async

#: Statements that belong to real migrations (vs. the bootstrap CREATE TABLE,
#: which also mentions ``schema_migrations``).
_MIGRATION_STATEMENTS = {s for _, _, stmts in STORAGE_MIGRATIONS for s in stmts}


class _FakeDatabase:
    """Shared state simulating the migration-relevant Postgres surface.

    One instance is shared by every connection of every pool, mirroring a single
    database reached by multiple processes. The advisory lock is an
    ``asyncio.Lock`` so a second locker genuinely blocks, like ``pg_advisory_lock``.
    """

    def __init__(self, *, fail_on_apply: bool = False) -> None:
        self.migrations: dict[int, str] = {}
        self.applied_statements: list[str] = []
        self.advisory_lock = asyncio.Lock()
        self.lock_acquires = 0
        self.lock_releases = 0
        self.fail_on_apply = fail_on_apply


class _FakeTransaction:
    async def __aenter__(self) -> _FakeTransaction:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeConnection:
    """Just enough of asyncpg's Connection for ``migrate()``."""

    def __init__(self, db: _FakeDatabase) -> None:
        self._db = db

    async def execute(self, query: str, *args: Any) -> str:
        # Yield first so two concurrent migrators interleave at every statement
        # boundary — exactly the cross-process race the advisory lock must close.
        await asyncio.sleep(0)
        if "pg_advisory_lock" in query:
            await self._db.advisory_lock.acquire()
            self._db.lock_acquires += 1
        elif "pg_advisory_unlock" in query:
            self._db.advisory_lock.release()
            self._db.lock_releases += 1
        elif "INSERT INTO schema_migrations" in query:
            version, name = args
            # ON CONFLICT (version) DO NOTHING
            self._db.migrations.setdefault(version, name)
        elif query in _MIGRATION_STATEMENTS:
            if self._db.fail_on_apply:
                raise RuntimeError("simulated migration failure")
            self._db.applied_statements.append(query)
        # else: the bootstrap CREATE TABLE IF NOT EXISTS — a no-op here.
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        return [{"version": v} for v in self._db.migrations]

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()


class _FakeAcquire:
    def __init__(self, db: _FakeDatabase) -> None:
        self._db = db

    async def __aenter__(self) -> _FakeConnection:
        return _FakeConnection(self._db)

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakePool:
    def __init__(self, db: _FakeDatabase) -> None:
        self._db = db

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self._db)


def test_concurrent_migrate_applies_each_migration_once() -> None:
    """Two simultaneous migrate() calls serialise: one applies, the other no-ops."""

    async def scenario() -> None:
        db = _FakeDatabase()
        # Two services with separate pools over one shared database == two
        # processes booting against the same Postgres.
        first = PostgresStorageService(pool=_FakePool(db))
        second = PostgresStorageService(pool=_FakePool(db))

        results = await asyncio.gather(first.migrate(), second.migrate())

        # Exactly one runner applied the base schema; the loser re-read
        # schema_migrations under the lock and found nothing left to do.
        assert sorted(results, key=len) == [[], [1]]
        assert db.applied_statements.count(STORAGE_MIGRATIONS[0][2][0]) == 1
        assert db.migrations == {1: "base_schema"}
        # Both runners locked and unlocked; nothing is left held.
        assert db.lock_acquires == 2
        assert db.lock_releases == 2
        assert not db.advisory_lock.locked()

    run_async(scenario())


def test_migrate_fast_path_locks_and_unlocks_once() -> None:
    """An already-migrated database costs one lock/unlock round trip and no DDL."""

    async def scenario() -> None:
        db = _FakeDatabase()
        db.migrations = {1: "base_schema"}
        storage = PostgresStorageService(pool=_FakePool(db))

        assert await storage.migrate() == []
        assert db.applied_statements == []
        assert db.lock_acquires == 1
        assert db.lock_releases == 1
        assert not db.advisory_lock.locked()

    run_async(scenario())


def test_migrate_releases_advisory_lock_on_failure() -> None:
    """A failing migration still releases the advisory lock (no stuck peers)."""

    async def scenario() -> None:
        db = _FakeDatabase(fail_on_apply=True)
        storage = PostgresStorageService(pool=_FakePool(db))

        with pytest.raises(RuntimeError, match="simulated migration failure"):
            await storage.migrate()

        assert db.lock_acquires == 1
        assert db.lock_releases == 1
        assert not db.advisory_lock.locked()

    run_async(scenario())
