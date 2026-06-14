"""G4: the PUBLIC pooled readiness accessors on PostgresStorageService.

``/readyz`` must probe the durable backend WITHOUT reaching into the private pool or
opening a fresh ``asyncpg.connect``. These tests drive ``ping`` /
``applied_migration_versions`` against a recording fake pool (offline — no real
Postgres) to assert the query shape and the never-raise contract, plus
``code_migration_version`` against the shipped migration chain.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.services.storage.postgres import (
    STORAGE_MIGRATIONS,
    PostgresStorageService,
)


class _FakeAcquire:
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConn:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    def __init__(self, *, fetch: list[Any] | None = None, raise_on: str | None = None):
        self._fetch = fetch or []
        self._raise_on = raise_on
        self.calls: list[str] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append(sql)
        if self._raise_on and self._raise_on in sql:
            raise RuntimeError("boom")
        return "SELECT 1"

    async def fetch(self, sql: str, *args: Any) -> list[Any]:
        self.calls.append(sql)
        if self._raise_on and self._raise_on in sql:
            raise RuntimeError("boom")
        return self._fetch


class _FakePool:
    def __init__(self, conn: _FakeConn) -> None:
        self.conn = conn

    def acquire(self) -> _FakeAcquire:
        return _FakeAcquire(self.conn)


def _service(conn: _FakeConn) -> PostgresStorageService:
    # cipher=None keeps it plaintext/offline; the pool is the fake.
    return PostgresStorageService(pool=_FakePool(conn), cipher=None)


# ---------------------------------------------------------------------- ping


@pytest.mark.asyncio
async def test_ping_true_runs_select_1() -> None:
    conn = _FakeConn()
    svc = _service(conn)
    assert await svc.ping() is True
    assert any("SELECT 1" in sql for sql in conn.calls)


@pytest.mark.asyncio
async def test_ping_false_on_error_never_raises() -> None:
    conn = _FakeConn(raise_on="SELECT 1")
    svc = _service(conn)
    assert await svc.ping() is False


@pytest.mark.asyncio
async def test_ping_false_without_pool() -> None:
    svc = PostgresStorageService(pool=None, cipher=None)
    assert await svc.ping() is False


# --------------------------------------------------- applied_migration_versions


@pytest.mark.asyncio
async def test_applied_versions_reuses_schema_migrations_query() -> None:
    conn = _FakeConn(fetch=[{"version": 3}, {"version": 1}, {"version": 2}])
    svc = _service(conn)
    assert await svc.applied_migration_versions() == [1, 2, 3]
    assert any("SELECT version FROM schema_migrations" in sql for sql in conn.calls)


@pytest.mark.asyncio
async def test_applied_versions_empty_on_error() -> None:
    conn = _FakeConn(raise_on="schema_migrations")
    svc = _service(conn)
    assert await svc.applied_migration_versions() == []


@pytest.mark.asyncio
async def test_applied_versions_empty_without_pool() -> None:
    svc = PostgresStorageService(pool=None, cipher=None)
    assert await svc.applied_migration_versions() == []


# --------------------------------------------------------- code_migration_version


def test_code_migration_version_matches_chain() -> None:
    expected = max(version for version, _, _ in STORAGE_MIGRATIONS)
    assert PostgresStorageService.code_migration_version() == expected
