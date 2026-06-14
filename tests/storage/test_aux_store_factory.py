"""K2: the shared aux-store backend selector (the single choke point).

Verifies the ``HIMMY_DATABASE_URL``-selects-the-backend decision lives in ONE place, that
:func:`select_aux_store` returns SQLite by default (byte-identical to the pre-K2 path) and
the Postgres variant only when a Postgres builder is supplied AND a Postgres DSN is set,
and that the shared background loop is a single process-wide instance (not one per store).
"""

from __future__ import annotations

import pytest

from himmy.config.secrets import configure_secrets
from himmy.services.storage.aux_store_factory import (
    aux_backend,
    aux_loop,
    aux_pool,
    aux_postgres_enabled,
    reset_aux_store_factory,
    select_aux_store,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset secrets to the env provider, clear the DSN, and tear the loop down."""
    configure_secrets(None)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    reset_aux_store_factory()
    yield
    reset_aux_store_factory()


def test_backend_is_sqlite_without_dsn() -> None:
    """No DSN → the aux backend is SQLite (the zero-config offline default)."""
    assert aux_backend() == "sqlite"
    assert aux_postgres_enabled() is False


def test_backend_is_postgres_with_postgres_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``postgres://`` DSN selects the Postgres aux backend."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")
    assert aux_backend() == "postgres"
    assert aux_postgres_enabled() is True


def test_backend_ignores_non_postgres_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-postgres DSN value never falsely selects Postgres."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "sqlite:///ignored.db")
    assert aux_backend() == "sqlite"


def test_select_returns_sqlite_when_no_pg_builder() -> None:
    """With no Postgres builder, the SQLite builder always wins (today's every caller)."""
    sqlite_marker = object()
    got = select_aux_store(lambda: sqlite_marker)
    assert got is sqlite_marker


def test_select_returns_sqlite_under_postgres_without_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Even under a PG DSN, no Postgres builder → SQLite (the pre-mirror state)."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")
    sqlite_marker = object()
    got = select_aux_store(lambda: sqlite_marker, None)
    assert got is sqlite_marker


def test_select_returns_postgres_when_builder_and_dsn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Postgres builder + a PG DSN routes to the Postgres variant (the K3/K4/K5 path)."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")
    sqlite_marker = object()
    pg_marker = object()
    got = select_aux_store(lambda: sqlite_marker, lambda: pg_marker)
    assert got is pg_marker


def test_select_prefers_sqlite_with_builder_but_no_dsn() -> None:
    """A Postgres builder present but NO DSN still resolves SQLite (offline default)."""
    sqlite_marker = object()
    pg_marker = object()
    got = select_aux_store(lambda: sqlite_marker, lambda: pg_marker)
    assert got is sqlite_marker


def test_pg_builder_not_invoked_on_sqlite_path() -> None:
    """The Postgres builder is lazy — never invoked when SQLite is selected (no asyncpg)."""
    calls = {"pg": 0}

    def _pg() -> object:
        calls["pg"] += 1
        return object()

    select_aux_store(lambda: object(), _pg)  # no DSN → SQLite
    assert calls["pg"] == 0


def test_aux_loop_is_a_shared_singleton() -> None:
    """``aux_loop`` returns ONE shared loop across calls (not N loops/threads)."""
    loop1 = aux_loop()
    loop2 = aux_loop()
    assert loop1 is loop2
    # The shared loop actually runs coroutines.

    async def _echo() -> int:
        return 42

    assert loop1.run(_echo()) == 42


def test_reset_replaces_the_loop() -> None:
    """Resetting the factory tears the loop down so the next call builds a fresh one."""
    loop1 = aux_loop()
    reset_aux_store_factory()
    loop2 = aux_loop()
    assert loop1 is not loop2


def test_aux_pool_is_none_without_published_storage() -> None:
    """With no published server storage, there is no pool to reuse (CLI/offline)."""
    from himmy.services.storage.factory import reset_server_storage

    reset_server_storage()
    assert aux_pool() is None


def test_aux_pool_returns_published_storage_pool() -> None:
    """When a Postgres storage is published, its pool is returned for reuse (one pool)."""
    from himmy.services.storage.factory import (
        reset_server_storage,
        set_server_storage,
    )

    class _FakePgStorage:
        _pool = object()

    fake = _FakePgStorage()
    set_server_storage(fake)
    try:
        assert aux_pool() is _FakePgStorage._pool
    finally:
        reset_server_storage()


def test_aux_pool_null_safe_for_poolless_storage() -> None:
    """A published storage without a ``_pool`` attr (SQLite) yields None, never raises."""
    from himmy.services.storage.factory import (
        reset_server_storage,
        set_server_storage,
    )

    set_server_storage(object())
    try:
        assert aux_pool() is None
    finally:
        reset_server_storage()
