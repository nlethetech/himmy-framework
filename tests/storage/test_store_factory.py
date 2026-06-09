"""Tests for :class:`StoreFactory` — the durable-default backend selection matrix.

* ``for_cli`` is ALWAYS in-memory and ignores ``HIMMY_DATABASE_URL``.
* ``for_server`` -> SQLite (at ``HIMMY_STORE_PATH``) when no DSN, Postgres when a
  ``postgres://`` DSN is set.
* ``for_context`` follows the ``_SERVER_CONTEXT`` contextvar.

The Postgres connect path is mocked so no real database is needed; ``HIMMY_STORE_PATH``
is monkeypatched into ``tmp_path`` so nothing is written to the repo ``.himmy/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.services.storage import StorageService
from himmy.services.storage.factory import (
    HIMMY_STORE_PATH,
    StoreFactory,
    in_server_context,
    reset_server_context,
    set_server_context,
)
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Reset secrets to the env provider, clear DSN, and steer the store into tmp_path."""
    configure_secrets(None)  # default env-backed provider
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "server" / "storage.db"))


def test_default_store_path_constant() -> None:
    """The exported default lives under ``.himmy/`` for a zero-config server."""
    assert HIMMY_STORE_PATH == ".himmy/storage.db"


def test_for_cli_is_always_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """``for_cli`` returns the in-memory store even with a DSN exported."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")
    store = StoreFactory.for_cli()
    assert isinstance(store, StorageService)
    assert not isinstance(store, SqliteStorageService)


def test_for_server_without_dsn_is_sqlite(tmp_path: Path) -> None:
    """With no DSN, ``for_server`` builds a file-backed SQLite store at the store path."""
    store = run_async(StoreFactory.for_server())
    assert isinstance(store, SqliteStorageService)
    assert store.path == str(tmp_path / "server" / "storage.db")
    assert Path(store.path).parent.is_dir()
    run_async(store.close())


def test_for_server_default_path_when_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``HIMMY_STORE_PATH`` is unset, the SQLite store uses the package default.

    ``chdir`` into ``tmp_path`` first so the relative default ``.himmy/storage.db`` is
    created under the temp dir, never the repo.
    """
    monkeypatch.delenv("HIMMY_STORE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    store = StoreFactory._sqlite_store()
    assert isinstance(store, SqliteStorageService)
    assert store.path == HIMMY_STORE_PATH
    assert (tmp_path / HIMMY_STORE_PATH).is_file()


def test_for_server_with_postgres_dsn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``postgres://`` DSN routes ``for_server`` to a (mocked) Postgres backend."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")

    sentinel = object()
    calls: dict[str, Any] = {}

    class _FakePg:
        async def migrate(self) -> list[int]:
            calls["migrated"] = True
            return [1]

    fake = _FakePg()

    async def _fake_connect(dsn: str, **_: Any) -> _FakePg:
        calls["dsn"] = dsn
        return fake

    import himmy.services.storage.postgres as pg

    monkeypatch.setattr(pg.PostgresStorageService, "connect", _fake_connect)
    assert sentinel is not None  # keep the linter calm about unused locals

    store = run_async(StoreFactory.for_server())
    assert store is fake
    assert calls["dsn"] == "postgres://user@host/db"
    assert calls.get("migrated") is True


def test_for_server_ignores_non_postgres_dsn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-postgres value in the DSN env still yields SQLite (no false Postgres route)."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "sqlite:///ignored.db")
    store = run_async(StoreFactory.for_server())
    assert isinstance(store, SqliteStorageService)
    run_async(store.close())


def test_for_context_follows_server_flag(tmp_path: Path) -> None:
    """``for_context`` is in-memory by default and SQLite inside a server context."""
    assert in_server_context() is False
    assert isinstance(StoreFactory.for_context(), StorageService)

    token = set_server_context(True)
    try:
        assert in_server_context() is True
        durable = StoreFactory.for_context()
        assert isinstance(durable, SqliteStorageService)
    finally:
        reset_server_context(token)
    # The flag is restored after reset.
    assert in_server_context() is False
    assert isinstance(StoreFactory.for_context(), StorageService)
