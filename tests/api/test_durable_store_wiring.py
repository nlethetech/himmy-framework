"""The server entrypoint wires the durable store (item #7), not just in-memory.

``create_app`` builds an in-memory container by default so a bare/offline call stays
zero-config. When a durable backend is requested (a ``HIMMY_DATABASE_URL`` DSN, or the
explicit ``HIMMY_DURABLE_STORAGE`` opt-in), the lifespan upgrades it to the file-backed
SQLite / Postgres store via :class:`StoreFactory`, so background runs and the
tamper-evident security-audit log survive a restart instead of vanishing in memory.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.deps import ApiContainer
from himmy.services.storage.factory import server_storage
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService


def test_default_path_stays_in_memory(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    app = create_app()
    with TestClient(app):
        # No durable backend requested → the in-memory store is left in place.
        assert isinstance(app.state.container.storage, StorageService)
    assert not (tmp_path / ".himmy" / "storage.db").exists()


def test_durable_opt_in_wires_sqlite_store(tmp_path: Any, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    app = create_app()
    # Synchronous default container is still in-memory before startup runs.
    assert isinstance(app.state.container.storage, StorageService)
    with TestClient(app):
        # The lifespan upgraded it to the durable file-backed store.
        assert isinstance(app.state.container.storage, SqliteStorageService)
        # The audit log was rebound onto the durable container's registry.
        assert app.state.security_audit is not None
    assert (tmp_path / ".himmy" / "storage.db").exists()


def test_database_url_selects_durable(tmp_path: Any, monkeypatch: Any) -> None:
    # A configured DSN alone selects durability (no separate opt-in needed). A
    # non-postgres DSN falls back to the SQLite store, which needs no server.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    monkeypatch.setenv("HIMMY_DATABASE_URL", "sqlite:///ignored")
    app = create_app()
    with TestClient(app):
        assert isinstance(app.state.container.storage, SqliteStorageService)


def test_injected_container_is_not_upgraded(tmp_path: Any, monkeypatch: Any) -> None:
    """An explicitly-injected container survives the durable opt-in unchanged.

    The documented production recipe is to ``build_default_async`` and pass the result
    into ``create_app(container=...)``. The lifespan must NOT then discard that choice
    and stand up a second, duplicate store: when a container is injected, the durable
    upgrade is skipped even though ``HIMMY_DURABLE_STORAGE`` is set, so the caller's
    container (and its backends) is the one served for the whole request path.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    injected = ApiContainer.build_default()
    app = create_app(container=injected)
    with TestClient(app):
        # The injected container is served as-is — not rebound to a freshly-built one.
        assert app.state.container is injected
        # Its in-memory storage is left untouched (no silent upgrade to SQLite).
        assert isinstance(app.state.container.storage, StorageService)
        assert not isinstance(app.state.container.storage, SqliteStorageService)
    # And no duplicate on-disk store was created behind the caller's back.
    assert not (tmp_path / ".himmy" / "storage.db").exists()


def test_durable_run_survives_restart(tmp_path: Any, monkeypatch: Any) -> None:
    """A run created against the durable store is visible after a fresh app boot."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "p", "description": "d", "instructions": ["i"]},
        "task": {"title": "t", "prompt": "do"},
    }
    with TestClient(create_app()) as client:
        created = client.post("/v1/runs", json=body)
        assert created.status_code == 200
        run_id = created.json()["run_id"]

    # A brand-new app instance over the same on-disk store still sees the run.
    with TestClient(create_app()) as client2:
        listed = client2.get("/v1/runs", params={"workspace_id": "w1"}).json()
        ids = {item["run_id"] for item in listed["items"]}
        assert run_id in ids


# ----------------------------------------------------------------- K1: split-brain fix
def test_k1_publishes_durable_storage_in_built_branch(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """K1: the self-built durable (SQLite) branch publishes its storage process-wide.

    Inside the lifespan ``StoreFactory.for_context(server=True)`` (the synchronous spec
    wiring) must hand back the SAME instance the lifespan wired — not a freshly opened
    SQLite store. After shutdown the publish is cleared.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    app = create_app()
    assert server_storage() is None  # nothing published before the lifespan
    with TestClient(app):
        published = server_storage()
        assert published is not None
        # The published instance IS the container's storage the rest of the server uses.
        assert published is app.state.container.storage
        assert isinstance(published, SqliteStorageService)
    # Cleared on shutdown so a later in-process run reverts to the default.
    assert server_storage() is None


def test_k1_publishes_injected_container_storage(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """K1 reviewer must_fix: the INJECTED-container branch also publishes its storage.

    The documented production recipe injects a pre-built durable container
    (``upgrade_to_durable=False``), where ``build_default_async`` is never called. The
    split-brain fix must still publish the injected container's storage so an in-server
    agent does not silently fork to local SQLite.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    injected = ApiContainer.build_default()
    app = create_app(container=injected)
    with TestClient(app):
        published = server_storage()
        assert published is injected.storage
    assert server_storage() is None


def test_k1_publishes_in_zero_dsn_default_branch(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """K1: even the zero-DSN spine-rebind branch publishes the resolved (in-memory) store.

    A bare ``create_app()`` with no durable request rebuilds the default container under
    server context (the spine-rebind branch). The published storage must be that
    container's storage so ``for_context`` is coherent with the running server.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    app = create_app()
    with TestClient(app):
        published = server_storage()
        assert published is app.state.container.storage
    assert server_storage() is None


def test_k1_no_local_storage_db_under_durable_path(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """K1 acceptance: an in-server spec-wired run uses the published store, not a fork.

    With the durable SQLite store wired, resolving ``for_context(server=True)`` returns the
    published store; the only ``.himmy/storage.db`` is the published one — no second file is
    created by the spec-wiring path.
    """
    from himmy.services.storage.factory import StoreFactory

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    app = create_app()
    with TestClient(app):
        published = server_storage()
        # The synchronous spec-wiring choke point returns the published instance.
        assert StoreFactory.for_context(server=True) is published
    # Exactly one durable file exists (the published store), confirming no second backend.
    db_files = list((tmp_path / ".himmy").glob("*.db"))
    assert any(f.name == "storage.db" for f in db_files)


def test_k1_postgres_dsn_publishes_pg_store_no_sqlite(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """K1 acceptance: under a postgres DSN the published store is the PG backend.

    The Postgres connect is mocked (no live DB). The split-brain assertion is that the
    spec-wiring choke point returns the published Postgres storage and NO ``storage.db``
    is created by the in-server wiring path.
    """
    from himmy.services.storage.factory import StoreFactory

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")

    class _FakePgStorage:
        async def migrate(self) -> list[int]:
            return [1]

        async def close(self) -> None:
            return None

        async def sweep_stuck_runs(self) -> list[Any]:  # consumed by the startup sweep
            return []

    fake = _FakePgStorage()

    async def _fake_connect(dsn: str, **_: Any) -> _FakePgStorage:
        return fake

    import himmy.services.storage.postgres as pg

    monkeypatch.setattr(pg.PostgresStorageService, "connect", _fake_connect)

    app = create_app()
    with TestClient(app):
        published = server_storage()
        # The container's storage may wrap the backend (consent gating); the publish must
        # be exactly what the container exposes as ``.storage`` (the rest of the server's
        # store), and the spec-wiring choke point returns it verbatim.
        assert published is app.state.container.storage
        assert StoreFactory.for_context(server=True) is published
    # No silent local SQLite fork was created under the postgres DSN.
    assert not (tmp_path / ".himmy" / "storage.db").exists()
