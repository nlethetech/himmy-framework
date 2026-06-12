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
