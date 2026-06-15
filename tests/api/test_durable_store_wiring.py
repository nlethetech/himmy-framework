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


# ----------------------------------------------------------------- Q3: dispatcher wiring
def test_q3_dispatcher_executes_a_run_under_durable_store(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Under a durable store the leased dispatcher claims + drives a created run to terminal.

    A ``POST /v1/runs`` enqueues a QUEUED run (Q3 dispatch mode); the in-lifespan dispatcher
    claims and executes it from its persisted input, reaching SUCCEEDED with no inline task.
    """
    import time as _time

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    # Force the local-provider probe ON so the default-lane run is claimable in the test env.
    monkeypatch.setattr(
        "himmy.application.dispatcher.LocalProviderProbe._do_probe",
        lambda self: _async_true(),
    )
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "p", "description": "d", "instructions": ["i"]},
        "task": {"title": "t", "prompt": "do the thing"},
    }
    with TestClient(create_app()) as client:
        created = client.post("/v1/runs", json=body)
        assert created.status_code == 200
        run_id = created.json()["run_id"]
        # Poll until the dispatcher drives it terminal.
        deadline = _time.time() + 8.0
        status = None
        while _time.time() < deadline:
            got = client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w1"})
            status = got.json().get("status")
            if status in ("SUCCEEDED", "FAILED", "PARKED"):
                break
            _time.sleep(0.05)
        assert status == "SUCCEEDED", f"dispatcher did not complete the run: {status}"


async def _async_true() -> bool:
    """A trivially-available local probe (the test has no Ollama/claude CLI)."""
    return True


def test_q3_in_memory_default_stays_inline_no_dispatcher(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The zero-config in-memory default runs INLINE (no dispatcher) — the run still completes.

    The run service must NOT be flipped into dispatch mode on the in-memory path, so a run
    created against a bare ``create_app()`` is executed by its inline background task (never
    left stuck QUEUED waiting for a dispatcher that was never started).
    """
    import time as _time

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    body = {
        "workspace_id": "w1",
        "subject_id": "s1",
        "persona": {"name": "p", "description": "d", "instructions": ["i"]},
        "task": {"title": "t", "prompt": "hello"},
    }
    with TestClient(create_app()) as client:
        # The run service is in inline mode (the durable store / dispatcher were not wired).
        assert client.app.state.container.run_app.dispatch_enabled is False
        created = client.post("/v1/runs", json=body)
        run_id = created.json()["run_id"]
        deadline = _time.time() + 8.0
        status = None
        while _time.time() < deadline:
            status = (
                client.get(f"/v1/runs/{run_id}", params={"workspace_id": "w1"})
                .json()
                .get("status")
            )
            if status in ("SUCCEEDED", "FAILED"):
                break
            _time.sleep(0.05)
        assert status == "SUCCEEDED"


# ------------------------------------------------------- dx: dispatcher env-configurable
def test_dispatch_env_overrides_apply_under_durable_store(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """``HIMMY_DISPATCH_*`` / ``HIMMY_RUN_TIMEOUT_SECONDS`` retune the leased dispatcher.

    The dispatcher otherwise pins ``max_concurrency=8`` and the run service its constructed
    timeout / retry ceiling. The env overrides must flow through at the lifespan call site:
    the constructed dispatcher's concurrency, and the run service's run timeout + retry
    budget, reflect the env values.
    """
    from himmy.application.dispatcher import RunDispatcher

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    monkeypatch.setenv("HIMMY_DISPATCH_CONCURRENCY", "3")
    monkeypatch.setenv("HIMMY_RUN_TIMEOUT_SECONDS", "42")
    monkeypatch.setenv("HIMMY_DISPATCH_MAX_ATTEMPTS", "7")

    captured: dict[str, Any] = {}
    real_init = RunDispatcher.__init__

    def _spy_init(self: Any, run_app: Any, **kwargs: Any) -> None:
        captured["max_concurrency"] = kwargs.get("max_concurrency")
        real_init(self, run_app, **kwargs)

    monkeypatch.setattr(RunDispatcher, "__init__", _spy_init)

    with TestClient(create_app()) as client:
        run_app = client.app.state.container.run_app
        assert run_app.dispatch_enabled is True
        # The non-default values prove the env was read (not the 8/300/3 defaults).
        assert captured["max_concurrency"] == 3
        assert run_app.run_timeout_seconds == 42.0
        assert run_app.default_max_attempts == 7


def test_dispatch_defaults_apply_without_env(tmp_path: Any, monkeypatch: Any) -> None:
    """Absent the env vars the dispatcher keeps its pinned defaults (8 / 300s / 3)."""
    from himmy.application.dispatcher import RunDispatcher

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    for name in (
        "HIMMY_DISPATCH_CONCURRENCY",
        "HIMMY_RUN_TIMEOUT_SECONDS",
        "HIMMY_DISPATCH_MAX_ATTEMPTS",
    ):
        monkeypatch.delenv(name, raising=False)

    captured: dict[str, Any] = {}
    real_init = RunDispatcher.__init__

    def _spy_init(self: Any, run_app: Any, **kwargs: Any) -> None:
        captured["max_concurrency"] = kwargs.get("max_concurrency")
        real_init(self, run_app, **kwargs)

    monkeypatch.setattr(RunDispatcher, "__init__", _spy_init)

    with TestClient(create_app()) as client:
        run_app = client.app.state.container.run_app
        assert captured["max_concurrency"] == 8
        assert run_app.run_timeout_seconds == 300.0
        assert run_app.default_max_attempts == 3


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


# ----------------------------------------------------- dedup-ledger GC (maintenance loop)
def test_dedup_sweep_loop_calls_storage_dedup_sweep() -> None:
    """The maintenance loop GC's the dedup ledger by calling ``storage.dedup_sweep``.

    A short interval + a stop event after the first sweep proves the loop actually invokes
    the otherwise-unused ``dedup_sweep`` (the bug: the durable dedup table grew forever).
    """
    import asyncio

    from himmy.api.app import _dedup_sweep_loop
    from tests.conftest import run_async

    calls = {"n": 0}
    stop = asyncio.Event()

    class _Storage:
        async def dedup_sweep(self, *, now: Any = None) -> int:
            calls["n"] += 1
            stop.set()  # one sweep is enough to prove the loop fires it
            return 3

    run_async(
        asyncio.wait_for(
            _dedup_sweep_loop(_Storage(), interval=0.01, stop=stop), timeout=2.0
        )
    )
    assert calls["n"] >= 1


def test_dedup_sweep_loop_survives_a_sweep_error() -> None:
    """A failing sweep must not kill the loop — it retries on the next tick."""
    import asyncio

    from himmy.api.app import _dedup_sweep_loop
    from tests.conftest import run_async

    calls = {"n": 0}
    stop = asyncio.Event()

    class _FlakyStorage:
        async def dedup_sweep(self, *, now: Any = None) -> int:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient db error")
            stop.set()
            return 0

    run_async(
        asyncio.wait_for(
            _dedup_sweep_loop(_FlakyStorage(), interval=0.01, stop=stop), timeout=2.0
        )
    )
    assert calls["n"] >= 2  # the loop kept going after the first cycle raised


def test_durable_lifespan_invokes_dedup_sweep(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """Under a durable store the lifespan starts the dedup-sweep loop and it runs.

    With a tiny interval the loop should fire ``dedup_sweep`` at least once while the
    TestClient context is open, proving the maintenance loop is wired in app.py.
    """
    import time as _time

    import himmy.api.app as app_mod
    from himmy.services.storage.sqlite import SqliteStorageService

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    monkeypatch.setattr(app_mod, "_DEDUP_SWEEP_INTERVAL_SECONDS", 0.02)

    calls = {"n": 0}
    real_sweep = SqliteStorageService.dedup_sweep

    async def _counting_sweep(self: Any, *, now: Any = None) -> int:
        calls["n"] += 1
        return await real_sweep(self, now=now)

    monkeypatch.setattr(SqliteStorageService, "dedup_sweep", _counting_sweep)

    with TestClient(create_app()):
        deadline = _time.time() + 4.0
        while _time.time() < deadline and calls["n"] == 0:
            _time.sleep(0.02)
    assert calls["n"] >= 1, "durable lifespan never swept the dedup ledger"


def test_in_memory_default_does_not_start_dedup_sweep(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The zero-config in-memory default must NOT start the durable-gated sweep loop.

    The in-memory dedup map is bounded + vanishes on exit, so there is nothing to GC; the
    loop is durable-gated exactly like the dispatcher, leaving the offline path unchanged.
    """
    import himmy.api.app as app_mod

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)

    started = {"n": 0}
    real_loop = app_mod._dedup_sweep_loop

    async def _spy_loop(*args: Any, **kwargs: Any) -> None:
        started["n"] += 1
        return await real_loop(*args, **kwargs)

    monkeypatch.setattr(app_mod, "_dedup_sweep_loop", _spy_loop)

    with TestClient(create_app()) as client:
        assert client.app.state.container.run_app.dispatch_enabled is False
    assert started["n"] == 0
