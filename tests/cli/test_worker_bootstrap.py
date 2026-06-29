"""Tests for the ``himmy worker`` standalone run substrate + the run-now durable bridge.

The worker must, with NO API server, bring up the SAME durable run substrate the FastAPI
lifespan builds: a durable run store, ``enable_dispatch``, a started :class:`RunDispatcher`,
the routine container provider, and the :class:`RoutineScheduler` — then tear it all down on
shutdown. These tests drive the shared bootstrap + the worker async body directly (no real
signals / no infinite block) and assert the wiring, not model prose.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from himmy.api import routines as svc
from himmy.api.studio_runs import reset_run_store
from tests.conftest import run_async


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project root with an agent spec + fresh cwd-keyed durable stores."""
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / ".himmy" / "storage.db"))
    # Force the durable run store so the dispatcher engages (no Postgres needed).
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    svc.reset_routines_store()
    svc.reset_scheduler()
    reset_run_store()
    yield tmp_path
    svc.reset_routines_store()
    svc.reset_scheduler()


# ----------------------------------------------------------- shared bootstrap


def test_start_run_substrate_engages_dispatch_on_durable_store(
    workspace: Path,
) -> None:
    """Under a durable store the substrate enables dispatch + starts the dispatcher.

    This is the worker's core promise: ``enable_dispatch`` is a fiction unless a durable
    store is wired AND the dispatcher is started — assert all of it actually happened.
    """
    from himmy.api.routines import resolve_routine_container
    from himmy.api.runtime_bootstrap import (
        build_durable_container,
        start_run_substrate,
        stop_run_substrate,
    )
    from himmy.services.storage.factory import (
        reset_server_context,
        set_server_context,
    )

    async def _drive() -> None:
        token = set_server_context(True)
        container, built = await build_durable_container()
        assert built, "durable container was not built under HIMMY_DURABLE_STORAGE"
        substrate = await start_run_substrate(container, install_providers=True)
        try:
            assert substrate.backend == "sqlite"
            assert substrate.dispatch_on is True
            assert substrate.dispatcher is not None
            # The run service is now in leased-dispatch mode (create_run ENQUEUES).
            assert container.run_app.dispatch_enabled is True
            # The routine container provider resolves the SAME container (so agent_id
            # routines run offline — the whole point of the worker).
            assert resolve_routine_container() is container
        finally:
            await stop_run_substrate(substrate, clear_providers=True)
            reset_server_context(token)
        # Providers cleared on teardown so a later in-process run is not poisoned.
        assert resolve_routine_container() is None

    run_async(_drive())


def test_start_run_substrate_inline_on_in_memory_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With NO durable store requested the substrate stays INLINE (no dispatcher).

    The dispatcher's value is recovering runs across a restart; an in-memory store loses
    everything on exit, so the bootstrap must NOT start a dispatcher there — keeping the
    zero-config path byte-unchanged. The worker logs a warning in this case.
    """
    from himmy.api.runtime_bootstrap import (
        build_durable_container,
        start_run_substrate,
        stop_run_substrate,
    )
    from himmy.services.storage.factory import (
        reset_server_context,
        set_server_context,
    )

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)

    async def _drive() -> None:
        token = set_server_context(True)
        container, built = await build_durable_container()
        assert built is False
        substrate = await start_run_substrate(container, install_providers=True)
        try:
            assert substrate.backend == "memory"
            assert substrate.dispatch_on is False
            assert substrate.dispatcher is None
            assert container.run_app.dispatch_enabled is False
        finally:
            await stop_run_substrate(substrate, clear_providers=True)
            reset_server_context(token)

    run_async(_drive())


# ------------------------------------------------------------- worker command


def test_worker_starts_scheduler_and_dispatcher_then_shuts_down(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker async body starts the scheduler + dispatcher, then tears both down.

    We stub the blocking ``stop.wait()`` to fire immediately so the body runs its full
    startup → (instant) block → graceful shutdown path without a real signal. Asserts the
    scheduler was started (then stopped) and the dispatcher engaged.
    """
    import himmy.cli.commands as commands_mod
    from himmy.api.routines import get_scheduler, reset_scheduler

    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "on")
    reset_scheduler()

    states: dict[str, Any] = {}
    real_start = type(get_scheduler()).start
    real_stop = type(get_scheduler()).stop
    reset_scheduler()

    def _spy_start(self: Any) -> None:
        states["scheduler_started_active"] = True
        return real_start(self)

    async def _spy_stop(self: Any) -> None:
        states["scheduler_stopped"] = True
        return await real_stop(self)

    monkeypatch.setattr(type(get_scheduler()), "start", _spy_start)
    monkeypatch.setattr(type(get_scheduler()), "stop", _spy_stop)
    reset_scheduler()

    # Make the worker's block return immediately (no real SIGINT needed).
    import asyncio as _asyncio

    real_event = _asyncio.Event

    class _InstantEvent(real_event):  # type: ignore[misc, valid-type]
        async def wait(self) -> bool:  # noqa: D401 - test double
            return True

    monkeypatch.setattr(_asyncio, "Event", _InstantEvent)

    run_async(
        commands_mod._run_worker(run_scheduler=True, run_dispatcher=True)
    )

    assert states.get("scheduler_started_active") is True
    assert states.get("scheduler_stopped") is True
    # After the worker exits, the scheduler loop is stopped (not active).
    assert get_scheduler().active is False


def test_worker_scheduler_only_mode_stops_the_dispatcher(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--scheduler-only`` ticks routines but does NOT keep a run-queue dispatcher up."""
    import asyncio as _asyncio

    import himmy.cli.commands as commands_mod
    from himmy.application import dispatcher as dispatcher_mod

    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "on")

    started: dict[str, int] = {"start": 0, "stop": 0}
    real_start = dispatcher_mod.RunDispatcher.start
    real_stop = dispatcher_mod.RunDispatcher.stop

    def _spy_start(self: Any) -> None:
        started["start"] += 1
        return real_start(self)

    async def _spy_stop(self: Any, *, timeout: float = 30.0) -> None:
        started["stop"] += 1
        return await real_stop(self, timeout=timeout)

    monkeypatch.setattr(dispatcher_mod.RunDispatcher, "start", _spy_start)
    monkeypatch.setattr(dispatcher_mod.RunDispatcher, "stop", _spy_stop)

    class _InstantEvent(_asyncio.Event):
        async def wait(self) -> bool:
            return True

    monkeypatch.setattr(_asyncio, "Event", _InstantEvent)

    run_async(
        commands_mod._run_worker(run_scheduler=True, run_dispatcher=False)
    )

    # The bootstrap starts the dispatcher (durable store), then scheduler-only mode stops
    # it during startup; either way it must be stopped by the time the worker exits.
    assert started["stop"] >= 1


# --------------------------------------------------- leader / worker split


def test_worker_only_mode_never_starts_the_scheduler(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--no-scheduler`` (worker-only) drains the queue but NEVER ticks the scheduler.

    This is the leader/worker split's worker half: many such nodes drain the run-queue while
    exactly one leader ticks. Assert the scheduler's ``start`` is never called and it stays
    inactive — even though the dispatcher engages.
    """
    import asyncio as _asyncio

    import himmy.cli.commands as commands_mod
    from himmy.api.routines import get_scheduler, reset_scheduler

    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "on")
    reset_scheduler()

    started = {"n": 0}
    real_start = type(get_scheduler()).start

    def _spy_start(self: Any) -> None:
        started["n"] += 1
        return real_start(self)

    monkeypatch.setattr(type(get_scheduler()), "start", _spy_start)
    reset_scheduler()

    class _InstantEvent(_asyncio.Event):
        async def wait(self) -> bool:
            return True

    monkeypatch.setattr(_asyncio, "Event", _InstantEvent)

    run_async(
        commands_mod._run_worker(run_scheduler=False, run_dispatcher=True)
    )

    assert started["n"] == 0, "worker-only mode must never start the scheduler"
    assert get_scheduler().active is False


def test_worker_acquires_sqlite_host_leadership_and_starts_scheduler(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lone SQLite worker wins the host-flock topology guard and ticks the scheduler."""
    import asyncio as _asyncio

    import himmy.cli.commands as commands_mod
    from himmy.api.routines import get_scheduler, reset_scheduler

    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "on")
    reset_scheduler()

    started = {"n": 0}
    real_start = type(get_scheduler()).start

    def _spy_start(self: Any) -> None:
        started["n"] += 1
        return real_start(self)

    monkeypatch.setattr(type(get_scheduler()), "start", _spy_start)
    reset_scheduler()

    class _InstantEvent(_asyncio.Event):
        async def wait(self) -> bool:
            return True

    monkeypatch.setattr(_asyncio, "Event", _InstantEvent)

    run_async(commands_mod._run_worker(run_scheduler=True, run_dispatcher=True))

    assert started["n"] == 1, "the lone SQLite scheduler should have been started once"
    assert get_scheduler().active is False  # stopped on teardown


def test_worker_scheduler_disabled_by_edge_falsy_token_n(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HIMMY_ROUTINES_SCHEDULER=n`` disables the worker scheduler (canonical vocabulary).

    Regression for the two-readers/two-vocabularies bypass: the ad-hoc worker parse honored
    only ``off/0/false/no`` so ``=n`` left the scheduler ENABLED here while
    studio_routines.py's ``not env_falsy(...)`` (which recognises ``n``) DISABLED it. Now the
    worker routes through ``env_falsy`` too, so ``=n`` disables the scheduler in both — the
    leadership bid is never even attempted.
    """
    import asyncio as _asyncio

    import himmy.cli.commands as commands_mod
    from himmy.api.routines import get_scheduler, reset_scheduler

    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "n")
    reset_scheduler()

    started = {"n": 0}
    real_start = type(get_scheduler()).start

    def _spy_start(self: Any) -> None:
        started["n"] += 1
        return real_start(self)

    monkeypatch.setattr(type(get_scheduler()), "start", _spy_start)
    reset_scheduler()

    acquired = {"n": 0}
    import himmy.api.scheduler_leader as leader_mod

    async def _never_acquire(*_a: Any, **_k: Any) -> Any:  # pragma: no cover - must not run
        acquired["n"] += 1
        raise AssertionError("leadership must not be bid when scheduler is disabled")

    monkeypatch.setattr(leader_mod, "acquire_scheduler_leadership", _never_acquire)

    class _InstantEvent(_asyncio.Event):
        async def wait(self) -> bool:
            return True

    monkeypatch.setattr(_asyncio, "Event", _InstantEvent)

    run_async(commands_mod._run_worker(run_scheduler=True, run_dispatcher=True))

    assert started["n"] == 0, "=n must DISABLE the scheduler (canonical falsy vocabulary)"
    assert acquired["n"] == 0, "no leadership bid when the scheduler is disabled"


def test_worker_require_ack_honored_for_edge_truthy_token_y(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HIMMY_SCHEDULER_REQUIRE_ACK=y`` requires single-scheduler ack in the worker.

    Regression for the ad-hoc truthy vocabulary (``1/true/yes/on`` only) that silently
    dropped ``y`` — diverging from studio_routines.py's ``env_truthy`` which recognises it.
    We capture the ``require_single_scheduler_ack`` kwarg the worker passes to
    ``acquire_scheduler_leadership`` and assert it is ``True`` for ``=y``.
    """
    import asyncio as _asyncio

    import himmy.api.scheduler_leader as leader_mod
    import himmy.cli.commands as commands_mod
    from himmy.api.routines import reset_scheduler

    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "on")
    monkeypatch.setenv("HIMMY_SCHEDULER_REQUIRE_ACK", "y")
    reset_scheduler()

    captured: dict[str, Any] = {}

    class _Leadership:
        is_leader = False
        mode = "stub"
        reason = "stubbed for ack-vocabulary assertion"

        async def release(self) -> None:
            return None

    async def _spy_acquire(active: Any, *, require_single_scheduler_ack: bool = False) -> Any:
        captured["require_ack"] = require_single_scheduler_ack
        return _Leadership()

    monkeypatch.setattr(leader_mod, "acquire_scheduler_leadership", _spy_acquire)

    class _InstantEvent(_asyncio.Event):
        async def wait(self) -> bool:
            return True

    monkeypatch.setattr(_asyncio, "Event", _InstantEvent)

    run_async(commands_mod._run_worker(run_scheduler=True, run_dispatcher=True))

    assert captured.get("require_ack") is True, (
        "=y must require single-scheduler ack (canonical truthy vocabulary)"
    )


def test_sqlite_second_scheduler_refused_but_first_runs(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On SQLite the topology guard refuses a SECOND scheduler while the first holds the host flock."""

    from himmy.api.scheduler_leader import acquire_scheduler_leadership

    async def _drive() -> None:
        from himmy.api.runtime_bootstrap import (
            build_durable_container,
            start_run_substrate,
            stop_run_substrate,
        )
        from himmy.services.storage.factory import (
            reset_server_context,
            set_server_context,
        )

        token = set_server_context(True)
        container, _ = await build_durable_container()
        substrate = await start_run_substrate(container, install_providers=True)
        try:
            assert substrate.backend == "sqlite"
            lock_base = tmp_path / "locks-host"
            first = await acquire_scheduler_leadership(
                container, lock_base=lock_base
            )
            assert first.is_leader is True
            assert first.mode == "host-flock"
            # A second scheduler on this host cannot coordinate over SQLite -> refused.
            second = await acquire_scheduler_leadership(
                container, lock_base=lock_base
            )
            assert second.is_leader is False
            assert second.mode == "refused"
            await first.release()
            await second.release()
        finally:
            await stop_run_substrate(substrate, clear_providers=True)
            reset_server_context(token)

    run_async(_drive())


# --------------------------------------------------- run-now durable bridge


def test_run_now_records_durable_run_with_routine_actor(
    workspace: Path,
) -> None:
    """``himmy routines run-now`` on an agent_path routine records a DURABLE canonical run.

    The bridge contract: an OS-cron line calling run-now lands a run in the SAME durable
    store ``himmy runs`` / ``GET /v1/runs`` / Studio read, stamped with ``actor.routine_id``
    + ``source=routine`` so it is attributable to its routine. Uses the offline stub provider.
    """
    import argparse

    from himmy.cli.routines_cmd import _cmd_run_now
    from himmy.services.storage.models import LOCAL_WORKSPACE
    from himmy.services.storage.sqlite import SqliteStorageService

    # Create a local (agent_path) routine bound to the workspace's agent.yaml.
    routine = svc.Routine(
        name="nightly",
        agent_path="agent.yaml",
        prompt="say hello",
        schedule=svc.Schedule(kind="every", hours=6),
        workspace_id=LOCAL_WORKSPACE,
    )
    svc.get_routines_store().upsert(routine)

    args = argparse.Namespace(routine_id=routine.id, json=False)
    rc = _cmd_run_now(args)
    assert rc == 0, "run-now did not succeed"

    # The run is durable in the canonical store, carrying the routine actor + lineage source.
    store_path = workspace / ".himmy" / "storage.db"
    assert store_path.exists(), "run-now did not write the durable canonical store"
    store = SqliteStorageService(str(store_path))
    try:
        runs = run_async(store.list_runs(workspace_id=LOCAL_WORKSPACE))
    finally:
        run_async(store.close())
    assert runs, "no canonical run recorded by run-now"
    rec = runs[0]
    meta = rec.metadata or {}
    assert meta.get("source") == "routine"
    assert meta.get("routine_id") == routine.id
    assert (meta.get("actor") or {}).get("routine_id") == routine.id


# ----------------------------------------- worker tool-capability gate (P0)


def _drive_worker_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run the worker async body once (instant block, no real signal)."""
    import asyncio as _asyncio

    import himmy.cli.commands as commands_mod
    from himmy.api.routines import reset_scheduler

    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    reset_scheduler()

    class _InstantEvent(_asyncio.Event):
        async def wait(self) -> bool:
            return True

    monkeypatch.setattr(_asyncio, "Event", _InstantEvent)
    run_async(commands_mod._run_worker(run_scheduler=False, run_dispatcher=True))


def test_worker_wires_tool_authz_when_authenticator_configured(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION: the worker wires the tool-capability gate under a configured authenticator.

    The `himmy worker` node drains the SAME durable queue / ticks routines as the API
    server, but historically wired NEITHER half of the gate: ``mark_auth_configured`` was
    never called and ``run_app._access_policy`` was never set. So a least-privilege tenant's
    run/routine dispatched on a worker executed with the agent's FULL toolset (a cross-process
    confused-deputy escalation). Assert both halves engage when an authenticator is configured.
    """
    import json

    from himmy.api.runtime_bootstrap import build_durable_container, start_run_substrate
    from himmy.services.storage.factory import reset_server_context, set_server_context
    from himmy.services.tools.ambient import auth_is_configured, mark_auth_configured

    # A tenant-mapped keys file → build_authenticator() returns a tenant-binding authenticator
    # (passes the multi-tenant posture refusal AND engages the gate).
    keys = workspace / "keys.json"
    keys.write_text(
        json.dumps(
            {"tenant-key": {"subject": "u", "tenant_ids": ["acme"], "roles": ["operator"]}}
        )
    )
    monkeypatch.setenv("HIMMY_API_KEYS_FILE", str(keys))
    # Reset the process-wide sentinel so we observe the worker flip it (not a prior app).
    mark_auth_configured(False)

    _drive_worker_once(monkeypatch)

    # The sentinel is ON → the chokepoint fails CLOSED when a path reaches it un-scoped.
    assert auth_is_configured() is True

    # And the run_app carries the RBAC policy so it rebuilds a per-run capability gate from
    # each claimed run's actor metadata. Re-derive the substrate's run_app to inspect it.
    async def _check_policy() -> None:
        token = set_server_context(True)
        try:
            container, _ = await build_durable_container()
            from himmy.api.runtime_bootstrap import (
                stop_run_substrate,
                wire_tool_authz,
            )

            substrate = await start_run_substrate(container, install_providers=True)
            try:
                engaged = wire_tool_authz(getattr(substrate.active, "run_app", None))
                assert engaged is True
                run_app = substrate.active.run_app
                assert run_app._access_policy is not None
            finally:
                await stop_run_substrate(substrate, clear_providers=True)
        finally:
            reset_server_context(token)

    run_async(_check_policy())
    mark_auth_configured(False)  # restore the process default for later tests


def test_worker_leaves_gate_off_when_no_authenticator(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INVARIANT: with NO authenticator the worker leaves the gate OFF (offline byte-unchanged).

    The chokepoint stays a pure pass and ``run_app._access_policy`` is unset — exactly like
    the FastAPI body, so the zero-config worker path is unchanged.
    """
    from himmy.services.tools.ambient import auth_is_configured, mark_auth_configured

    monkeypatch.delenv("HIMMY_API_KEYS_FILE", raising=False)
    monkeypatch.delenv("HIMMY_AUTH_MODE", raising=False)
    monkeypatch.delenv("HIMMY_INTERNAL_API_KEY", raising=False)
    mark_auth_configured(True)  # poison it; the worker must flip it back OFF

    _drive_worker_once(monkeypatch)

    assert auth_is_configured() is False


def test_worker_refuses_multi_tenant_posture_without_tenant_binding(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REGRESSION: a multi-tenant worker that cannot bind tenants refuses to start.

    Parity with the FastAPI server's ``_enforce_multi_tenant_posture``: a shared-key-only
    (non-tenant-binding) authenticator under a declared multi-tenant posture would run every
    claimed run as an all-tenants admin — the worker must refuse, not start.
    """
    from himmy.core.errors import HimmyError

    monkeypatch.delenv("HIMMY_API_KEYS_FILE", raising=False)
    monkeypatch.setenv("HIMMY_MULTI_TENANT", "1")
    monkeypatch.setenv("HIMMY_INTERNAL_API_KEY", "shared-only-key")

    with pytest.raises(HimmyError):
        _drive_worker_once(monkeypatch)


def test_run_now_agent_id_without_durable_store_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent_id (/v1) routine cannot run-now without a durable store — refuse honestly.

    ``create_run`` needs a durable store to be meaningful; with none configured the bridge
    must refuse with a clear message rather than silently losing the run.
    """
    import argparse

    from himmy.cli.routines_cmd import _cmd_run_now
    from himmy.services.storage.models import LOCAL_WORKSPACE

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / "routines.db"))
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    svc.reset_routines_store()

    routine = svc.Routine(
        name="tenant-job",
        agent_id="agt_123",
        prompt="do the thing",
        schedule=svc.Schedule(kind="every", hours=6),
        workspace_id=LOCAL_WORKSPACE,
    )
    svc.get_routines_store().upsert(routine)

    args = argparse.Namespace(routine_id=routine.id, json=False)
    rc = _cmd_run_now(args)
    assert rc == 1, "agent_id run-now without a durable store must be refused"
    svc.reset_routines_store()
