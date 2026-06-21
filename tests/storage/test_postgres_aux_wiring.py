"""K3 + K4 (offline): the Postgres aux mirrors' SELECTION + sync-facade plumbing.

No live Postgres runs in the unit lane, so these tests exercise the parts of the K3/K4
mirrors that DO NOT need a real database:

* **Routing.** Under a ``postgres://`` ``HIMMY_DATABASE_URL`` every aux ``get_*_store``
  resolves the Postgres mirror class (the K3/K4 builders wired into ``select_aux_store``),
  and with no DSN it stays on the durable-SQLite / in-memory path byte-for-byte.
* **Tenant discriminator (K3 must_fix).** Studio's checkpoint store binds the ``"studio"``
  tenant and /v1's binds ``"v1"`` — the two HITL inboxes can never collide. Every SQL the
  mirrors emit is scoped on ``tenant``.
* **Sync-facade plumbing.** Each mirror drives its async body on the shared aux loop
  (:func:`aux_loop`) and reuses the published pool (:func:`aux_pool`); a recording fake pool
  proves the right tenant-scoped SQL is issued and the command-tag rowcount is parsed.

The authoritative SQL-semantics coverage (``jsonb_set`` / ``IS NOT DISTINCT FROM`` / the
two-writer atomic tick) lives in the live-PG-gated ``test_postgres_aux_live.py`` and — for
the dialect-shared conditional-claim logic — in ``test_routine_atomic_tick.py``.
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


class _FakeTxn:
    async def __aenter__(self) -> _FakeTxn:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeConn:
    """Records every SQL it is handed; returns canned rows / command tags by script."""

    def __init__(self, script: dict[str, Any]) -> None:
        self._script = script
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _FakeTxn:
        return _FakeTxn()

    async def execute(self, sql: str, *args: Any) -> str:
        self.calls.append((sql, args))
        return self._script.get("execute", "UPDATE 1")

    async def fetchrow(self, sql: str, *args: Any) -> Any:
        self.calls.append((sql, args))
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
    """Make every aux store resolve ``pool`` as its (loop-bound) pool, for isolated wiring.

    The aux stores no longer reuse the main-loop server pool (a loop-affinity bug — an
    asyncpg pool is bound to its creating loop, so the aux loop could not use it). They open
    their OWN pool on the aux loop instead. These wiring tests assert the emitted SQL, not the
    pool plumbing, so we short-circuit ``_AuxPgPool._resolve_async`` to hand back the recording
    fake — independent of any DSN / live DB.
    """
    from himmy.services.storage.postgres_aux import _AuxPgPool

    async def _resolve(self: _AuxPgPool) -> Any:
        return pool

    monkeypatch.setattr(_AuxPgPool, "_resolve_async", _resolve, raising=True)


def test_owned_pool_resolves_inline_without_nested_loop_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The owned-pool path opens its pool by AWAITING inline — never a nested loop.run().

    Regression for the K3/K4 owned-pool deadlock: an aux store with a DSN but no
    server-published pool (tests, or any non-server embedding) resolves its pool from inside
    an async body already running ON the shared aux loop. The OLD ``_resolve`` did a nested
    ``self._loop.run(...)``, parking the loop's only thread in ``Future.result()`` while
    submitting a coroutine to that same loop — a hard deadlock. ``acquire()`` must instead
    defer to ``_resolve_async`` and ``await`` pool creation. A loop stand-in whose ``run``
    explodes proves resolution never blocks on it (offline: no asyncpg, no live DB).
    """
    import asyncio

    import himmy.services.storage.aux_store_factory as factory
    from himmy.services.storage.postgres_aux import _AuxPgPool

    monkeypatch.setattr(factory, "aux_pool", lambda: None)  # force the owned-pool branch

    class _ExplodingLoop:
        def run(self, coro: Any) -> Any:  # pragma: no cover - asserted unreached
            coro.close()
            raise AssertionError(
                "owned-pool resolution must await inline, not nest a _LoopThread.run"
            )

    pool = _AuxPgPool(_ExplodingLoop(), dsn="postgresql://u@h/db")
    fake = _FakePool()

    async def _fake_create(_dsn: str) -> Any:
        return fake

    monkeypatch.setattr(pool, "_create_owned_pool", _fake_create)

    async def _drive() -> Any:
        async with pool.acquire() as conn:
            return conn

    conn = asyncio.run(_drive())
    assert conn is fake.conn
    assert pool._owned_pool is fake  # created once, memoised


# --------------------------------------------------------------------- routing wiring


def test_get_checkpoint_store_routes_to_pg_with_studio_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://u@h/db")
    from himmy.api import studio_approvals
    from himmy.services.storage.postgres_aux import PostgresCheckpointStore

    studio_approvals.reset_checkpoint_store()
    store = studio_approvals.get_checkpoint_store()
    try:
        assert isinstance(store, PostgresCheckpointStore)
        assert store._tenant == "studio"
    finally:
        studio_approvals.reset_checkpoint_store()


def test_v1_checkpoint_store_routes_to_pg_with_v1_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://u@h/db")
    from himmy.api.deps import ApiContainer
    from himmy.services.storage.factory import (
        reset_server_context,
        set_server_context,
    )
    from himmy.services.storage.postgres_aux import PostgresCheckpointStore

    token = set_server_context(True)
    try:
        store = ApiContainer._build_v1_checkpoint_store()
        assert isinstance(store, PostgresCheckpointStore)
        assert store._tenant == "v1"
    finally:
        reset_server_context(token)


def test_studio_and_v1_checkpoint_tenants_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The K3 must_fix: Studio and /v1 bind DIFFERENT tenants (no cross-surface read)."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://u@h/db")
    from himmy.api import studio_approvals
    from himmy.api.deps import ApiContainer
    from himmy.services.storage.factory import (
        reset_server_context,
        set_server_context,
    )

    token = set_server_context(True)
    studio_approvals.reset_checkpoint_store()
    try:
        studio = studio_approvals.get_checkpoint_store()
        v1 = ApiContainer._build_v1_checkpoint_store()
        assert studio._tenant != v1._tenant
        assert {studio._tenant, v1._tenant} == {"studio", "v1"}
    finally:
        studio_approvals.reset_checkpoint_store()
        reset_server_context(token)


def test_graph_checkpoint_store_routes_to_pg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://u@h/db")
    from himmy.api import teams_store
    from himmy.services.storage.factory import (
        reset_server_context,
        set_server_context,
    )
    from himmy.services.storage.postgres_aux import PostgresGraphCheckpointStore

    token = set_server_context(True)
    teams_store.reset_graph_checkpoint_store()
    try:
        store = teams_store.get_graph_checkpoint_store()
        assert isinstance(store, PostgresGraphCheckpointStore)
        assert store._tenant == "local"
    finally:
        teams_store.reset_graph_checkpoint_store()
        reset_server_context(token)


def test_teams_workflows_routines_conversations_route_to_pg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://u@h/db")
    from himmy.api import routines as routines_mod
    from himmy.api import teams_store
    from himmy.services.storage import conversations as conv_mod
    from himmy.services.storage.postgres_aux import (
        PostgresConversationStore,
        PostgresRoutinesStore,
        PostgresTeamsStore,
        PostgresWorkflowsStore,
    )

    teams_store.reset_teams_store()
    teams_store.reset_workflows_store()
    routines_mod.reset_routines_store()
    conv_mod.reset_conversation_store()
    try:
        assert isinstance(teams_store.get_teams_store(), PostgresTeamsStore)
        assert isinstance(teams_store.get_workflows_store(), PostgresWorkflowsStore)
        assert isinstance(routines_mod.get_routines_store(), PostgresRoutinesStore)
        assert isinstance(
            conv_mod.get_conversation_store(), PostgresConversationStore
        )
    finally:
        teams_store.reset_teams_store()
        teams_store.reset_workflows_store()
        routines_mod.reset_routines_store()
        conv_mod.reset_conversation_store()


def test_offline_path_stays_sqlite(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DSN → every aux store stays on the SQLite/in-memory path (byte-for-byte)."""
    from himmy.api import routines as routines_mod
    from himmy.api import studio_approvals, teams_store
    from himmy.api.routines import RoutinesStore
    from himmy.api.teams_store import TeamsStore, WorkflowsStore
    from himmy.runtime.checkpoint import SqliteCheckpointStore
    from himmy.services.storage import conversations as conv_mod
    from himmy.services.storage.conversations import ConversationStore

    studio_approvals.reset_checkpoint_store()
    teams_store.reset_teams_store()
    teams_store.reset_workflows_store()
    routines_mod.reset_routines_store()
    conv_mod.reset_conversation_store()
    try:
        assert isinstance(studio_approvals.get_checkpoint_store(), SqliteCheckpointStore)
        assert isinstance(teams_store.get_teams_store(), TeamsStore)
        assert isinstance(teams_store.get_workflows_store(), WorkflowsStore)
        assert isinstance(routines_mod.get_routines_store(), RoutinesStore)
        assert isinstance(conv_mod.get_conversation_store(), ConversationStore)
    finally:
        studio_approvals.reset_checkpoint_store()
        teams_store.reset_teams_store()
        teams_store.reset_workflows_store()
        routines_mod.reset_routines_store()
        conv_mod.reset_conversation_store()


# --------------------------------------------------------- sync facade + tenant scoping


def test_checkpoint_save_load_emits_tenant_scoped_sql(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The sync facade drives the loop + every SQL carries the bound tenant (K3)."""
    from himmy.runtime.checkpoint import AgentCheckpoint
    from himmy.services.storage.postgres_aux import PostgresCheckpointStore

    cp = AgentCheckpoint()
    pool = _FakePool({"fetchrow": {"data": cp.model_dump(mode="json")}})
    _install_pool(monkeypatch, pool)

    store = PostgresCheckpointStore(tenant="studio")
    store.save(cp)
    loaded = store.load(cp.checkpoint_id)

    assert loaded is not None
    assert loaded.checkpoint_id == cp.checkpoint_id
    # Both the INSERT and the SELECT bound "studio" as the tenant (first positional arg).
    for sql, args in pool.conn.calls:
        assert "aux_agent_checkpoints" in sql
        assert args[0] == "studio"


def test_checkpoint_claim_parses_rowcount(monkeypatch: pytest.MonkeyPatch) -> None:
    """``claim`` returns True iff the conditional UPDATE's command tag is ``UPDATE 1``."""
    from himmy.services.storage.postgres_aux import PostgresCheckpointStore

    won_pool = _FakePool({"execute": "UPDATE 1"})
    _install_pool(monkeypatch, won_pool)
    assert PostgresCheckpointStore(tenant="v1").claim("cp-1") is True

    lost_pool = _FakePool({"execute": "UPDATE 0"})
    _install_pool(monkeypatch, lost_pool)
    assert PostgresCheckpointStore(tenant="v1").claim("cp-1") is False


def test_routine_mark_started_unconditional_for_run_now(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``run_now`` (UNCONDITIONAL) issues the unconditional SELECT … FOR UPDATE claim."""
    from himmy.api.routines import UNCONDITIONAL, Routine, Schedule
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    r = Routine(
        name="n",
        agent_path="a.yaml",
        prompt="p",
        schedule=Schedule(kind="every", hours=1),
    )
    pool = _FakePool({"fetchrow": {"data": r.model_dump(mode="json")}})
    _install_pool(monkeypatch, pool)

    store = PostgresRoutinesStore(tenant="local")
    claimed = store.mark_started(
        r.id, "2026-06-13T00:00:00+00:00", expected_last_run_at=UNCONDITIONAL
    )
    assert claimed is True
    # The unconditional path's SELECT must NOT carry the IS NOT DISTINCT FROM predicate.
    select_sqls = [s for s, _ in pool.conn.calls if "SELECT" in s and "FOR UPDATE" in s]
    assert select_sqls
    assert all("IS NOT DISTINCT FROM" not in s for s in select_sqls)


def test_routine_mark_started_conditional_for_tick(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tick path threads a captured sentinel → the conditional claim predicate."""
    from himmy.api.routines import Routine, Schedule
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    r = Routine(
        name="n",
        agent_path="a.yaml",
        prompt="p",
        schedule=Schedule(kind="every", hours=1),
    )
    pool = _FakePool({"fetchrow": {"data": r.model_dump(mode="json")}})
    _install_pool(monkeypatch, pool)

    store = PostgresRoutinesStore(tenant="local")
    store.mark_started(r.id, "2026-06-13T00:00:00+00:00", expected_last_run_at=None)
    select_sqls = [s for s, _ in pool.conn.calls if "SELECT" in s and "FOR UPDATE" in s]
    assert select_sqls
    assert any("IS NOT DISTINCT FROM" in s for s in select_sqls)
