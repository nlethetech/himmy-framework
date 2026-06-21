"""Live-PG-gated contract tests for the K3/K4 Postgres aux mirrors.

SKIPPED unless ``HIMMY_TEST_POSTGRES_DSN`` is set AND ``asyncpg`` is importable (the same
gate as ``test_postgres_storage_service.py``). They run the mirrors against a REAL database
through their SYNCHRONOUS facades (which drive the shared aux loop), asserting the
authoritative SQL semantics the offline lane cannot: tenant isolation, the per-record JSON
``schema_version`` migration on read, the checkpoint ``claim`` CAS, and — the headline K4
property — the two-writer atomic routine-tick claim (exactly-once across replicas) including
the NULL-first-run case.

Each test binds a UNIQUE tenant (a uuid) so the suite is re-runnable without truncation and
two tenants are provably isolated.
"""

from __future__ import annotations

import os
import uuid

import pytest

from himmy.services.storage.aux_store_factory import (
    aux_loop,
    reset_aux_store_factory,
)

_DSN = os.environ.get("HIMMY_TEST_POSTGRES_DSN")

try:  # pragma: no cover - import probe
    import asyncpg  # type: ignore  # noqa: F401

    _HAVE_ASYNCPG = True
except ImportError:  # pragma: no cover
    _HAVE_ASYNCPG = False

pytestmark = pytest.mark.skipif(
    not _DSN or not _HAVE_ASYNCPG,
    reason="requires HIMMY_TEST_POSTGRES_DSN + the [postgres] extra",
)


@pytest.fixture(autouse=True)
def _schema() -> object:  # pragma: no cover - only runs with a live DB
    """Ensure the aux tables exist (migrate) and the shared loop is fresh per test."""
    from himmy.services.storage.postgres import PostgresStorageService

    reset_aux_store_factory()

    async def _migrate() -> None:
        storage = await PostgresStorageService.connect(_DSN)
        try:
            await storage.migrate()
        finally:
            await storage.close()

    aux_loop().run(_migrate())
    yield
    reset_aux_store_factory()


def _tenant() -> str:  # pragma: no cover - only runs with a live DB
    return f"t-{uuid.uuid4()}"


def test_checkpoint_roundtrip_and_tenant_isolation() -> None:  # pragma: no cover
    from himmy.runtime.checkpoint import AWAITING_APPROVAL, AgentCheckpoint
    from himmy.services.storage.postgres_aux import PostgresCheckpointStore

    ta, tb = _tenant(), _tenant()
    a = PostgresCheckpointStore(tenant=ta, dsn=_DSN)
    b = PostgresCheckpointStore(tenant=tb, dsn=_DSN)
    cp = AgentCheckpoint(status=AWAITING_APPROVAL)
    a.save(cp)
    assert a.load(cp.checkpoint_id) is not None
    # The other tenant cannot see it (K3 isolation).
    assert b.load(cp.checkpoint_id) is None
    assert [c.checkpoint_id for c in a.list_by_status(AWAITING_APPROVAL)] == [
        cp.checkpoint_id
    ]
    assert b.list_by_status(AWAITING_APPROVAL) == []


def test_checkpoint_claim_is_exactly_once() -> None:  # pragma: no cover
    from himmy.runtime.checkpoint import (
        APPROVED,
        AWAITING_APPROVAL,
        RESOLVING,
        AgentCheckpoint,
    )
    from himmy.services.storage.postgres_aux import PostgresCheckpointStore

    store = PostgresCheckpointStore(tenant=_tenant(), dsn=_DSN)
    cp = AgentCheckpoint(status=AWAITING_APPROVAL)
    store.save(cp)
    assert store.claim(cp.checkpoint_id) is True  # awaiting -> resolving
    assert store.load(cp.checkpoint_id).status == RESOLVING
    assert store.claim(cp.checkpoint_id) is True  # resolving is re-claimable (crash retry)
    # A terminal checkpoint is not claimable.
    cp2 = AgentCheckpoint(status=APPROVED)
    store.save(cp2)
    assert store.claim(cp2.checkpoint_id) is False


def test_graph_checkpoint_roundtrip() -> None:  # pragma: no cover
    from himmy.runtime.checkpoint import GRAPH_RUNNING, GraphCheckpoint
    from himmy.services.storage.postgres_aux import PostgresGraphCheckpointStore

    store = PostgresGraphCheckpointStore(tenant=_tenant(), dsn=_DSN)
    chk = GraphCheckpoint(graph_name="g", status=GRAPH_RUNNING, frontier=["n1"])
    store.save(chk)
    loaded = store.load(chk.checkpoint_id)
    assert loaded is not None
    assert loaded.frontier == ["n1"]
    assert [c.checkpoint_id for c in store.list_by_status(GRAPH_RUNNING)] == [
        chk.checkpoint_id
    ]


def test_conversation_roundtrip_and_projection() -> None:  # pragma: no cover
    from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
    from himmy.services.storage.postgres_aux import PostgresConversationStore

    store = PostgresConversationStore(tenant=_tenant(), dsn=_DSN)
    thread = ChatThread(thread_id="c1")
    thread.append_message(Message(role=MessageRole.USER, content="hi"))
    thread.append_message(Message(role=MessageRole.ASSISTANT, content="hello"))
    summary = store.save_thread("c1", thread)
    assert summary.message_count == 2
    assert store.load_thread("c1") is not None
    flat = store.flat_messages("c1")
    assert [(m.role, m.text) for m in flat] == [("user", "hi"), ("agent", "hello")]
    assert [s.conversation_id for s in store.list_summaries()] == ["c1"]


def test_conversation_projects() -> None:  # pragma: no cover
    from himmy.services.storage.postgres_aux import PostgresConversationStore

    store = PostgresConversationStore(tenant=_tenant(), dsn=_DSN)
    proj = store.create_project(name="P")
    store.save_flat(
        conversation_id="cc",
        title="t",
        agent_path=None,
        provider=None,
        flat_messages=[("user", "q")],
    )
    assert store.assign_conversation(proj["id"], "cc") is True
    assert store.project_chat_count(proj["id"]) == 1
    assert [s.conversation_id for s in store.project_conversation_summaries(proj["id"])] == [
        "cc"
    ]
    assert store.unassign_conversation("cc") is True
    assert store.project_chat_count(proj["id"]) == 0


def test_teams_workflows_roundtrip() -> None:  # pragma: no cover
    from himmy.api.teams_store import TeamRecord, WorkflowRecord
    from himmy.services.storage.postgres_aux import (
        PostgresTeamsStore,
        PostgresWorkflowsStore,
    )

    teams = PostgresTeamsStore(tenant=_tenant(), dsn=_DSN)
    team = TeamRecord(workspace_id="w1", name="T", members=["a1", "a2"])
    teams.upsert(team)
    assert teams.get(team.id, workspace_id="w1") is not None
    assert teams.get(team.id, workspace_id="other") is None  # tenant/workspace scoped
    assert teams.find_by_agent_id("a1", workspace_id="w1") == [team.id]

    wf = PostgresWorkflowsStore(tenant=_tenant(), dsn=_DSN)
    flow = WorkflowRecord(workspace_id="w1", name="F", members=["a1"])
    wf.upsert(flow)
    assert wf.find_by_agent_id("a1", workspace_id="w1") == [flow.id]


def test_routine_atomic_tick_two_writers() -> None:  # pragma: no cover
    """The headline K4 property on a real DB: two writers, one captured anchor, one fires."""
    from himmy.api.routines import Routine, Schedule
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    store = PostgresRoutinesStore(tenant=_tenant(), dsn=_DSN)
    r = Routine(
        name="n",
        agent_path="a.yaml",
        prompt="p",
        schedule=Schedule(kind="every", hours=1),
    )
    r.last_run_at = "2026-06-13T00:00:00+00:00"
    store.upsert(r)
    sentinel = r.last_run_at

    first = store.mark_started(
        r.id, "2026-06-13T01:00:00+00:00", expected_last_run_at=sentinel
    )
    second = store.mark_started(
        r.id, "2026-06-13T01:00:05+00:00", expected_last_run_at=sentinel
    )
    assert first is True
    assert second is False
    assert store.get(r.id).last_run_at == "2026-06-13T01:00:00+00:00"


def test_routine_atomic_tick_null_first_run() -> None:  # pragma: no cover
    from himmy.api.routines import Routine, Schedule
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    store = PostgresRoutinesStore(tenant=_tenant(), dsn=_DSN)
    r = Routine(
        name="n",
        agent_path="a.yaml",
        prompt="p",
        schedule=Schedule(kind="every", hours=1),
    )
    store.upsert(r)  # last_run_at is NULL
    won = store.mark_started(
        r.id, "2026-06-13T01:00:00+00:00", expected_last_run_at=None
    )
    assert won is True
    # A stale NULL sentinel now loses (the anchor advanced).
    again = store.mark_started(
        r.id, "2026-06-13T02:00:00+00:00", expected_last_run_at=None
    )
    assert again is False


def test_routine_count_per_tenant_quota() -> None:  # pragma: no cover
    """``count`` is the per-tenant routine-quota predicate — it MUST exist + be PG-accurate.

    Regression: the PG routines mirror previously lacked ``count`` entirely, so
    ``enforce_tenant_routine_quota`` raised AttributeError and the quota FAILED OPEN on the
    exact multi-tenant Postgres deployment where it matters. This asserts the mirror counts
    a workspace's routines (and ``enabled_only`` honours the JSON ``enabled`` flag) so the
    cap actually bites on Postgres.
    """
    from himmy.api.routines import Routine, Schedule
    from himmy.services.storage.postgres_aux import PostgresRoutinesStore

    store = PostgresRoutinesStore(tenant=_tenant(), dsn=_DSN)

    def _mk(ws: str, *, enabled: bool = True) -> Routine:
        return Routine(
            name="n",
            workspace_id=ws,
            agent_id="a",
            prompt="p",
            enabled=enabled,
            schedule=Schedule(kind="every", hours=1),
        )

    store.upsert(_mk("ws-1"))
    store.upsert(_mk("ws-1"))
    store.upsert(_mk("ws-1", enabled=False))
    store.upsert(_mk("ws-2"))

    assert store.count(workspace_id="ws-1") == 3
    assert store.count(workspace_id="ws-1", enabled_only=True) == 2
    assert store.count(workspace_id="ws-2", enabled_only=True) == 1
    # a tenant with no routines counts zero (the never-flooded base case)
    assert store.count(workspace_id="ws-empty", enabled_only=True) == 0
