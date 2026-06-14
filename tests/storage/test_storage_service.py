"""Tests for the storage kernel: threads, events, runs, recommendations, idempotency."""

from __future__ import annotations

from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
from himmy.core.events import EventType, RunEvent
from himmy.services.storage import (
    ActionRecord,
    AgentStateRecord,
    ContextEvidenceRecord,
    EnvironmentStateRecord,
    EpisodicMemoryObject,
    MemoryObject,
    PostgresStorageService,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
    StorageService,
    ThreadEventStore,
)
from tests.conftest import run_async


def test_save_and_load_thread() -> None:
    """A thread round-trips by id; missing id returns None."""
    storage = StorageService()
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="hi"))
    run_async(storage.save_thread(thread))
    loaded = run_async(storage.load_thread(thread.thread_id))
    assert loaded is not None
    assert loaded.thread_id == thread.thread_id
    assert run_async(storage.load_thread("nope")) is None


def test_append_and_filter_events() -> None:
    """Events are appended and filterable by thread/trace id (EventSink role)."""
    storage = StorageService()
    e1 = RunEvent(
        event_type=EventType.AGENT_RUN_STARTED, thread_id="t1", trace_id="tr1"
    )
    e2 = RunEvent(
        event_type=EventType.AGENT_RUN_FINISHED, thread_id="t2", trace_id="tr2"
    )
    run_async(storage.append_event(e1))
    run_async(storage.append_event(e2))
    assert len(run_async(storage.list_events())) == 2
    assert len(run_async(storage.list_events(thread_id="t1"))) == 1
    assert run_async(storage.list_events(trace_id="tr2"))[0].event_id == e2.event_id


def test_storage_satisfies_thread_event_store_protocol() -> None:
    """StorageService is a structural ThreadEventStore."""
    assert isinstance(StorageService(), ThreadEventStore)


def test_run_crud_and_status_filter() -> None:
    """Runs save, fetch, and list filtered by workspace/subject/status."""
    storage = StorageService()
    run = RunRecord(workspace_id="w1", subject_id="s1", status=RunStatus.QUEUED)
    run_async(storage.save_run(run))
    assert run_async(storage.get_run(run.run_id)) is not None
    queued = run_async(storage.list_runs(workspace_id="w1", status=RunStatus.QUEUED))
    assert len(queued) == 1
    assert run_async(storage.list_runs(status=RunStatus.SUCCEEDED)) == []


def test_load_run_by_idempotency() -> None:
    """A run is resolvable by (workspace_id, idempotency_key)."""
    storage = StorageService()
    run = RunRecord(workspace_id="w1", subject_id="s1", idempotency_key="key-1")
    run_async(storage.save_run(run))
    found = run_async(storage.load_run_by_idempotency("w1", "key-1"))
    assert found is not None and found.run_id == run.run_id
    assert run_async(storage.load_run_by_idempotency("w1", "missing")) is None


def test_recommendation_crud_and_update() -> None:
    """Recommendations save, list, and transition status/notes."""
    storage = StorageService()
    item = RecommendationItem(
        run_id="r1",
        workspace_id="w1",
        subject_id="s1",
        kind="trade",
        title="Buy ACME",
    )
    run_async(storage.save_recommendation(item))
    listed = run_async(storage.list_recommendations(workspace_id="w1", kind="trade"))
    assert len(listed) == 1
    updated = run_async(
        storage.update_recommendation(
            item.recommendation_id,
            status=RecommendationStatus.ACCEPTED,
            notes="approved",
        )
    )
    assert updated is not None
    assert updated.status == RecommendationStatus.ACCEPTED
    assert updated.notes == "approved"
    assert (
        run_async(
            storage.update_recommendation(
                "missing", status=RecommendationStatus.DISMISSED
            )
        )
        is None
    )


def test_save_run_stamps_updated_at() -> None:
    """save_run always refreshes updated_at — storage owns the field (SE-11)."""
    storage = StorageService()
    run = RunRecord(workspace_id="w1", subject_id="s1", updated_at="STALE")
    run_async(storage.save_run(run))
    fetched = run_async(storage.get_run(run.run_id))
    assert fetched is not None
    assert fetched.updated_at != "STALE"


def test_save_run_if_absent_by_idempotency_is_atomic() -> None:
    """The atomic primitive creates once and returns the existing run after (SE-2/4)."""
    storage = StorageService()
    first = RunRecord(workspace_id="w1", subject_id="s1", idempotency_key="k1")
    stored1, created1 = run_async(storage.save_run_if_absent_by_idempotency(first))
    assert created1 is True and stored1.run_id == first.run_id

    # A second record with the same (workspace, key) returns the first, not a dup.
    second = RunRecord(workspace_id="w1", subject_id="s1", idempotency_key="k1")
    stored2, created2 = run_async(storage.save_run_if_absent_by_idempotency(second))
    assert created2 is False
    assert stored2.run_id == first.run_id
    assert len(run_async(storage.list_runs(workspace_id="w1"))) == 1

    # A null-key run is always created.
    third = RunRecord(workspace_id="w1", subject_id="s1")
    _, created3 = run_async(storage.save_run_if_absent_by_idempotency(third))
    assert created3 is True
    assert len(run_async(storage.list_runs(workspace_id="w1"))) == 2


def test_save_run_if_absent_under_concurrency() -> None:
    """Concurrent same-key creates collapse to one run (no TOCTOU dup) (SE-2)."""
    import asyncio

    async def scenario() -> int:
        storage = StorageService()
        runs = [
            RunRecord(workspace_id="w1", subject_id="s1", idempotency_key="dup")
            for _ in range(20)
        ]
        results = await asyncio.gather(
            *(storage.save_run_if_absent_by_idempotency(r) for r in runs)
        )
        created = sum(1 for _, was_created in results if was_created)
        run_ids = {stored.run_id for stored, _ in results}
        assert created == 1
        assert len(run_ids) == 1
        return len(await storage.list_runs(workspace_id="w1"))

    assert run_async(scenario()) == 1


def test_list_runs_multi_predicate() -> None:
    """list_runs honours workspace + subject + status simultaneously (SE-7)."""
    storage = StorageService()
    run_async(
        storage.save_run(
            RunRecord(workspace_id="w1", subject_id="s1", status=RunStatus.RUNNING)
        )
    )
    run_async(
        storage.save_run(
            RunRecord(workspace_id="w1", subject_id="s2", status=RunStatus.RUNNING)
        )
    )
    run_async(
        storage.save_run(
            RunRecord(workspace_id="w1", subject_id="s1", status=RunStatus.SUCCEEDED)
        )
    )
    got = run_async(
        storage.list_runs(workspace_id="w1", subject_id="s1", status=RunStatus.RUNNING)
    )
    assert len(got) == 1


def test_list_recommendations_multi_predicate() -> None:
    """list_recommendations honours all dimensions simultaneously (SE-7)."""
    storage = StorageService()
    run_async(
        storage.save_recommendation(
            RecommendationItem(
                run_id="r1",
                workspace_id="w1",
                subject_id="s1",
                kind="trade",
                title="A",
                status=RecommendationStatus.PROPOSED,
            )
        )
    )
    run_async(
        storage.save_recommendation(
            RecommendationItem(
                run_id="r1",
                workspace_id="w1",
                subject_id="s1",
                kind="alert",
                title="B",
                status=RecommendationStatus.PROPOSED,
            )
        )
    )
    got = run_async(
        storage.list_recommendations(
            workspace_id="w1",
            subject_id="s1",
            run_id="r1",
            kind="trade",
            status=RecommendationStatus.PROPOSED,
        )
    )
    assert len(got) == 1 and got[0].kind == "trade"


def test_evidence_memory_orchestration_crud() -> None:
    """Evidence + memory + orchestration records save/get/list round-trip (SE-7)."""
    storage = StorageService()

    run_async(storage.save_context_evidence(ContextEvidenceRecord(key="k")))
    assert len(storage._context_store._evidence) == 1  # type: ignore[attr-defined]

    mem = MemoryObject(subject_id="s1", payload={"note": "x"})
    run_async(storage.save_memory(mem))
    assert run_async(storage.get_memory(mem.memory_id)) is not None
    assert len(run_async(storage.list_memory(subject_id="s1"))) == 1

    epi = EpisodicMemoryObject(subject_id="s1")
    run_async(storage.save_episodic_memory(epi))
    assert len(run_async(storage.list_episodic_memory(subject_id="s1"))) == 1

    st = AgentStateRecord(environment_name="env", round=1)
    run_async(storage.save_agent_state(st))
    assert (
        len(run_async(storage.list_agent_states(environment_name="env", round=1))) == 1
    )

    act = ActionRecord(environment_name="env", round=2)
    run_async(storage.save_action(act))
    assert len(run_async(storage.list_actions(environment_name="env", round=2))) == 1

    es = EnvironmentStateRecord(environment_name="env", round=3)
    run_async(storage.save_environment_state(es))
    assert (
        len(run_async(storage.list_environment_states(environment_name="env", round=3)))
        == 1
    )


def test_postgres_scaffold_imports_and_exposes_ddl() -> None:
    """The Postgres scaffold imports offline and ships an inspectable DDL string.

    NOTE: this is an import/offline smoke check only. The real SQLite<->Postgres schema
    *parity* guard (which fails when a table/object is added to one chain but not the
    other) lives in ``tests/storage/test_schema_parity.py`` (K2) — these string-contents
    assertions deliberately stay as a cheap offline tripwire that the DDL is well-formed.
    """
    from himmy.services.storage.postgres import STORAGE_DDL, STORAGE_MIGRATIONS

    assert "CREATE TABLE IF NOT EXISTS" in STORAGE_DDL
    assert "ai_call_log" in STORAGE_DDL
    assert "schema_migrations" in STORAGE_DDL
    assert "TIMESTAMPTZ" in STORAGE_DDL
    # The migration runner ships an ordered, versioned migration list.
    assert STORAGE_MIGRATIONS and STORAGE_MIGRATIONS[0][0] == 1
    # The class is importable without asyncpg / a live DB and supports close().
    assert PostgresStorageService is not None
    assert hasattr(PostgresStorageService, "close")
    assert hasattr(PostgresStorageService, "save_run_if_absent_by_idempotency")
    assert hasattr(PostgresStorageService, "migrate")


def test_postgres_data_methods_require_pool() -> None:
    """Without a live pool, data methods raise a clear HimmyError (offline)."""
    from himmy.core.errors import HimmyError

    svc = PostgresStorageService(pool=None)
    for coro in (
        svc.save_run(RunRecord(workspace_id="w", subject_id="s")),
        svc.get_run("x"),
        svc.list_runs(),
        svc.load_run_by_idempotency("w", "k"),
    ):
        try:
            run_async(coro)
        except HimmyError:
            continue
        raise AssertionError("expected HimmyError without a live pool")
