"""Tests for the durable, file-backed :class:`SqliteStorageService`.

Every per-concern method round-trips, two instances on the same file see each other's
writes (durability), and upserts/idempotent creates behave exactly like the in-memory
``StorageService``. All artifacts live under ``tmp_path`` — nothing touches the repo.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
from himmy.core.events import EventType, RunEvent
from himmy.services.context.models import ContextField, ContextSnapshot, EvidenceRef
from himmy.services.evaluation.models import EvaluationRun
from himmy.services.storage import (
    ActionRecord,
    AgentStateRecord,
    ContextEvidenceRecord,
    EnvironmentStateRecord,
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
    ThreadEventStore,
)
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _store(tmp_path: Path) -> SqliteStorageService:
    """A fresh SQLite store backed by a file under ``tmp_path``."""
    return SqliteStorageService(str(tmp_path / "storage.db"))


def test_satisfies_thread_event_store_protocol(tmp_path: Path) -> None:
    """The SQLite backend structurally satisfies the runtime-facing protocol."""
    assert isinstance(_store(tmp_path), ThreadEventStore)


def test_auto_creates_parent_directory(tmp_path: Path) -> None:
    """A nested store path has its parent dir created on first construction."""
    path = tmp_path / "nested" / "deeper" / "storage.db"
    SqliteStorageService(str(path))
    assert path.parent.is_dir()


def test_thread_round_trip(tmp_path: Path) -> None:
    """A thread round-trips by id; a missing id returns None."""
    storage = _store(tmp_path)
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="hi"))
    run_async(storage.save_thread(thread))
    loaded = run_async(storage.load_thread(thread.thread_id))
    assert loaded is not None
    assert loaded.thread_id == thread.thread_id
    assert loaded.messages[0].content == "hi"
    assert run_async(storage.load_thread("nope")) is None


def test_event_append_filter_and_idempotency(tmp_path: Path) -> None:
    """Events append, filter by thread/trace id, preserve order, and dedupe by id."""
    storage = _store(tmp_path)
    e1 = RunEvent(event_type=EventType.AGENT_RUN_STARTED, thread_id="t1", trace_id="x")
    e2 = RunEvent(event_type=EventType.AGENT_RUN_FINISHED, thread_id="t1", trace_id="y")
    e3 = RunEvent(event_type=EventType.AGENT_RUN_STARTED, thread_id="t2", trace_id="x")
    for e in (e1, e2, e3):
        run_async(storage.append_event(e))
    # Re-appending an existing event is a no-op (idempotent on event_id).
    run_async(storage.append_event(e1))

    by_thread = run_async(storage.list_events(thread_id="t1"))
    assert [e.event_id for e in by_thread] == [e1.event_id, e2.event_id]
    by_trace = run_async(storage.list_events(trace_id="x"))
    assert {e.event_id for e in by_trace} == {e1.event_id, e3.event_id}
    assert len(run_async(storage.list_events())) == 3


def test_context_field_round_trip(tmp_path: Path) -> None:
    """Context fields upsert by (subject_id, key) and list by subject."""
    storage = _store(tmp_path)
    field = ContextField(key="ticker", value="ACME", metadata={"subject_id": "s1"})
    run_async(storage.save_context_field(field))
    got = run_async(storage.get_context_field("s1", "ticker"))
    assert got is not None and got.value == "ACME"
    # Upsert replaces in place (no duplicate row).
    field.value = "BETA"
    run_async(storage.save_context_field(field))
    fields = run_async(storage.list_context_fields("s1"))
    assert len(fields) == 1 and fields[0].value == "BETA"


def test_snapshot_and_evidence_round_trip(tmp_path: Path) -> None:
    """Snapshots round-trip by id; evidence records persist (idempotent on id)."""
    storage = _store(tmp_path)
    snap = ContextSnapshot(subject_id="s1")
    snap.fields["k"] = ContextField(key="k", value=1)
    run_async(storage.save_snapshot(snap))
    loaded = run_async(storage.load_snapshot(snap.snapshot_id))
    assert loaded is not None and loaded.subject_id == "s1"
    assert "k" in loaded.fields

    ev = ContextEvidenceRecord(subject_id="s1", snapshot_id=snap.snapshot_id, key="k")
    run_async(storage.save_context_evidence(ev))
    # An EvidenceRef-like object also persists without a model failure.
    ref = EvidenceRef(source_type="tool", source_id="t1")
    run_async(storage.save_context_evidence(ref))


def test_run_round_trip_and_listing(tmp_path: Path) -> None:
    """Runs round-trip; listing filters by workspace/subject/status."""
    storage = _store(tmp_path)
    run = RunRecord(workspace_id="w1", subject_id="s1", status=RunStatus.QUEUED)
    saved = run_async(storage.save_run(run))
    assert saved.updated_at  # storage stamps updated_at
    got = run_async(storage.get_run(run.run_id))
    assert got is not None and got.workspace_id == "w1"

    run2 = RunRecord(workspace_id="w1", subject_id="s2", status=RunStatus.SUCCEEDED)
    run_async(storage.save_run(run2))
    assert len(run_async(storage.list_runs(workspace_id="w1"))) == 2
    assert len(run_async(storage.list_runs(subject_id="s2"))) == 1
    succeeded = run_async(storage.list_runs(status=RunStatus.SUCCEEDED))
    assert [r.run_id for r in succeeded] == [run2.run_id]


def test_run_idempotent_create(tmp_path: Path) -> None:
    """``save_run_if_absent_by_idempotency`` creates once, then returns the existing run."""
    storage = _store(tmp_path)
    r1 = RunRecord(workspace_id="w1", subject_id="s1", idempotency_key="K")
    out1, created1 = run_async(storage.save_run_if_absent_by_idempotency(r1))
    assert created1 is True and out1.run_id == r1.run_id

    r2 = RunRecord(workspace_id="w1", subject_id="s1", idempotency_key="K")
    out2, created2 = run_async(storage.save_run_if_absent_by_idempotency(r2))
    assert created2 is False
    assert out2.run_id == r1.run_id  # the first run wins
    found = run_async(storage.load_run_by_idempotency("w1", "K"))
    assert found is not None and found.run_id == r1.run_id


def test_claim_run_for_resume_cas(tmp_path: Path) -> None:
    """The run-level resume CAS flips AWAITING_APPROVAL -> RESOLVING exactly once."""
    storage = _store(tmp_path)
    run = RunRecord(
        workspace_id="w1", subject_id="s1", status=RunStatus.AWAITING_APPROVAL
    )
    run_async(storage.save_run(run))

    # First claim wins; the persisted status (column AND payload-derived read) is RESOLVING.
    assert run_async(storage.claim_run_for_resume(run.run_id, workspace_id="w1")) is True
    got = run_async(storage.get_run(run.run_id))
    assert got is not None and got.status is RunStatus.RESOLVING

    # A second claim of the now-RESOLVING run loses (only AWAITING_APPROVAL is claimable).
    assert (
        run_async(storage.claim_run_for_resume(run.run_id, workspace_id="w1")) is False
    )


def test_claim_run_for_resume_is_workspace_scoped(tmp_path: Path) -> None:
    """A claim from the wrong workspace cannot flip the run (tenant isolation)."""
    storage = _store(tmp_path)
    run = RunRecord(
        workspace_id="w1", subject_id="s1", status=RunStatus.AWAITING_APPROVAL
    )
    run_async(storage.save_run(run))

    assert (
        run_async(storage.claim_run_for_resume(run.run_id, workspace_id="other"))
        is False
    )
    still = run_async(storage.get_run(run.run_id))
    assert still is not None and still.status is RunStatus.AWAITING_APPROVAL


def test_claim_run_for_resume_concurrent_one_winner(tmp_path: Path) -> None:
    """asyncio.gather of N concurrent claims yields exactly one winner (durable CAS)."""
    import asyncio

    async def scenario() -> None:
        storage = _store(tmp_path)
        run = RunRecord(
            workspace_id="w1", subject_id="s1", status=RunStatus.AWAITING_APPROVAL
        )
        await storage.save_run(run)
        results = await asyncio.gather(
            *(storage.claim_run_for_resume(run.run_id, workspace_id="w1") for _ in range(20))
        )
        assert sum(1 for won in results if won) == 1

    run_async(scenario())


def test_recommendation_round_trip_and_update(tmp_path: Path) -> None:
    """Recommendations round-trip, list with filters, and update status/notes in place."""
    storage = _store(tmp_path)
    item = RecommendationItem(
        run_id="r1", workspace_id="w1", subject_id="s1", kind="buy", title="ACME"
    )
    run_async(storage.save_recommendation(item))
    got = run_async(storage.get_recommendation(item.recommendation_id))
    assert got is not None and got.status is RecommendationStatus.PROPOSED

    updated = run_async(
        storage.update_recommendation(
            item.recommendation_id,
            status=RecommendationStatus.ACCEPTED,
            notes="ok",
        )
    )
    assert updated is not None
    assert updated.status is RecommendationStatus.ACCEPTED
    assert updated.notes == "ok"
    # The change is persisted, not just returned.
    reread = run_async(storage.get_recommendation(item.recommendation_id))
    assert reread is not None and reread.status is RecommendationStatus.ACCEPTED
    listed = run_async(storage.list_recommendations(workspace_id="w1", kind="buy"))
    assert [r.recommendation_id for r in listed] == [item.recommendation_id]
    assert run_async(storage.update_recommendation("missing")) is None


def test_evaluation_run_round_trip(tmp_path: Path) -> None:
    """Evaluation runs round-trip and list by suite id."""
    storage = _store(tmp_path)
    er = EvaluationRun(suite_id="suite1", suite_name="Smoke", aggregate_score=0.9)
    run_async(storage.save_evaluation_run(er))
    got = run_async(storage.get_evaluation_run(er.run_id))
    assert got is not None and got.aggregate_score == 0.9
    assert len(run_async(storage.list_evaluation_runs("suite1"))) == 1
    assert run_async(storage.list_evaluation_runs("other")) == []


def test_orchestration_records_round_trip(tmp_path: Path) -> None:
    """Memory, episodic, agent-state, action, and environment-state records all round-trip."""
    storage = _store(tmp_path)
    mem = MemoryObject(subject_id="s1", payload={"note": "x"})
    run_async(storage.save_memory(mem))
    assert run_async(storage.get_memory(mem.memory_id)) is not None
    assert len(run_async(storage.list_memory("s1"))) == 1

    epi = EpisodicMemoryObject(subject_id="s1", payload={"e": 1})
    run_async(storage.save_episodic_memory(epi))
    assert run_async(storage.get_episodic_memory(epi.episode_id)) is not None
    assert len(run_async(storage.list_episodic_memory("s1"))) == 1

    st = AgentStateRecord(environment_name="env", round=1, payload={"a": 1})
    run_async(storage.save_agent_state(st))
    assert run_async(storage.get_agent_state(st.state_id)) is not None
    assert len(run_async(storage.list_agent_states("env", 1))) == 1

    act = ActionRecord(environment_name="env", round=1, payload={"move": "up"})
    run_async(storage.save_action(act))
    assert run_async(storage.get_action(act.action_id)) is not None
    assert len(run_async(storage.list_actions("env", 1))) == 1

    es = EnvironmentStateRecord(environment_name="env", round=2, payload={"w": 1})
    run_async(storage.save_environment_state(es))
    assert run_async(storage.get_environment_state(es.environment_state_id)) is not None
    assert len(run_async(storage.list_environment_states("env", 2))) == 1


def test_cross_instance_durability(tmp_path: Path) -> None:
    """A second instance on the same file sees the first instance's committed writes."""
    path = str(tmp_path / "shared.db")
    writer = SqliteStorageService(path)
    run = RunRecord(workspace_id="w1", subject_id="s1", status=RunStatus.RUNNING)
    run_async(writer.save_run(run))
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="persist me"))
    run_async(writer.save_thread(thread))
    run_async(writer.close())

    reader = SqliteStorageService(path)
    got = run_async(reader.get_run(run.run_id))
    assert got is not None and got.status is RunStatus.RUNNING
    loaded = run_async(reader.load_thread(thread.thread_id))
    assert loaded is not None and loaded.messages[0].content == "persist me"
    run_async(reader.close())


def test_upsert_is_idempotent_no_duplicate_rows(tmp_path: Path) -> None:
    """Saving the same record twice updates in place rather than duplicating."""
    storage = _store(tmp_path)
    run = RunRecord(workspace_id="w1", subject_id="s1", status=RunStatus.QUEUED)
    run_async(storage.save_run(run))
    run.status = RunStatus.SUCCEEDED
    run_async(storage.save_run(run))
    rows = run_async(storage.list_runs(workspace_id="w1"))
    assert len(rows) == 1 and rows[0].status is RunStatus.SUCCEEDED


def test_path_property_and_memory_backend() -> None:
    """The ``path`` property reports the backing file (or ``:memory:``)."""
    storage = SqliteStorageService(":memory:")
    assert storage.path == ":memory:"
    # An in-memory SQLite store still round-trips within its lifetime.
    mem: Any = MemoryObject(subject_id="s", payload={"k": 1})
    run_async(storage.save_memory(mem))
    assert run_async(storage.get_memory(mem.memory_id)) is not None


# ---------------------------------------------------------- transactional hygiene
class _FlakyCommitConnection:
    """Delegating connection wrapper whose next ``commit`` raises (failure injection)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self.fail_next_commit = False

    def commit(self) -> None:
        if self.fail_next_commit:
            self.fail_next_commit = False
            raise sqlite3.OperationalError("disk I/O error (injected)")
        self._conn.commit()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


def test_failed_commit_rolls_back_and_releases_write_lock(tmp_path: Path) -> None:
    """A mid-write failure leaves no open transaction and no torn pending write.

    Without rollback-on-failure the shared connection would stay inside the open
    transaction: the WAL write lock would block other connections ('database is
    locked') and the next caller's commit would silently fold the torn write in.
    """
    storage = _store(tmp_path)
    flaky = _FlakyCommitConnection(storage._conn)
    storage._conn = flaky  # type: ignore[assignment]
    flaky.fail_next_commit = True
    with pytest.raises(sqlite3.OperationalError):
        run_async(storage.save_memory(MemoryObject(subject_id="s", payload={"k": 1})))

    # The failed write was rolled back: no open-transaction residue...
    assert flaky.in_transaction is False
    # ...so a second connection to the same file can write immediately.
    other = sqlite3.connect(str(tmp_path / "storage.db"), timeout=0.2)
    try:
        other.execute("PRAGMA busy_timeout=200")
        other.execute(
            "INSERT INTO chat_threads (thread_id, payload, updated_at) "
            "VALUES ('t-probe', '{}', 'now')"
        )
        other.commit()
    finally:
        other.close()
    # The torn write is gone and is NOT resurrected by the next caller's commit.
    assert run_async(storage.list_memory()) == []
    ok = MemoryObject(subject_id="s", payload={"k": 2})
    run_async(storage.save_memory(ok))
    assert [m.memory_id for m in run_async(storage.list_memory())] == [ok.memory_id]


def test_constraint_violation_leaves_no_open_transaction(tmp_path: Path) -> None:
    """An insert aborted by ``runs_idempotency_idx`` rolls back; later writes work."""
    storage = _store(tmp_path)
    run_async(
        storage.save_run(
            RunRecord(workspace_id="w", subject_id="s", idempotency_key="k")
        )
    )
    dupe = RunRecord(workspace_id="w", subject_id="s", idempotency_key="k")
    with pytest.raises(sqlite3.IntegrityError):
        run_async(storage.save_run(dupe))
    assert storage._conn.in_transaction is False
    # Subsequent writes commit cleanly and the aborted run never landed.
    run_async(
        storage.save_run(
            RunRecord(workspace_id="w", subject_id="s", idempotency_key="k2")
        )
    )
    runs = run_async(storage.list_runs(workspace_id="w"))
    assert len(runs) == 2
    assert dupe.run_id not in {r.run_id for r in runs}


def test_update_recommendation_atomic_read_modify_write_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recommendation read-modify-write is one transaction; failure rolls it back."""
    storage = _store(tmp_path)
    item = RecommendationItem(
        run_id="r", workspace_id="w", subject_id="s", kind="advice", title="t"
    )
    run_async(storage.save_recommendation(item))

    in_txn_at_modify: list[bool] = []

    def failing_dump(obj: Any) -> str:
        in_txn_at_modify.append(storage._conn.in_transaction)
        raise sqlite3.OperationalError("injected mid read-modify-write")

    monkeypatch.setattr(SqliteStorageService, "_dump", staticmethod(failing_dump))
    with pytest.raises(sqlite3.OperationalError):
        run_async(
            storage.update_recommendation(
                item.recommendation_id, status=RecommendationStatus.ACCEPTED
            )
        )
    # The read ran inside the write transaction (atomic read-modify-write)...
    assert in_txn_at_modify == [True]
    # ...and the aborted update was rolled back, leaving the stored row untouched.
    assert storage._conn.in_transaction is False
    monkeypatch.undo()
    stored = run_async(storage.get_recommendation(item.recommendation_id))
    assert stored is not None and stored.status is RecommendationStatus.PROPOSED
    updated = run_async(
        storage.update_recommendation(
            item.recommendation_id, status=RecommendationStatus.ACCEPTED
        )
    )
    assert updated is not None and updated.status is RecommendationStatus.ACCEPTED
    assert storage._conn.in_transaction is False


def test_idempotent_create_paths_leave_no_open_transaction(tmp_path: Path) -> None:
    """Both branches of the idempotent run create end their explicit transaction."""
    storage = _store(tmp_path)
    first = RunRecord(workspace_id="w", subject_id="s", idempotency_key="k")
    _, created = run_async(storage.save_run_if_absent_by_idempotency(first))
    assert created is True
    assert storage._conn.in_transaction is False
    second = RunRecord(workspace_id="w", subject_id="s", idempotency_key="k")
    got, created = run_async(storage.save_run_if_absent_by_idempotency(second))
    assert created is False and got.run_id == first.run_id
    assert storage._conn.in_transaction is False
    # A missing update target also ends its transaction cleanly.
    assert run_async(storage.update_recommendation("nope")) is None
    assert storage._conn.in_transaction is False
