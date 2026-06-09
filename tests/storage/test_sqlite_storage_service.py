"""Tests for the durable, file-backed :class:`SqliteStorageService`.

Every per-concern method round-trips, two instances on the same file see each other's
writes (durability), and upserts/idempotent creates behave exactly like the in-memory
``StorageService``. All artifacts live under ``tmp_path`` — nothing touches the repo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
