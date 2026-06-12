"""Retention/compaction: explicit prune routines bound otherwise-unbounded growth.

run_events, resolved checkpoints, terminal graph checkpoints, and sessions are never
auto-deleted; these tests pin the operator-invoked prune calls that reclaim them and
prove the conservative policy (live/unresolved work is preserved).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from himmy.agents.base_agent.thread import ChatThread
from himmy.core.events import EventType, RunEvent
from himmy.runtime import session as session_mod
from himmy.runtime.checkpoint import (
    APPROVED,
    AWAITING_APPROVAL,
    GRAPH_COMPLETED,
    GRAPH_RUNNING,
    REJECTED,
    AgentCheckpoint,
    GraphCheckpoint,
    SqliteCheckpointStore,
    SqliteGraphCheckpointStore,
)
from himmy.runtime.session import SqliteSessionStore
from himmy.services.storage import sqlite as sqlite_mod
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _old_iso(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


# --------------------------------------------------------------- run_events prune


def test_prune_events_keep_last_caps_stream(tmp_path: Path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    for _ in range(5):
        run_async(store.append_event(RunEvent(event_type=EventType.AGENT_RUN_STARTED)))
    removed = run_async(store.prune_events(keep_last=2))
    assert removed == 3
    assert len(run_async(store.list_events())) == 2


def test_prune_events_older_than_days(tmp_path: Path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    run_async(
        store.append_event(
            RunEvent(event_type=EventType.AGENT_RUN_STARTED, timestamp=_old_iso(100))
        )
    )
    run_async(store.append_event(RunEvent(event_type=EventType.AGENT_RUN_STARTED)))  # now
    removed = run_async(store.prune_events(older_than_days=90))
    assert removed == 1
    remaining = run_async(store.list_events())
    assert len(remaining) == 1  # the recent one survived


def test_prune_events_requires_a_bound(tmp_path: Path) -> None:
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    try:
        run_async(store.prune_events())
    except ValueError:
        pass
    else:  # pragma: no cover - must raise
        raise AssertionError("expected ValueError with no bound given")


def test_prune_events_chunks_oversized_in_list(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A doomed set larger than the IN-list chunk size deletes without raising.

    The single ``DELETE ... WHERE seq IN (?, …)`` binds one variable per doomed row,
    so a first prune of a long-lived db can exceed SQLITE_MAX_VARIABLE_NUMBER and
    raise ``sqlite3.OperationalError: too many SQL variables``. We reproduce that
    limit cheaply by lowering the connection's variable cap (``setlimit``) instead of
    inserting hundreds of thousands of rows, and force a tiny chunk size so a single
    un-chunked IN-list (24 vars) would overrun the cap while the chunked delete (4
    vars per batch) stays under it.
    """
    import sqlite3

    monkeypatch.setattr(sqlite_mod, "_PRUNE_DELETE_CHUNK", 4)
    store = SqliteStorageService(str(tmp_path / "storage.db"))
    for _ in range(25):  # >> the forced chunk size of 4
        run_async(store.append_event(RunEvent(event_type=EventType.AGENT_RUN_STARTED)))
    # SQLITE_LIMIT_VARIABLE_NUMBER == 9. Cap at 5 so any IN-list of 24 doomed seqs
    # would raise "too many SQL variables" unless deletes are chunked under the cap.
    store._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 5)
    removed = run_async(store.prune_events(keep_last=1))
    assert removed == 24  # all but the 1 kept, deleted across multiple chunks
    assert len(run_async(store.list_events())) == 1


# --------------------------------------------------------- resolved checkpoints


def _ckpt(status: str, created_at: str) -> AgentCheckpoint:
    return AgentCheckpoint(status=status, created_at=created_at)


def test_prune_resolved_drops_old_terminal_keeps_live(tmp_path: Path) -> None:
    store = SqliteCheckpointStore(str(tmp_path / "ckpt.db"))
    store.save(_ckpt(APPROVED, _old_iso(60)))  # old + resolved -> pruned
    store.save(_ckpt(REJECTED, _old_iso(60)))  # old + resolved -> pruned
    store.save(_ckpt(APPROVED, _old_iso(1)))  # recent + resolved -> kept
    live = _ckpt(AWAITING_APPROVAL, _old_iso(365))  # old but UNRESOLVED -> kept
    store.save(live)

    removed = store.prune_resolved(older_than_days=30)
    assert removed == 2
    # The unresolved (live) checkpoint is never deleted by age.
    assert store.load(live.checkpoint_id) is not None
    assert len(store.list_by_status(APPROVED)) == 1


# ----------------------------------------------------- terminal graph checkpoints


def test_prune_terminal_graph_keeps_running(tmp_path: Path) -> None:
    store = SqliteGraphCheckpointStore(str(tmp_path / "graph.db"))
    done = GraphCheckpoint(
        graph_name="g", status=GRAPH_COMPLETED, updated_at=_old_iso(60)
    )
    running = GraphCheckpoint(
        graph_name="g", status=GRAPH_RUNNING, updated_at=_old_iso(60)
    )
    store.save(done)
    store.save(running)
    removed = store.prune_terminal(older_than_days=30)
    assert removed == 1
    assert store.load(running.checkpoint_id) is not None  # resumable work preserved
    assert store.load(done.checkpoint_id) is None


# --------------------------------------------------------------- session prune


def test_session_prune_keep_last(tmp_path: Path) -> None:
    store = SqliteSessionStore(str(tmp_path / "sessions.db"))
    for i in range(4):
        store.save(f"s{i}", ChatThread(agent_id="a"))
    removed = store.prune(keep_last=2)
    assert removed == 2
    assert len(store.list_sessions()) == 2


def test_session_prune_older_than_days(tmp_path: Path) -> None:
    store = SqliteSessionStore(str(tmp_path / "sessions.db"))
    store.save("fresh", ChatThread(agent_id="a"))
    # Backdate one session's updated_at directly.
    store._conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
        (_old_iso(100), "fresh"),
    )
    store._conn.commit()
    store.save("newer", ChatThread(agent_id="a"))
    removed = store.prune(older_than_days=90)
    assert removed == 1
    ids = {s.session_id for s in store.list_sessions()}
    assert ids == {"newer"}


def test_session_prune_requires_a_bound(tmp_path: Path) -> None:
    store = SqliteSessionStore(str(tmp_path / "sessions.db"))
    try:
        store.prune()
    except ValueError:
        pass
    else:  # pragma: no cover - must raise
        raise AssertionError("expected ValueError with no bound given")


def test_session_prune_chunks_oversized_in_list(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """A doomed session set larger than the IN-list chunk size deletes without raising.

    Mirrors the run-event chunking guard: the per-row ``DELETE ... WHERE session_id IN
    (?, …)`` would trip SQLITE_MAX_VARIABLE_NUMBER on a first prune of a long-lived
    store. We lower the connection's variable cap to reproduce that limit cheaply and
    force a tiny chunk size so the batched delete must stay under it.
    """
    import sqlite3

    monkeypatch.setattr(session_mod, "_PRUNE_DELETE_CHUNK", 4)
    store = SqliteSessionStore(str(tmp_path / "sessions.db"))
    for i in range(25):  # >> the forced chunk size of 4
        store.save(f"s{i}", ChatThread(agent_id="a"))
    store._conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 5)
    removed = store.prune(keep_last=1)
    assert removed == 24  # deleted across multiple chunks under the lowered cap
    assert len(store.list_sessions()) == 1
