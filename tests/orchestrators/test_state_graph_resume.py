"""StateGraph durable resume: an interrupted graph continues from a checkpoint.

Verifies the BSP-superstep checkpoint contract: after each completed superstep the
graph persists state + frontier + visit counts to a
:class:`GraphCheckpointStore`, so a run interrupted by a timeout (or a crash, or a
HITL pause) resumes deterministically from where it stopped — including across a
brand-new compiled graph object, the cross-process guarantee.
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.orchestrators.state_graph import END, GraphError, StateGraph
from himmy.runtime.checkpoint import (
    GRAPH_COMPLETED,
    GRAPH_INTERRUPTED,
    InMemoryGraphCheckpointStore,
    SqliteGraphCheckpointStore,
)
from tests.conftest import run_async


def _slow_graph(slow_sleep: float) -> StateGraph:
    g = StateGraph("pipeline")

    async def fast(state: dict) -> dict:
        return {"step1": True}

    async def slow(state: dict) -> dict:
        await asyncio.sleep(slow_sleep)
        return {"step2": True}

    g.add_node("fast", fast)
    g.add_node("slow", slow)
    g.add_edge("fast", "slow")
    g.add_edge("slow", END)
    g.set_entry_point("fast")
    return g


def test_checkpoint_written_after_each_superstep() -> None:
    store = InMemoryGraphCheckpointStore()
    g = StateGraph("two")
    g.add_node("a", lambda s: {"a": 1})
    g.add_node("b", lambda s: {"b": 2})
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.set_entry_point("a")
    cg = g.compile(checkpoint_store=store)

    res = run_async(cg.invoke({}))
    saved = store.load(res.checkpoint_id)
    assert saved is not None
    assert saved.status == GRAPH_COMPLETED
    assert saved.state == {"a": 1, "b": 2}
    assert saved.superstep == 2
    assert saved.completed_nodes == ["a", "b"]


def test_timeout_interrupts_and_checkpoints_progress() -> None:
    store = SqliteGraphCheckpointStore(":memory:")
    cg = _slow_graph(slow_sleep=5.0).compile(checkpoint_store=store)

    res = run_async(cg.invoke({}, timeout_seconds=0.2))
    assert res.status == GRAPH_INTERRUPTED
    # The fast superstep committed before the slow node hung.
    assert res.final_state == {"step1": True}

    saved = store.load(res.checkpoint_id)
    assert saved is not None
    assert saved.status == GRAPH_INTERRUPTED
    assert saved.frontier == ["slow"]
    assert saved.state == {"step1": True}
    assert saved.completed_nodes == ["fast"]


def test_resume_continues_from_checkpoint_in_fresh_graph() -> None:
    store = SqliteGraphCheckpointStore(":memory:")
    cg = _slow_graph(slow_sleep=5.0).compile(checkpoint_store=store)
    interrupted = run_async(cg.invoke({}, timeout_seconds=0.2))
    assert interrupted.status == GRAPH_INTERRUPTED

    # A brand-new compiled graph (same topology, fast slow node) resumes by id.
    cg2 = _slow_graph(slow_sleep=0.0).compile(checkpoint_store=store)
    resumed = run_async(cg2.invoke(resume=interrupted.checkpoint_id))
    assert resumed.status == GRAPH_COMPLETED
    assert resumed.final_state == {"step1": True, "step2": True}
    # fast already ran before the interrupt; only slow runs on resume.
    assert resumed.node_sequence == ["fast", "slow"]
    assert resumed.checkpoint_id == interrupted.checkpoint_id


def test_resume_from_checkpoint_object() -> None:
    store = InMemoryGraphCheckpointStore()
    cg = _slow_graph(slow_sleep=5.0).compile(checkpoint_store=store)
    interrupted = run_async(cg.invoke({}, timeout_seconds=0.2))
    chk = store.load(interrupted.checkpoint_id)
    assert chk is not None

    cg2 = _slow_graph(slow_sleep=0.0).compile(checkpoint_store=store)
    resumed = run_async(cg2.invoke(resume=chk))
    assert resumed.status == GRAPH_COMPLETED
    assert resumed.final_state["step2"] is True


def test_resume_unknown_checkpoint_raises() -> None:
    cg = _slow_graph(slow_sleep=0.0).compile()
    with pytest.raises(GraphError, match="no graph checkpoint"):
        run_async(cg.invoke(resume="does-not-exist"))


def test_resume_wrong_graph_name_raises() -> None:
    store = InMemoryGraphCheckpointStore()
    cg = _slow_graph(slow_sleep=5.0).compile(checkpoint_store=store)
    interrupted = run_async(cg.invoke({}, timeout_seconds=0.2))

    other = StateGraph("a_different_graph")
    other.add_node("x", lambda s: {})
    other.set_entry_point("x")
    cg_other = other.compile(checkpoint_store=store)
    with pytest.raises(GraphError, match="checkpoint is for graph"):
        run_async(cg_other.invoke(resume=interrupted.checkpoint_id))


def test_completed_run_is_idempotent_on_resume() -> None:
    """Resuming an already-completed checkpoint yields the same final state."""
    store = InMemoryGraphCheckpointStore()
    g = StateGraph("once")
    g.add_node("a", lambda s: {"a": 1})
    g.add_edge("a", END)
    g.set_entry_point("a")
    cg = g.compile(checkpoint_store=store)
    first = run_async(cg.invoke({}))

    # The completed checkpoint has an empty frontier; resuming is a no-op loop.
    resumed = run_async(cg.invoke(resume=first.checkpoint_id))
    assert resumed.status == GRAPH_COMPLETED
    assert resumed.final_state == {"a": 1}


def test_sqlite_checkpoint_survives_new_store_instance() -> None:
    """A checkpoint written via one Sqlite store is readable via another (durable)."""
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        store1 = SqliteGraphCheckpointStore(path)
        cg = _slow_graph(slow_sleep=5.0).compile(checkpoint_store=store1)
        interrupted = run_async(cg.invoke({}, timeout_seconds=0.2))
        store1.close()

        # A fresh store over the same file (simulating a process restart) resumes.
        store2 = SqliteGraphCheckpointStore(path)
        loaded = store2.load(interrupted.checkpoint_id)
        assert loaded is not None
        assert loaded.frontier == ["slow"]
        cg2 = _slow_graph(slow_sleep=0.0).compile(checkpoint_store=store2)
        resumed = run_async(cg2.invoke(resume=interrupted.checkpoint_id))
        assert resumed.status == GRAPH_COMPLETED
        assert resumed.final_state == {"step1": True, "step2": True}
        store2.close()
    finally:
        for suffix in ("", "-wal", "-shm"):
            try:
                os.remove(path + suffix)
            except OSError:
                pass


def test_interrupted_checkpoints_listable_by_status() -> None:
    store = SqliteGraphCheckpointStore(":memory:")
    cg = _slow_graph(slow_sleep=5.0).compile(checkpoint_store=store)
    run_async(cg.invoke({}, timeout_seconds=0.2))
    pending = store.list_by_status(GRAPH_INTERRUPTED)
    assert len(pending) == 1
    assert pending[0].frontier == ["slow"]
