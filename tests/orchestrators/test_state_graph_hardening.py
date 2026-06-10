"""Hardening tests for the StateGraph orchestrator.

Covers the superstep concurrency contract (a failing node must cancel AND await
its sibling tasks instead of orphaning them) and the opt-in ``max_state_bytes``
OOM guard (typed :class:`GraphStateSizeError`, default unlimited unchanged).
"""

from __future__ import annotations

import asyncio

import pytest

from himmy.orchestrators.state_graph import (
    END,
    GraphError,
    GraphStateSizeError,
    StateGraph,
    add_reducer,
)
from tests.conftest import run_async


def test_failing_node_cancels_and_awaits_siblings() -> None:
    """One node raising must not orphan its parallel siblings (no leaked tasks)."""

    async def main() -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = False
        sibling_finished = False

        async def slow(state: dict) -> dict:
            nonlocal sibling_cancelled, sibling_finished
            sibling_started.set()
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                sibling_cancelled = True
                raise
            sibling_finished = True
            return {}

        async def fail(state: dict) -> dict:
            # Only blow up once the sibling is definitely in flight.
            await sibling_started.wait()
            raise RuntimeError("boom")

        g = StateGraph("fanout-fail")
        g.add_node("fan", lambda state: {})
        g.add_node("slow", slow)
        g.add_node("fail", fail)
        g.add_edge("fan", "slow")
        g.add_edge("fan", "fail")
        g.add_edge("slow", END)
        g.add_edge("fail", END)
        g.set_entry_point("fan")

        with pytest.raises(GraphError, match="boom"):
            await g.compile().invoke({})

        # By the time invoke raises, the sibling must have been cancelled and
        # awaited — a bare gather would leave it running detached.
        assert sibling_cancelled
        assert not sibling_finished
        current = asyncio.current_task()
        leftovers = [
            t for t in asyncio.all_tasks() if t is not current and not t.done()
        ]
        assert leftovers == []

    run_async(main())


def _growing_graph() -> StateGraph:
    """A self-looping node whose ``add``-reduced key grows every superstep."""
    g = StateGraph("growing")
    g.set_reducer("acc", add_reducer)

    def grow(state: dict) -> dict:
        return {"acc": ["x" * 64]}

    def router(state: dict) -> str:
        return "grow" if len(state.get("acc", [])) < 10 else END

    g.add_node("grow", grow)
    g.add_conditional_edges("grow", router)
    g.set_entry_point("grow")
    return g


def test_max_state_bytes_raises_typed_graph_error() -> None:
    compiled = _growing_graph().compile(max_state_bytes=200)
    with pytest.raises(GraphStateSizeError, match="max_state_bytes=200"):
        run_async(compiled.invoke({}))


def test_max_state_bytes_default_is_unlimited() -> None:
    # No limit set anywhere: the same growth completes exactly as before.
    res = run_async(_growing_graph().compile().invoke({}))
    assert res.status == "completed"
    assert len(res.final_state["acc"]) == 10


def test_max_state_bytes_invoke_overrides_compile() -> None:
    compiled = _growing_graph().compile()  # unlimited at compile time
    with pytest.raises(GraphStateSizeError):
        run_async(compiled.invoke({}, max_state_bytes=200))
    # The override is per-run: the same compiled graph stays unlimited after.
    res = run_async(compiled.invoke({}))
    assert res.status == "completed"


def test_max_state_bytes_must_be_positive() -> None:
    with pytest.raises(GraphError, match="max_state_bytes"):
        _growing_graph().compile(max_state_bytes=0)
    compiled = _growing_graph().compile()
    with pytest.raises(GraphError, match="max_state_bytes"):
        run_async(compiled.invoke({}, max_state_bytes=0))
