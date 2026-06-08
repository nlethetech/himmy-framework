"""Behavioral tests for the StateGraph orchestrator.

Covers the load-bearing graph semantics: typed shared state with reducers,
static + conditional edges, parallel fan-out / join via the BSP superstep model,
loops with guards (per-node visit cap + global recursion limit), and the
friendly dead-end / validation behavior. Audit emission, durable resume, and
deterministic replay live in sibling files.
"""

from __future__ import annotations

import pytest

from himmy.orchestrators.state_graph import (
    END,
    GraphError,
    GraphRecursionError,
    StateGraph,
    add_reducer,
)
from tests.conftest import run_async


def test_linear_graph_threads_state() -> None:
    g = StateGraph("linear")

    async def a(state: dict) -> dict:
        return {"a": 1}

    async def b(state: dict) -> dict:
        # b sees a's contribution merged into shared state.
        return {"b": state["a"] + 1}

    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.set_entry_point("a")

    res = run_async(g.compile().invoke({}))
    assert res.status == "completed"
    assert res.final_state == {"a": 1, "b": 2}
    assert res.node_sequence == ["a", "b"]


def test_sync_node_is_supported() -> None:
    g = StateGraph("sync")

    def double(state: dict) -> dict:
        return {"n": state.get("n", 0) * 2}

    g.add_node("double", double)
    g.add_edge("double", END)
    g.set_entry_point("double")

    res = run_async(g.compile().invoke({"n": 21}))
    assert res.final_state["n"] == 42


def test_node_returning_none_is_a_noop_delta() -> None:
    g = StateGraph("noop")

    async def touch(state: dict) -> None:
        return None

    g.add_node("touch", touch)
    g.add_edge("touch", END)
    g.set_entry_point("touch")

    res = run_async(g.compile().invoke({"keep": True}))
    assert res.final_state == {"keep": True}


def test_conditional_loop_with_router_terminates() -> None:
    g = StateGraph("counter")

    async def inc(state: dict) -> dict:
        return {"n": state.get("n", 0) + 1}

    def route(state: dict) -> str:
        return "inc" if state["n"] < 3 else END

    g.add_node("inc", inc)
    g.add_conditional_edges("inc", route)
    g.set_entry_point("inc")

    res = run_async(g.compile().invoke({"n": 0}))
    assert res.status == "completed"
    assert res.final_state["n"] == 3
    assert res.node_sequence == ["inc", "inc", "inc"]
    assert res.supersteps == 3


def test_parallel_fan_out_and_join_with_reducer() -> None:
    g = StateGraph("fanout")
    g.set_reducer("results", add_reducer)

    async def fan(state: dict) -> dict:
        return {}

    async def worker_a(state: dict) -> dict:
        return {"results": ["A"]}

    async def worker_b(state: dict) -> dict:
        return {"results": ["B"]}

    async def join(state: dict) -> dict:
        # join runs once and sees BOTH parallel contributions merged.
        return {"joined": sorted(state["results"])}

    g.add_node("fan", fan)
    g.add_node("worker_a", worker_a)
    g.add_node("worker_b", worker_b)
    g.add_node("join", join)
    g.add_conditional_edges("fan", lambda s: ["worker_a", "worker_b"])
    g.add_edge("worker_a", "join")
    g.add_edge("worker_b", "join")
    g.add_edge("join", END)
    g.set_entry_point("fan")

    res = run_async(g.compile().invoke({}))
    assert res.status == "completed"
    assert sorted(res.final_state["results"]) == ["A", "B"]
    assert res.final_state["joined"] == ["A", "B"]
    # join target reached by two predecessors must run exactly once.
    assert res.node_sequence.count("join") == 1


def test_join_runs_once_per_superstep() -> None:
    """A diamond fan-in collapses the duplicated frontier to a single node run."""
    g = StateGraph("diamond")
    g.set_reducer("trail", add_reducer)

    async def start(state: dict) -> dict:
        return {"trail": ["start"]}

    async def left(state: dict) -> dict:
        return {"trail": ["left"]}

    async def right(state: dict) -> dict:
        return {"trail": ["right"]}

    async def merge(state: dict) -> dict:
        return {"trail": ["merge"]}

    g.add_node("start", start)
    g.add_node("left", left)
    g.add_node("right", right)
    g.add_node("merge", merge)
    g.add_conditional_edges("start", lambda s: ["left", "right"])
    g.add_edge("left", "merge")
    g.add_edge("right", "merge")
    g.add_edge("merge", END)
    g.set_entry_point("start")

    res = run_async(g.compile().invoke({}))
    assert res.node_sequence == ["start", "left", "right", "merge"]
    assert res.final_state["trail"].count("merge") == 1


def test_add_reducer_semantics() -> None:
    assert add_reducer(None, [1]) == [1]
    assert add_reducer([1], [2, 3]) == [1, 2, 3]
    assert add_reducer([1], 2) == [1, 2]
    assert add_reducer(1, 2) == 3
    assert add_reducer("old", "new") == "new"  # non-additive -> last-write-wins


def test_unreduced_key_is_last_write_wins() -> None:
    g = StateGraph("lww")

    async def a(state: dict) -> dict:
        return {"x": "first"}

    async def b(state: dict) -> dict:
        return {"x": "second"}

    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.set_entry_point("a")

    res = run_async(g.compile().invoke({}))
    assert res.final_state["x"] == "second"


def test_recursion_limit_guard_trips() -> None:
    g = StateGraph("spin")

    async def spin(state: dict) -> dict:
        return {"n": state.get("n", 0) + 1}

    g.add_node("spin", spin)
    g.add_conditional_edges("spin", lambda s: "spin")
    g.set_entry_point("spin")

    cg = g.compile(recursion_limit=5)
    with pytest.raises(GraphRecursionError):
        run_async(cg.invoke({}))


def test_per_node_max_visits_guard() -> None:
    g = StateGraph("limited")

    async def loop(state: dict) -> dict:
        return {"n": state.get("n", 0) + 1}

    g.add_node("loop", loop, max_visits=2)
    g.add_conditional_edges("loop", lambda s: "loop")
    g.set_entry_point("loop")

    cg = g.compile(recursion_limit=100)
    with pytest.raises(GraphRecursionError) as exc:
        run_async(cg.invoke({}))
    assert "max_visits" in str(exc.value)


def test_node_with_no_outgoing_edges_ends_branch() -> None:
    g = StateGraph("deadend")

    async def only(state: dict) -> dict:
        return {"done": True}

    g.add_node("only", only)
    g.set_entry_point("only")

    res = run_async(g.compile().invoke({}))
    assert res.status == "completed"
    assert res.final_state["done"] is True


def test_compile_rejects_missing_entry_point() -> None:
    g = StateGraph("noentry")
    g.add_node("x", lambda s: {})
    with pytest.raises(GraphError, match="entry point"):
        g.compile()


def test_compile_rejects_dangling_edge_target() -> None:
    g = StateGraph("dangle")
    g.add_node("a", lambda s: {})
    g.add_edge("a", "ghost")
    g.set_entry_point("a")
    with pytest.raises(GraphError, match="ghost"):
        g.compile()


def test_router_returning_unknown_target_raises() -> None:
    g = StateGraph("badroute")
    g.add_node("a", lambda s: {})
    g.add_conditional_edges("a", lambda s: "nope")
    g.set_entry_point("a")
    with pytest.raises(GraphError, match="nope"):
        run_async(g.compile().invoke({}))


def test_node_raising_surfaces_as_graph_error() -> None:
    g = StateGraph("boom")

    async def boom(state: dict) -> dict:
        raise ValueError("kaboom")

    g.add_node("boom", boom)
    g.set_entry_point("boom")
    with pytest.raises(GraphError, match="kaboom"):
        run_async(g.compile().invoke({}))


def test_node_returning_non_mapping_is_rejected() -> None:
    g = StateGraph("badreturn")

    async def bad(state: dict) -> int:
        return 42  # not a mapping

    g.add_node("bad", bad)  # type: ignore[arg-type]
    g.set_entry_point("bad")
    with pytest.raises(GraphError, match="mapping"):
        run_async(g.compile().invoke({}))


def test_reserved_and_duplicate_node_names_rejected() -> None:
    g = StateGraph("names")
    with pytest.raises(GraphError, match="reserved"):
        g.add_node(END, lambda s: {})
    g.add_node("a", lambda s: {})
    with pytest.raises(GraphError, match="duplicate"):
        g.add_node("a", lambda s: {})


def test_node_cannot_mutate_caller_state_in_place() -> None:
    """A node receives a copy of state, so mutating its arg cannot corrupt the run."""
    g = StateGraph("isolation")

    async def naughty(state: dict) -> dict:
        state["secret"] = "leaked"  # mutate the arg directly
        return {"clean": True}

    g.add_node("naughty", naughty)
    g.add_edge("naughty", END)
    g.set_entry_point("naughty")

    res = run_async(g.compile().invoke({"orig": 1}))
    # The in-place mutation did not leak into shared state; only the returned
    # delta merged.
    assert "secret" not in res.final_state
    assert res.final_state == {"orig": 1, "clean": True}
