"""StateGraph provenance: every node/edge/state transition lands on the spine.

Verifies the graph emits its lifecycle as :class:`RunEvent`s through the same
isolated fan-out the runtime uses (event sink + entity registry + caller
callback), so a graph run is auditable and replayable just like an agent run.
"""

from __future__ import annotations

from himmy.core.events import EventType, RunEvent
from himmy.entities.registry import EntityRegistry
from himmy.orchestrators.state_graph import END, StateGraph
from tests.conftest import run_async


class _RecordingSink:
    """A minimal :class:`~himmy.core.events.EventSink` that captures events."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


def _two_node_graph() -> StateGraph:
    g = StateGraph("audited")

    async def n1(state: dict) -> dict:
        return {"x": 1}

    async def n2(state: dict) -> dict:
        return {"y": 2}

    g.add_node("n1", n1)
    g.add_node("n2", n2)
    g.add_edge("n1", "n2")
    g.add_edge("n2", END)
    g.set_entry_point("n1")
    return g


def test_lifecycle_events_emitted_to_sink() -> None:
    sink = _RecordingSink()
    cg = _two_node_graph().compile(memory_store=sink)
    run_async(cg.invoke({}))

    types = [e.event_type for e in sink.events]
    assert types[0] == EventType.GRAPH_STARTED
    assert types[-1] == EventType.GRAPH_FINISHED
    assert EventType.GRAPH_NODE_STARTED in types
    assert EventType.GRAPH_NODE_COMPLETED in types
    assert EventType.GRAPH_EDGE_TAKEN in types
    assert EventType.GRAPH_CHECKPOINTED in types


def test_node_started_completed_pair_per_node() -> None:
    sink = _RecordingSink()
    cg = _two_node_graph().compile(memory_store=sink)
    run_async(cg.invoke({}))

    started = [
        e.payload["node"]
        for e in sink.events
        if e.event_type == EventType.GRAPH_NODE_STARTED
    ]
    completed = [
        e.payload["node"]
        for e in sink.events
        if e.event_type == EventType.GRAPH_NODE_COMPLETED
    ]
    assert started == ["n1", "n2"]
    assert completed == ["n1", "n2"]


def test_node_completed_carries_delta_keys() -> None:
    sink = _RecordingSink()
    cg = _two_node_graph().compile(memory_store=sink)
    run_async(cg.invoke({}))

    completed = {
        e.payload["node"]: e.payload["delta_keys"]
        for e in sink.events
        if e.event_type == EventType.GRAPH_NODE_COMPLETED
    }
    assert completed["n1"] == ["x"]
    assert completed["n2"] == ["y"]


def test_edges_recorded_with_from_to() -> None:
    sink = _RecordingSink()
    cg = _two_node_graph().compile(memory_store=sink)
    run_async(cg.invoke({}))

    edges = [
        (e.payload["from"], e.payload["to"])
        for e in sink.events
        if e.event_type == EventType.GRAPH_EDGE_TAKEN
    ]
    assert ("n1", "n2") in edges
    assert ("n2", END) in edges


def test_events_projected_into_entity_registry() -> None:
    reg = EntityRegistry()
    cg = _two_node_graph().compile(entity_registry=reg)
    run_async(cg.invoke({}))

    run_events = reg.list_by_kind("run_event")
    # GRAPH_STARTED + 2x(NODE_STARTED+NODE_COMPLETED) + edges + checkpoints + FINISHED.
    assert len(run_events) >= 8
    kinds = {r.payload["event_type"] for r in run_events}
    assert "GRAPH_STARTED" in kinds
    assert "GRAPH_NODE_COMPLETED" in kinds
    assert "GRAPH_FINISHED" in kinds


def test_caller_callback_receives_progress() -> None:
    seen: list[EventType] = []

    async def cb(event: RunEvent) -> None:
        seen.append(event.event_type)

    cg = _two_node_graph().compile(on_event=cb)
    run_async(cg.invoke({}))
    assert EventType.GRAPH_STARTED in seen
    assert EventType.GRAPH_FINISHED in seen


def test_failing_sink_does_not_break_the_run() -> None:
    class _BoomSink:
        async def append_event(self, event: RunEvent) -> None:
            raise RuntimeError("sink down")

    cg = _two_node_graph().compile(memory_store=_BoomSink())
    res = run_async(cg.invoke({}))
    # The run still completes despite the broken sink (isolated fan-out).
    assert res.status == "completed"
    assert res.final_state == {"x": 1, "y": 2}


def test_graph_node_projects_to_record() -> None:
    """A node descriptor projects to a deterministic, idempotent EntityRecord."""
    g = StateGraph("proj")
    g.add_node("alpha", lambda s: {})
    g.set_entry_point("alpha")
    cg = g.compile()  # noqa: F841 - exercises compile path

    from himmy.orchestrators.state_graph import _Node

    node = _Node(name="alpha", fn=lambda s: {})
    rec1 = node.to_record()
    rec2 = node.to_record()
    assert rec1.record_id == rec2.record_id
    assert rec1.kind == "graph_node"
    assert rec1.payload == {"name": "alpha", "max_visits": None}


def test_failed_run_emits_failed_node_event() -> None:
    sink = _RecordingSink()
    g = StateGraph("boom")

    async def boom(state: dict) -> dict:
        raise ValueError("nope")

    g.add_node("boom", boom)
    g.set_entry_point("boom")
    cg = g.compile(memory_store=sink)

    raised = False
    try:
        run_async(cg.invoke({}))
    except Exception:
        raised = True
    assert raised
    failed = [
        e
        for e in sink.events
        if e.event_type == EventType.GRAPH_NODE_COMPLETED
        and e.payload.get("status") == "failed"
    ]
    assert len(failed) == 1
    assert failed[0].error is not None
