"""StateGraph HITL machinery: NodeInterrupt escape, durable interrupt, resume splice.

These exercise the graph-level pieces that make nested orchestration HITL work, below the
runner integration in ``tests/application/test_orchestration_hitl.py``:

* a node that raises :class:`NodeInterrupt` ESCAPES the node runner as a durable graph
  interrupt (NOT re-wrapped as a node failure), with the node left in the frontier;
* the orchestration resume splice runs a ``resume_member`` callback exactly once, advances
  the paused node once, and completes;
* a pre-existing TIMEOUT-interrupted checkpoint (no member reference) takes the ordinary
  durable resume and never errors / never approves;
* two nodes pausing in one superstep is rejected with a clear error;
* the v1 -> v2 graph checkpoint schema migration round-trips.

Plus the orchestrator-level FAIL-CLOSED for the multi_agent delegate sub-run (where
delegates are actually wired — the stored-agent runner path drops them).
"""

from __future__ import annotations

import json

import pytest

from himmy.orchestrators import (
    END,
    CompiledStateGraph,
    GraphError,
    MemberResumeOutcome,
    NodeInterrupt,
    StateGraph,
)
from himmy.runtime.checkpoint import (
    GRAPH_CHECKPOINT_SCHEMA_VERSION,
    GraphCheckpoint,
    InMemoryGraphCheckpointStore,
    SqliteGraphCheckpointStore,
    _migrate_graph_checkpoint,
)
from tests.conftest import run_async


def _linear_graph(store: object, make: object) -> CompiledStateGraph:
    """A 2-node linear graph a -> b -> END over ``store``."""
    graph = StateGraph(name="workflow")
    graph.add_node("a", make("a"))
    graph.add_node("b", make("b"))
    graph.set_entry_point("a")
    graph.add_edge("a", "b")
    graph.add_edge("b", END)
    return graph.compile(checkpoint_store=store)


def test_node_interrupt_escapes_as_durable_interrupt() -> None:
    """A NodeInterrupt is a durable graph PAUSE, not a node failure.

    The paused node stays in the frontier (not completed, not visit-bumped), the member
    reference is stamped, and the status is interrupted — distinguishable from a timeout.
    """
    store = InMemoryGraphCheckpointStore()

    def _make(name: str):
        async def _node(state: dict) -> dict:
            if name == "a":
                raise NodeInterrupt("a", "member-cp-1")
            return {"ran": name}

        return _node

    compiled = _linear_graph(store, _make)
    result = run_async(compiled.invoke({}, checkpoint_id="cp"))

    assert result.status == "interrupted"
    chk = store.load("cp")
    assert chk is not None
    assert chk.status == "interrupted"
    assert chk.paused_node == "a"
    assert chk.member_checkpoint_id == "member-cp-1"
    assert chk.frontier == ["a"]  # still pending — a resume re-enters it
    assert chk.completed_nodes == []
    assert chk.error is None  # NOT a failure


def test_node_failure_is_still_a_graph_error() -> None:
    """A genuine node exception is STILL surfaced as a GraphError (regression guard)."""
    store = InMemoryGraphCheckpointStore()

    def _make(name: str):
        async def _node(state: dict) -> dict:
            if name == "a":
                raise ValueError("boom")
            return {"ran": name}

        return _node

    compiled = _linear_graph(store, _make)
    with pytest.raises(GraphError, match="raised"):
        run_async(compiled.invoke({}, checkpoint_id="cp"))


def test_resume_splice_runs_member_once_and_completes() -> None:
    """The resume splice runs ``resume_member`` once, advances the node, and completes."""
    store = InMemoryGraphCheckpointStore()
    calls: list[str] = []

    def _make(name: str):
        async def _node(state: dict) -> dict:
            if name == "a":
                raise NodeInterrupt("a", "member-cp-1")
            return {"last_output": "b-done"}

        return _node

    compiled = _linear_graph(store, _make)
    paused = run_async(compiled.invoke({}, checkpoint_id="cp"))
    assert paused.status == "interrupted"

    async def _resume_member(node: str, member_cp: str, state: dict):
        calls.append(f"{node}:{member_cp}")
        return MemberResumeOutcome(delta={"last_output": "a-approved"})

    resumed = run_async(compiled.invoke(resume="cp", resume_member=_resume_member))

    assert calls == ["a:member-cp-1"]  # member resumed exactly once
    assert resumed.status == "completed"
    chk = store.load("cp")
    assert chk is not None
    assert chk.completed_nodes == ["a", "b"]  # paused node completed + b ran
    assert chk.visit_counts == {"a": 1, "b": 1}
    assert chk.member_checkpoint_id == ""  # cleared
    assert chk.paused_node == ""
    # The resumed member's delta + node b's output both landed on shared state.
    assert chk.state.get("last_output") == "b-done"
    assert resumed.final_state.get("last_output") == "b-done"


def test_resume_splice_can_pause_again_at_same_node() -> None:
    """When the member pauses AGAIN, the interrupt is re-stamped at the same node."""
    store = InMemoryGraphCheckpointStore()

    def _make(name: str):
        async def _node(state: dict) -> dict:
            if name == "a":
                raise NodeInterrupt("a", "member-cp-1")
            return {"ran": name}

        return _node

    compiled = _linear_graph(store, _make)
    run_async(compiled.invoke({}, checkpoint_id="cp"))

    async def _resume_member(node: str, member_cp: str, state: dict):
        return MemberResumeOutcome(paused_again=True, member_checkpoint_id="member-cp-2")

    again = run_async(compiled.invoke(resume="cp", resume_member=_resume_member))
    assert again.status == "interrupted"
    chk = store.load("cp")
    assert chk is not None
    assert chk.paused_node == "a"  # still at the same node
    assert chk.member_checkpoint_id == "member-cp-2"  # re-stamped
    assert chk.completed_nodes == []  # NOT advanced


def test_timeout_interrupt_takes_ordinary_resume_without_approve() -> None:
    """A pre-existing TIMEOUT interrupt (no member fields) resumes ordinarily, no error.

    The resume splice is gated on ``member_checkpoint_id`` being set; a timeout interrupt
    leaves it empty so ``resume_member`` is never invoked and the graph resumes normally.
    """
    store = InMemoryGraphCheckpointStore()
    ran: list[str] = []

    def _make(name: str):
        async def _node(state: dict) -> dict:
            ran.append(name)
            return {"ran": name}

        return _node

    compiled = _linear_graph(store, _make)
    # Seed a timeout-interrupted checkpoint paused before node b (member fields EMPTY).
    store.save(
        GraphCheckpoint(
            checkpoint_id="cp",
            graph_name="workflow",
            status="interrupted",
            state={"ran": "a"},
            frontier=["b"],
            superstep=1,
            visit_counts={"a": 1},
            completed_nodes=["a"],
        )
    )

    sentinel = object()

    async def _resume_member(node: str, member_cp: str, state: dict):  # pragma: no cover
        raise AssertionError("resume_member must not run for a timeout interrupt")

    result = run_async(compiled.invoke(resume="cp", resume_member=_resume_member))
    assert result.status == "completed"
    assert ran == ["b"]  # only b ran on resume; a was already done
    assert sentinel is sentinel


def test_two_nodes_pausing_in_one_superstep_is_rejected() -> None:
    """A fan-out where two nodes pause at once is a clear error (one member slot)."""
    store = InMemoryGraphCheckpointStore()

    def _make(name: str):
        async def _node(state: dict) -> dict:
            raise NodeInterrupt(name, f"member-cp-{name}")

        return _node

    # A fan-out: entry -> {a, b} both pause in the same superstep.
    graph = StateGraph(name="workflow")
    graph.add_node("entry", lambda s: {"x": 1})
    graph.add_node("a", _make("a"))
    graph.add_node("b", _make("b"))
    graph.set_entry_point("entry")
    graph.add_conditional_edges("entry", lambda s: ["a", "b"])
    graph.add_edge("a", END)
    graph.add_edge("b", END)
    compiled = graph.compile(checkpoint_store=store)

    with pytest.raises(GraphError, match="two or more nodes paused"):
        run_async(compiled.invoke({}, checkpoint_id="cp"))


def test_graph_checkpoint_v1_to_v2_migration_round_trips(tmp_path) -> None:
    """A persisted v1 graph checkpoint (no HITL fields) loads cleanly under v2."""
    raw_v1 = {
        "schema_version": 1,
        "checkpoint_id": "old",
        "graph_name": "workflow",
        "status": "interrupted",
        "state": {"k": "v"},
        "frontier": ["b"],
        "superstep": 1,
        "visit_counts": {"a": 1},
        "completed_nodes": ["a"],
        "error": None,
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
    }
    migrated = _migrate_graph_checkpoint(dict(raw_v1))
    assert migrated["schema_version"] == GRAPH_CHECKPOINT_SCHEMA_VERSION == 2
    assert migrated["paused_node"] == ""
    assert migrated["member_checkpoint_id"] == ""
    chk = GraphCheckpoint.model_validate(migrated)
    assert chk.completed_nodes == ["a"]
    assert chk.paused_node == ""

    # And through the durable SQLite store: write the v1 JSON, read it back upgraded.
    store = SqliteGraphCheckpointStore(str(tmp_path / "g.db"))
    store._conn.execute(
        "INSERT INTO graph_checkpoints "
        "(checkpoint_id, graph_name, status, data, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("old", "workflow", "interrupted", json.dumps(raw_v1), "t", "t"),
    )
    store._conn.commit()
    loaded = store.load("old")
    assert loaded is not None
    assert loaded.schema_version == 2
    assert loaded.member_checkpoint_id == ""


# --------------------------------------------------------------------------- #
# Orchestrator-level fail-closed: the multi_agent delegate sub-run             #
# --------------------------------------------------------------------------- #


def test_multi_agent_delegate_subrun_fails_closed() -> None:
    """A gated tool inside a multi_agent DELEGATE sub-run fails the run closed.

    Tested at the orchestrator level (where ``delegates`` are actually wired — the stored
    AgentSpec has no delegates field, so the runner path can't express this).
    """
    from himmy.agents.personas.persona import Persona
    from himmy.orchestrators import AgentTeam, MultiAgentOrchestrator, TeamMember
    from himmy.orchestrators.multi_agent import HitlUnsupportedError
    from himmy.runtime.builder import build_runtime
    from himmy.services.storage.service import StorageService
    from himmy.services.tools.registry import ToolRegistry
    from tests.application import _per_run_tools

    _per_run_tools.WIRE_CALLS.clear()
    storage = StorageService()
    registry = ToolRegistry()
    _per_run_tools.register(registry)
    runtime, _inf, _tools = build_runtime(tool_registry=registry, storage=storage)
    team = AgentTeam(
        members=[
            TeamMember(
                name="manager", persona=Persona(name="manager"), delegates=["worker"]
            ),
            TeamMember(
                name="worker",
                persona=Persona(name="worker"),
                tools=["wire_money"],
            ),
        ],
        entry="manager",
    )
    orch = MultiAgentOrchestrator(runtime, team, registry, max_turns=4)
    with pytest.raises(HitlUnsupportedError, match="wire_money"):
        run_async(orch.run("delegate to the worker to wire money"))
    assert not _per_run_tools.WIRE_CALLS  # the gated tool never executed un-approved
