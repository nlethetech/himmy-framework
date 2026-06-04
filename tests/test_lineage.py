"""Tests for the lineage READ layer: links_to, neighbors, and multi-hop trace.

Covers the in-memory ``EntityRegistry`` traversal primitives, the ``LineageGraph``
projections (DOT / filtering), and the end-to-end path where a real run's captured
provenance is traced back through ``RunAppService.get_run_lineage``.
"""

from __future__ import annotations

from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.application.services import RunAppService
from himmy.entities.lineage import LineageDirection, LineageGraph
from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _rec(registry: EntityRegistry, kind: str, sid: str) -> EntityRecord:
    """Register and return a simple v1 record of ``kind`` with stable_id ``sid``."""
    return registry.register(
        EntityRecord.create(stable_id=sid, version=1, kind=kind, payload={"n": sid})
    )


def _hub_graph() -> tuple[EntityRegistry, dict[str, EntityRecord]]:
    """A registry shaped like a run's lineage: a thread hub + spokes.

    Edges: thread->persona (uses_persona), thread->prompt (in_thread),
    thread->snapshot (built_from), snapshot->thread (observed_in_run), and an
    inbound-only event->thread (recorded_in) for direction tests.
    """
    reg = EntityRegistry()
    r = {
        "thread": _rec(reg, "chat_thread", "T"),
        "persona": _rec(reg, "persona", "P"),
        "prompt": _rec(reg, "prompt", "Q"),
        "snapshot": _rec(reg, "context_snapshot", "S"),
        "event": _rec(reg, "run_event", "E"),
    }
    reg.link(
        from_record_id=r["thread"].record_id,
        to_record_id=r["persona"].record_id,
        relation="uses_persona",
    )
    reg.link(
        from_record_id=r["thread"].record_id,
        to_record_id=r["prompt"].record_id,
        relation="in_thread",
    )
    reg.link(
        from_record_id=r["thread"].record_id,
        to_record_id=r["snapshot"].record_id,
        relation="built_from",
    )
    reg.link(
        from_record_id=r["snapshot"].record_id,
        to_record_id=r["thread"].record_id,
        relation="observed_in_run",
    )
    reg.link(
        from_record_id=r["event"].record_id,
        to_record_id=r["thread"].record_id,
        relation="recorded_in",
    )
    return reg, r


# ----------------------------------------------------------------- links_to
def test_links_to_returns_reverse_edges() -> None:
    """links_to surfaces edges pointing INTO a record (the reverse of links_from)."""
    reg, r = _hub_graph()
    into_persona = reg.links_to(r["persona"].record_id)
    assert [link.relation for link in into_persona] == ["uses_persona"]
    # The thread is pointed at by both the snapshot and the event.
    into_thread = {link.relation for link in reg.links_to(r["thread"].record_id)}
    assert into_thread == {"observed_in_run", "recorded_in"}


# ----------------------------------------------------------------- neighbors
def test_neighbors_direction_and_relation_filter() -> None:
    """neighbors respects OUT/IN/BOTH and an optional relation filter."""
    reg, r = _hub_graph()
    tid = r["thread"].record_id
    out = {link.relation for link in reg.neighbors(tid, direction=LineageDirection.OUT)}
    assert out == {"uses_persona", "in_thread", "built_from"}
    inbound = {
        link.relation for link in reg.neighbors(tid, direction=LineageDirection.IN)
    }
    assert inbound == {"observed_in_run", "recorded_in"}
    both = {
        link.relation for link in reg.neighbors(tid, direction=LineageDirection.BOTH)
    }
    assert both == out | inbound
    only = reg.neighbors(tid, direction=LineageDirection.BOTH, relation="uses_persona")
    assert len(only) == 1 and only[0].relation == "uses_persona"


# ----------------------------------------------------------------- trace
def test_trace_both_reaches_full_neighbourhood() -> None:
    """A BOTH trace from the hub reaches every connected record."""
    reg, r = _hub_graph()
    graph = reg.trace(r["thread"].record_id)
    assert isinstance(graph, LineageGraph)
    kinds = {rec.kind for rec in graph.nodes.values()}
    assert kinds == {
        "chat_thread",
        "persona",
        "prompt",
        "context_snapshot",
        "run_event",
    }
    assert graph.relations() == {
        "uses_persona",
        "in_thread",
        "built_from",
        "observed_in_run",
        "recorded_in",
    }
    assert graph.truncated is False


def test_trace_out_excludes_inbound_only_nodes() -> None:
    """An OUT trace does not reach a node connected only by an inbound edge."""
    reg, r = _hub_graph()
    graph = reg.trace(r["thread"].record_id, direction=LineageDirection.OUT)
    reached = graph.record_ids()
    assert r["persona"].record_id in reached
    assert r["prompt"].record_id in reached
    assert r["snapshot"].record_id in reached
    # event->thread is inbound only, so OUT never reaches the event.
    assert r["event"].record_id not in reached


def test_trace_relations_filter_restricts_walk() -> None:
    """Restricting relations prunes both the edges traversed and nodes reached."""
    reg, r = _hub_graph()
    graph = reg.trace(
        r["thread"].record_id,
        direction=LineageDirection.OUT,
        relations={"uses_persona"},
    )
    assert graph.record_ids() == {r["thread"].record_id, r["persona"].record_id}
    assert graph.relations() == {"uses_persona"}
    assert graph.edge_count == 1


def test_trace_max_depth_sets_truncated() -> None:
    """A depth cutoff stops the walk and flags truncation; a full walk does not."""
    reg = EntityRegistry()
    chain = [_rec(reg, "node", name) for name in ("a", "b", "c", "d")]
    for src, dst in zip(chain, chain[1:], strict=False):
        reg.link(
            from_record_id=src.record_id, to_record_id=dst.record_id, relation="next"
        )

    shallow = reg.trace(chain[0].record_id, direction=LineageDirection.OUT, max_depth=1)
    assert shallow.record_ids() == {chain[0].record_id, chain[1].record_id}
    assert shallow.truncated is True

    full = reg.trace(chain[0].record_id, direction=LineageDirection.OUT, max_depth=3)
    assert len(full.nodes) == 4
    assert full.truncated is False


def test_trace_unknown_root_is_empty_graph() -> None:
    """Tracing a record id that was never registered yields an empty graph."""
    reg, _ = _hub_graph()
    graph = reg.trace("does-not-exist")
    assert graph.root_id == "does-not-exist"
    assert graph.nodes == {}
    assert graph.edges == []
    assert graph.truncated is False


# ----------------------------------------------------------------- LineageGraph
def test_lineage_graph_to_dot_and_filter() -> None:
    """to_dot renders DOT with the root highlighted; filter_relations prunes edges."""
    reg, r = _hub_graph()
    graph = reg.trace(r["thread"].record_id)

    dot = graph.to_dot()
    assert dot.startswith("digraph lineage {")
    assert "uses_persona" in dot
    assert "lightyellow" in dot  # the root node is highlighted

    filtered = graph.filter_relations({"uses_persona", "in_thread"})
    assert filtered.relations() == {"uses_persona", "in_thread"}
    # The root is always retained; the snapshot/event drop out.
    assert r["thread"].record_id in filtered.nodes
    assert r["snapshot"].record_id not in filtered.nodes
    assert filtered.node_count == 3  # thread + persona + prompt


# ----------------------------------------------------- end-to-end over a real run
def _run_stack() -> tuple[RunAppService, EntityRegistry]:
    """A RunAppService wired to a registry-backed runtime (offline stub)."""
    storage = StorageService()
    registry = EntityRegistry()
    context = ContextService(storage_service=storage, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )
    run_app = RunAppService(runtime=runtime, storage=storage, entity_registry=registry)
    return run_app, registry


def test_get_run_lineage_traces_persona_and_prompt() -> None:
    """A completed run's lineage resolves back to its persona and prompt records."""
    run_app, _registry = _run_stack()
    persona = Persona(name="analyst", description="x")
    task = Task(title="t", prompt="Summarize ACME.")

    async def _scenario() -> Any:
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        await run_app.await_run(run.run_id, timeout=5.0)
        return (
            run,
            await run_app.get_run_lineage(run.run_id, workspace_id="w1"),
            await run_app.get_run_lineage(run.run_id, workspace_id="other"),
            await run_app.get_run_lineage("nonexistent", workspace_id="w1"),
        )

    run, graph, foreign, missing = run_async(_scenario())
    assert graph is not None
    assert graph.node_count >= 3  # thread + persona + prompt (snapshot when present)
    kinds = {rec.kind for rec in graph.nodes.values()}
    assert "chat_thread" in kinds
    assert "persona" in kinds
    assert "prompt" in kinds
    assert "uses_persona" in graph.relations()
    # Tenant isolation: a foreign workspace cannot read the lineage.
    assert foreign is None
    # Unknown run -> None (404 upstream).
    assert missing is None


def test_get_run_lineage_none_without_registry() -> None:
    """With no entity registry wired, lineage is unavailable (None, not an error)."""
    storage = StorageService()
    context = ContextService(storage_service=storage)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
    )
    run_app = RunAppService(runtime=runtime, storage=storage)

    async def _scenario() -> Any:
        run = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=Persona(name="a"),
            task=Task(title="t", prompt="p"),
        )
        await run_app.await_run(run.run_id, timeout=5.0)
        return await run_app.get_run_lineage(run.run_id, workspace_id="w1")

    assert run_async(_scenario()) is None
