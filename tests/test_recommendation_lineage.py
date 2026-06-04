"""Tests for recommendations as first-class lineage nodes.

Covers projecting an extracted recommendation into the entity graph, the
``derived_from`` (-> run thread hub) and ``cites`` (-> evidence) links, idempotent
re-projection, graceful degradation without a registry, and the end-to-end
"trace THIS recommendation back to its persona + evidence" path via
``RecommendationAppService.get_recommendation_lineage``.
"""

from __future__ import annotations

from typing import Any

from opensims.agents.base_agent.task import Task
from opensims.agents.personas.persona import Persona
from opensims.application.services import RecommendationAppService, RunAppService
from opensims.entities.records import EntityRecord, stable_id_for
from opensims.entities.registry import EntityRegistry
from opensims.runtime.single_agent import SingleAgentRuntime
from opensims.services.context.service import ContextService
from opensims.services.inference.client_manager import StubClientManager
from opensims.services.inference.service import InferenceService
from opensims.services.storage.models import RunRecord
from opensims.services.storage.service import StorageService
from tests.conftest import run_async


def _rec_record_id(registry: EntityRegistry, recommendation_id: str) -> str:
    """Resolve the registered recommendation record's id from its domain id."""
    sid = stable_id_for(recommendation_id, namespace="recommendation")
    record = registry.get_latest(sid)
    assert record is not None
    return record.record_id


def _seed_run_graph(registry: EntityRegistry) -> tuple[Any, Any, str]:
    """Simulate a completed run's captured graph: thread -> persona, plus evidence.

    Returns (thread_record, evidence_record, evidence_id). The thread's stable_id
    matches a run whose ``thread_id`` is ``"thread-1"``.
    """
    thread = registry.register(
        EntityRecord.create(
            stable_id=stable_id_for("thread-1", namespace="chat_thread"),
            version=1,
            kind="chat_thread",
            payload={},
        )
    )
    persona = registry.register(
        EntityRecord.create(
            stable_id="persona-1", version=1, kind="persona", payload={}
        )
    )
    registry.link(
        from_record_id=thread.record_id,
        to_record_id=persona.record_id,
        relation="uses_persona",
    )
    evidence_id = "ev-123"
    evidence = registry.register(
        EntityRecord.create(
            stable_id=stable_id_for(evidence_id, namespace="context_evidence"),
            version=1,
            kind="context_evidence",
            payload={"source_type": "doc"},
        )
    )
    return thread, evidence, evidence_id


def _run_with_recs(evidence_id: str, *, extra_ref: str | None = None) -> RunRecord:
    """A RunRecord whose structured output is a one-item recommendation envelope."""
    refs = [evidence_id] + ([extra_ref] if extra_ref else [])
    return RunRecord(
        workspace_id="w1",
        subject_id="s1",
        thread_id="thread-1",
        output_structured={
            "recommendations": [
                {
                    "kind": "buy",
                    "title": "Accumulate ACME",
                    "summary": "strong widget demand",
                    "confidence": 0.7,
                    "evidence_refs": refs,
                }
            ]
        },
    )


# --------------------------------------------------- projection: derived_from + cites
def test_extract_projects_recommendation_node_and_links() -> None:
    """Extraction registers a recommendation entity and links provenance edges."""
    storage = StorageService()
    registry = EntityRegistry()
    thread, evidence, evidence_id = _seed_run_graph(registry)
    rec_app = RecommendationAppService(storage=storage, entity_registry=registry)

    items = run_async(rec_app.extract_from_run(_run_with_recs(evidence_id)))
    assert len(items) == 1
    assert len(registry.list_by_kind("recommendation")) == 1

    rec_rid = _rec_record_id(registry, items[0].recommendation_id)
    links = {
        (link.relation, link.to_record_id) for link in registry.links_from(rec_rid)
    }
    assert ("derived_from", thread.record_id) in links
    assert ("cites", evidence.record_id) in links


def test_cites_skips_unresolved_evidence_but_keeps_real_ones() -> None:
    """A citation to evidence that is not a registered node is not graphed."""
    storage = StorageService()
    registry = EntityRegistry()
    _thread, evidence, evidence_id = _seed_run_graph(registry)
    rec_app = RecommendationAppService(storage=storage, entity_registry=registry)

    items = run_async(
        rec_app.extract_from_run(_run_with_recs(evidence_id, extra_ref="ghost-ref"))
    )
    rec_rid = _rec_record_id(registry, items[0].recommendation_id)
    cites = [
        link.to_record_id
        for link in registry.links_from(rec_rid)
        if link.relation == "cites"
    ]
    # Only the real evidence is graphed; the dangling ref is dropped from the graph
    # (it remains visible in the recommendation's evidence_refs payload).
    assert cites == [evidence.record_id]
    assert "ghost-ref" in items[0].evidence_refs


def test_projection_is_idempotent() -> None:
    """Re-projecting the same item neither duplicates the node nor its links."""
    storage = StorageService()
    registry = EntityRegistry()
    _thread, _evidence, evidence_id = _seed_run_graph(registry)
    rec_app = RecommendationAppService(storage=storage, entity_registry=registry)

    item = run_async(rec_app.extract_from_run(_run_with_recs(evidence_id)))[0]
    run = _run_with_recs(evidence_id)
    # Project the same item a second time directly.
    run_async(rec_app._project_lineage(item, run))

    assert len(registry.list_by_kind("recommendation")) == 1
    rec_rid = _rec_record_id(registry, item.recommendation_id)
    relations = [link.relation for link in registry.links_from(rec_rid)]
    assert relations.count("derived_from") == 1
    assert relations.count("cites") == 1


# ----------------------------------------------------------- trace from the rec node
def test_trace_from_recommendation_reaches_persona_and_evidence() -> None:
    """A BOTH trace from the recommendation reaches the run hub, persona, evidence."""
    storage = StorageService()
    registry = EntityRegistry()
    thread, evidence, evidence_id = _seed_run_graph(registry)
    rec_app = RecommendationAppService(storage=storage, entity_registry=registry)
    item = run_async(rec_app.extract_from_run(_run_with_recs(evidence_id)))[0]

    rec_rid = _rec_record_id(registry, item.recommendation_id)
    graph = registry.trace(rec_rid)
    kinds = {rec.kind for rec in graph.nodes.values()}
    assert {"recommendation", "chat_thread", "persona", "context_evidence"} <= kinds
    assert {"derived_from", "cites", "uses_persona"} <= graph.relations()
    assert thread.record_id in graph.record_ids()
    assert evidence.record_id in graph.record_ids()


# ----------------------------------------------- end-to-end over a real run + scoping
def _full_stack() -> tuple[RunAppService, RecommendationAppService, EntityRegistry]:
    storage = StorageService()
    registry = EntityRegistry()
    context = ContextService(storage_service=storage, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )
    rec_app = RecommendationAppService(storage=storage, entity_registry=registry)
    run_app = RunAppService(
        runtime=runtime,
        storage=storage,
        entity_registry=registry,
        recommendation_app=rec_app,
    )
    return run_app, rec_app, registry


def test_get_recommendation_lineage_traces_back_to_run_and_persona() -> None:
    """The README demo: trace a recommendation back to its run's persona."""
    run_app, rec_app, _registry = _full_stack()

    async def _scenario() -> Any:
        run = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=Persona(name="Analyst", description="careful"),
            task=Task(title="t", prompt="Summarize ACME."),
        )
        await run_app.await_run(run.run_id, timeout=5.0)
        # The stub does not emit a RecommendationEnvelope, so graph one from the
        # SAME run — its real thread_id wires derived_from to the run's hub.
        run = await run_app.get_run(run.run_id, workspace_id="w1")
        run.output_structured = {
            "recommendations": [
                {"kind": "buy", "title": "Accumulate", "confidence": 0.6}
            ]
        }
        items = await rec_app.extract_from_run(run)
        rec_id = items[0].recommendation_id
        return (
            await rec_app.get_recommendation_lineage(rec_id, workspace_id="w1"),
            await rec_app.get_recommendation_lineage(rec_id, workspace_id="other"),
            await rec_app.get_recommendation_lineage("missing", workspace_id="w1"),
        )

    graph, foreign, missing = run_async(_scenario())
    assert graph is not None
    kinds = {rec.kind for rec in graph.nodes.values()}
    assert {"recommendation", "chat_thread", "persona"} <= kinds
    assert "derived_from" in graph.relations()
    assert "uses_persona" in graph.relations()
    # Tenant isolation + unknown id both yield None (404 upstream).
    assert foreign is None
    assert missing is None


def test_extract_without_registry_still_works_and_lineage_is_none() -> None:
    """No registry: extraction still persists items; lineage is unavailable (None)."""
    storage = StorageService()
    rec_app = RecommendationAppService(storage=storage)
    run = RunRecord(
        workspace_id="w1",
        subject_id="s1",
        output_structured={"recommendations": [{"kind": "k", "title": "t"}]},
    )

    async def _scenario() -> Any:
        items = await rec_app.extract_from_run(run)
        lineage = await rec_app.get_recommendation_lineage(
            items[0].recommendation_id, workspace_id="w1"
        )
        return items, lineage

    items, lineage = run_async(_scenario())
    assert len(items) == 1
    assert lineage is None
