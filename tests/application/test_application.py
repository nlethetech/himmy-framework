"""Tests for the application kernel: run lifecycle, idempotency, recommendations."""

from __future__ import annotations

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.application import (
    ContextAppService,
    DashboardQueryService,
    Recommendation,
    RecommendationAppService,
    RecommendationEnvelope,
    RunAppService,
)
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import LLMConfig, ResponseFormat
from himmy.services.inference.service import InferenceService
from himmy.services.storage.models import (
    RecommendationStatus,
    RunRecord,
    RunStatus,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _stack():
    storage = StorageService()
    registry = EntityRegistry()
    context = ContextService(storage_service=storage, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
    )
    rec_app = RecommendationAppService(storage=storage)
    run_app = RunAppService(
        runtime=runtime,
        storage=storage,
        entity_registry=registry,
        recommendation_app=rec_app,
    )
    return storage, registry, runtime, run_app, rec_app


def test_create_run_returns_queued_then_succeeds() -> None:
    """create_run returns immediately (QUEUED); await_run reaches SUCCEEDED."""
    storage, _registry, _rt, run_app, _rec = _stack()
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        assert run.status == RunStatus.QUEUED
        terminal = await run_app.await_run(run.run_id, timeout=5.0)
        return terminal

    terminal = run_async(_scenario())
    assert terminal is not None
    assert terminal.status == RunStatus.SUCCEEDED
    assert terminal.output_text
    assert terminal.thread_id is not None


def test_create_run_is_idempotent() -> None:
    """Re-submitting an idempotency key returns the same run, not a duplicate."""
    storage, _registry, _rt, run_app, _rec = _stack()
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        first = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=persona,
            task=task,
            idempotency_key="key-1",
        )
        second = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=persona,
            task=task,
            idempotency_key="key-1",
        )
        return first, second

    first, second = run_async(_scenario())
    assert first.run_id == second.run_id
    all_runs = run_async(storage.list_runs(workspace_id="w1"))
    assert len(all_runs) == 1


def test_recommendation_extraction_from_structured_run() -> None:
    """A run whose output matches the envelope auto-extracts recommendation items.

    The schema is hand-built (no ``default`` on the array) so the offline
    synthesizer populates at least one recommendation, mirroring a real model.
    """
    storage, _registry, _rt, run_app, rec_app = _stack()
    persona = Persona(name="A")
    schema = {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["kind", "title"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["recommendations"],
    }
    task = Task(title="t", prompt="advise on ACME")
    cfg = LLMConfig(
        response_format=ResponseFormat.STRUCTURED_OUTPUT, output_json_schema=schema
    )

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1",
            subject_id="s1",
            persona=persona,
            task=task,
            llm_config=cfg,
        )
        await run_app.await_run(run.run_id, timeout=5.0)
        return await rec_app.list(workspace_id="w1")

    items = run_async(_scenario())
    # The synthesized envelope has at least one recommendation (array minItems=1).
    assert len(items) >= 1
    assert items[0].status == RecommendationStatus.PROPOSED


def test_recommendation_extract_from_run_direct() -> None:
    """extract_from_run parses a RecommendationEnvelope dict into items."""
    storage = StorageService()
    rec_app = RecommendationAppService(storage=storage)
    envelope = RecommendationEnvelope(
        recommendations=[
            Recommendation(kind="trade", title="Buy ACME", confidence=0.8)
        ],
        summary="bullish",
    )
    run = RunRecord(
        workspace_id="w1",
        subject_id="s1",
        output_structured=envelope.model_dump(),
    )

    items = run_async(rec_app.extract_from_run(run))
    assert len(items) == 1
    assert items[0].title == "Buy ACME"
    assert items[0].confidence == 0.8
    # Non-envelope output yields nothing.
    run2 = RunRecord(workspace_id="w1", subject_id="s1", output_structured={"x": 1})
    assert run_async(rec_app.extract_from_run(run2)) == []


def test_recommendation_status_transition() -> None:
    """update_status transitions a stored recommendation."""
    storage = StorageService()
    rec_app = RecommendationAppService(storage=storage)
    run = RunRecord(
        workspace_id="w1",
        subject_id="s1",
        output_structured={"recommendations": [{"kind": "trade", "title": "Buy"}]},
    )
    items = run_async(rec_app.extract_from_run(run))
    rid = items[0].recommendation_id
    updated = run_async(
        rec_app.update_status(rid, status=RecommendationStatus.ACCEPTED, notes="ok")
    )
    assert updated is not None
    assert updated.status == RecommendationStatus.ACCEPTED
    assert updated.notes == "ok"


def test_dashboard_summary_aggregates_counts() -> None:
    """The dashboard summary reports context, run, and recommendation counts."""
    storage, _registry, _rt, run_app, _rec = _stack()
    dashboard = DashboardQueryService(storage=storage)
    context_service = ContextService(storage_service=storage)
    context_app = ContextAppService(context_service=context_service, storage=storage)
    from himmy.services.context.models import ContextField

    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        await context_app.upsert_fields(
            "w1", "s1", [ContextField(key="k", value="v", confidence=0.5)]
        )
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        await run_app.await_run(run.run_id, timeout=5.0)
        return await dashboard.summary(subject_id="s1", workspace_id="w1")

    summary = run_async(_scenario())
    assert summary["context"]["field_count"] == 1
    assert summary["context"]["avg_confidence"] == 0.5
    assert summary["runs"]["total"] == 1
    assert summary["runs"]["by_status"][RunStatus.SUCCEEDED.value] == 1
    assert "recommendations" in summary


def test_dashboard_eval_summary_excludes_none_stamped_run() -> None:
    """scope-r3: a tenant's eval tile must NOT fold a None-stamped (admin/offline) run.

    ``_evaluation_summary`` listed ALL eval runs and re-filtered with the LENIENT
    ``in (None, workspace_id)`` predicate, so an offline/admin run with ``workspace_id``
    NULL leaked into a tenant's dashboard tile (aggregate_score, suite_name, latest_run_id)
    — a cross-tenant metadata disclosure. The filter is now STRICT ``== workspace_id``
    (matching the context-field tile and the GET-by-id IDOR path): a tenant sees only its
    own runs; an all_tenants (``workspace_id=None``) caller still sees everything.
    """
    from himmy.services.evaluation.models import EvaluationRun

    storage, _registry, _rt, _run_app, _rec = _stack()
    dashboard = DashboardQueryService(storage=storage)

    async def _scenario():
        # A None-stamped run (admin all-tenants / offline) and a tenant-w1 run.
        await storage.save_evaluation_run(
            EvaluationRun(
                suite_id="admin",
                suite_name="admin-suite",
                workspace_id=None,
                aggregate_score=0.99,
                created_at="2999-01-01T00:00:00Z",  # newest, so it would win `max`
            )
        )
        await storage.save_evaluation_run(
            EvaluationRun(
                suite_id="w1",
                suite_name="w1-suite",
                workspace_id="w1",
                aggregate_score=0.10,
                created_at="2000-01-01T00:00:00Z",
            )
        )
        tenant = await dashboard.summary(subject_id="s1", workspace_id="w1")
        admin = await dashboard.summary(subject_id="s1", workspace_id=None)
        # Offline / all_tenants caller explicitly scoping to w1 keeps the None-stamped
        # run visible — byte-unchanged zero-config (the lenient filter is retained).
        offline = await dashboard.summary(
            subject_id="s1", workspace_id="w1", all_tenants=True
        )
        return tenant, admin, offline

    tenant, admin, offline = run_async(_scenario())
    # Tenant w1 sees ONLY its own run — not the newer None-stamped admin run.
    assert tenant["evaluation"]["total"] == 1
    assert tenant["evaluation"]["latest_suite_name"] == "w1-suite"
    assert tenant["evaluation"]["latest_aggregate"] == 0.10
    # The all_tenants view (workspace=None) still folds every run.
    assert admin["evaluation"]["total"] == 2
    assert admin["evaluation"]["latest_suite_name"] == "admin-suite"
    # Offline invariant: all_tenants + explicit w1 keeps BOTH (the None-stamped run too).
    assert offline["evaluation"]["total"] == 2
    assert offline["evaluation"]["latest_suite_name"] == "admin-suite"


def test_dashboard_context_tile_excludes_none_stamped_field_for_tenant() -> None:
    """scope-r4: a tenant's context tile must NOT fold a None-stamped (admin/offline) field.

    ``DashboardQueryService.summary`` re-implemented the context-field filter INLINE with the
    LENIENT ``in (None, workspace_id)`` predicate, UNCONDITIONALLY — so a context field
    written for a SHARED ``subject_id`` by an offline/admin path with a blank/None
    ``workspace_id`` was folded into a tenant-bound caller's context tile (field_count,
    avg_confidence, avg_freshness), diverging from the strict reader
    ``ContextAppService.list_fields`` and the eval tile. The filter now gates on
    ``all_tenants`` exactly like the eval tile: STRICT ``== workspace_id`` for a tenant-bound
    caller (the None-stamped field is excluded), lenient only for the all_tenants/offline
    path (byte-unchanged).
    """
    from himmy.services.context.models import ContextField

    storage, _registry, _rt, _run_app, _rec = _stack()
    dashboard = DashboardQueryService(storage=storage)
    context_service = ContextService(storage_service=storage)
    context_app = ContextAppService(context_service=context_service, storage=storage)

    async def _scenario():
        # A field properly stamped for tenant w1.
        await context_app.upsert_fields(
            "w1", "shared-sub", [ContextField(key="own", value="v", confidence=0.4)]
        )
        # A legacy None-workspace-stamped field for the SAME subject (offline / admin / a
        # DIFFERENT tenant's unstamped write).
        await storage.save_context_field(
            ContextField(
                key="leak",
                value="x",
                confidence=1.0,
                metadata={"subject_id": "shared-sub", "workspace_id": None},
            )
        )
        tenant = await dashboard.summary(subject_id="shared-sub", workspace_id="w1")
        offline = await dashboard.summary(
            subject_id="shared-sub", workspace_id="w1", all_tenants=True
        )
        # The strict reader must agree with the tenant tile (one scoping implementation).
        strict_fields = await context_app.list_fields(
            "shared-sub", workspace_id="w1"
        )
        return tenant, offline, strict_fields

    tenant, offline, strict_fields = run_async(_scenario())
    # Tenant w1 sees ONLY its own field — the None-stamped one (conf 1.0) is excluded, so
    # the avg_confidence is its own 0.4, not inflated toward 0.7.
    assert tenant["context"]["field_count"] == 1
    assert tenant["context"]["avg_confidence"] == 0.4
    # It agrees with the strict dedicated reader (no divergence).
    assert len(strict_fields) == 1
    # Offline invariant: all_tenants keeps the lenient filter — the None-stamped field is
    # folded back in (byte-unchanged zero-config).
    assert offline["context"]["field_count"] == 2


def test_get_run_events_and_thread() -> None:
    """A completed run exposes its event stream and conversation thread."""
    storage, _registry, _rt, run_app, _rec = _stack()
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        await run_app.await_run(run.run_id, timeout=5.0)
        events = await run_app.get_run_events(run.run_id)
        thread = await run_app.get_run_thread(run.run_id)
        return events, thread

    events, thread = run_async(_scenario())
    assert events
    assert thread is not None
    assert thread.last_message is not None
