"""Expanded application-kernel coverage locking in the hardened AAEO behaviours.

Complements ``test_application_hardening.py`` with paths it does not exercise:

- The FAILED-run contract driven through the *runtime* ``run_task_detailed``
  (RunResult) rather than only via a failing client manager (AAEO-3).
- Schema-validation ``extraction_error`` recorded by the full ``_execute_run``
  path against a real ``output_json_schema`` mismatch (AAEO-6), plus direct
  ``_requested_schema`` / ``_validate_structured`` unit coverage.
- Snapshot + recommendation workspace isolation on reads/transitions (AAEO-4).
- Recommendation list pagination order/cap + count (AAEO-8).
- The dashboard folding evaluation scores (AAEO-15).

All tests are plain ``def test_*`` driving async via the ``run_async`` helper;
everything runs on the offline in-memory stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.application.services import (
    MAX_PAGE_LIMIT,
    ContextAppService,
    DashboardQueryService,
    RecommendationAppService,
    RunAppService,
    _requested_schema,
    _validate_structured,
)
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.models import ContextBuildSpec, ContextSpecKey
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


# --------------------------------------------------------------------- helpers
@dataclass
class _FakeThread:
    """Minimal stand-in for a ChatThread the app layer reads off a RunResult."""

    thread_id: str = "thr-1"


class _FakeRuntime:
    """A runtime whose ``run_task_detailed`` returns a pre-baked RunResult.

    Lets us assert the application layer honours the *typed* RunResult contract
    (AAEO-3) independent of how the underlying inference produced it.
    """

    def __init__(self, result: Any) -> None:
        self._result = result

    async def run_task_detailed(self, persona, task, *, llm_config=None):  # noqa: ANN001
        return self._result


def _run_app_over(runtime: Any, storage: StorageService) -> RunAppService:
    return RunAppService(runtime=runtime, storage=storage)


def _real_stack(manager=None):
    storage = StorageService()
    registry = EntityRegistry()
    context = ContextService(storage_service=storage, entity_registry=registry)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(
            manager or StubClientManager(), event_sink=storage
        ),
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
    return storage, run_app, rec_app


# ------------------------------------------------------------------- AAEO-3
def test_failed_runresult_from_runtime_marks_run_failed() -> None:
    """A FAILED RunResult (status != SUCCESS) yields a FAILED run + error, no recs."""
    from himmy.services.inference.models import InferenceStatus

    result = type(
        "RR",
        (),
        {
            "thread": _FakeThread(),
            "trace_id": "trace-1",
            "succeeded": False,
            "status": InferenceStatus.FAILED.value,
            "error": "model refused",
            "error_code": "INVALID_REQUEST",
            "output_text": None,
            "output_structured": None,
        },
    )()
    storage = StorageService()
    run_app = _run_app_over(_FakeRuntime(result), storage)
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        return await run_app.await_run(run.run_id, timeout=5.0)

    terminal = run_async(_scenario())
    assert terminal is not None
    assert terminal.status == RunStatus.FAILED
    assert terminal.error == "model refused"
    # The trace id is still stamped so the failed run is traceable.
    assert terminal.trace_id == "trace-1"


def test_failed_run_falls_back_to_error_code_when_no_message() -> None:
    """When the failed result has only an error_code, that becomes run.error."""
    from himmy.services.inference.models import InferenceStatus

    result = type(
        "RR",
        (),
        {
            "thread": _FakeThread(),
            "trace_id": None,
            "succeeded": False,
            "status": InferenceStatus.FAILED.value,
            "error": None,
            "error_code": "PROVIDER_UNAVAILABLE",
            "output_text": None,
            "output_structured": None,
        },
    )()
    storage = StorageService()
    run_app = _run_app_over(_FakeRuntime(result), storage)
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        return await run_app.await_run(run.run_id, timeout=5.0)

    terminal = run_async(_scenario())
    assert terminal is not None
    assert terminal.status == RunStatus.FAILED
    assert terminal.error == "PROVIDER_UNAVAILABLE"


# ------------------------------------------------------------------- AAEO-6
def test_execute_run_records_schema_validation_error() -> None:
    """A SUCCEEDED run whose structured output violates the schema records an error.

    The structured payload omits a required property, so ``_execute_run`` records
    ``extraction_error`` on the run while still marking it SUCCEEDED (a schema
    mismatch is observability signal, not a hard run failure).
    """
    from himmy.services.inference.models import InferenceStatus

    schema = {
        "type": "object",
        "properties": {"verdict": {"type": "string"}},
        "required": ["verdict"],
    }
    result = type(
        "RR",
        (),
        {
            "thread": _FakeThread(),
            "trace_id": "trace-2",
            "succeeded": True,
            "status": InferenceStatus.SUCCESS.value,
            "error": None,
            "error_code": None,
            "output_text": "{}",
            # Missing the required 'verdict' key.
            "output_structured": {"something_else": 1},
        },
    )()
    storage = StorageService()
    run_app = _run_app_over(_FakeRuntime(result), storage)
    persona = Persona(name="A")
    task = Task(title="t", prompt="advise", context={"output_schema": schema})
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
        return await run_app.await_run(run.run_id, timeout=5.0)

    terminal = run_async(_scenario())
    assert terminal is not None
    assert terminal.status == RunStatus.SUCCEEDED
    assert "extraction_error" in (terminal.metadata or {})
    assert "schema validation failed" in terminal.metadata["extraction_error"]


def test_requested_schema_precedence() -> None:
    """_requested_schema prefers llm_config schema, else task.context['output_schema']."""
    schema_cfg = {"type": "object"}
    schema_ctx = {"type": "array"}
    task_with_ctx = Task(title="t", prompt="p", context={"output_schema": schema_ctx})
    # llm_config wins when it carries a schema.
    cfg = LLMConfig(
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema=schema_cfg,
    )
    assert _requested_schema(cfg, task_with_ctx) == schema_cfg
    # Falls back to the task context schema when llm_config has none.
    assert _requested_schema(None, task_with_ctx) == schema_ctx
    # No schema anywhere -> None.
    assert _requested_schema(None, Task(title="t", prompt="p")) is None
    # A non-dict context schema is ignored.
    bad = Task(title="t", prompt="p", context={"output_schema": "nope"})
    assert _requested_schema(None, bad) is None


def test_validate_structured_passes_and_fails() -> None:
    """_validate_structured returns None on a valid instance and an error otherwise."""
    schema = {
        "type": "object",
        "properties": {"n": {"type": "integer"}},
        "required": ["n"],
    }
    assert _validate_structured({"n": 5}, schema) is None
    err = _validate_structured({}, schema)
    assert err is not None


# ------------------------------------------------------------------- AAEO-4
def test_snapshot_workspace_isolation() -> None:
    """get_snapshot rejects a foreign workspace AND fails closed on unstamped (scope-r1).

    ``context_snapshots`` is keyed by snapshot_id alone (no workspace column), so the
    in-Python ``_snapshot_in_workspace`` check is the SOLE tenant boundary. It now FAILS
    CLOSED: a tenant-bound caller (one that supplies a ``workspace_id``) may read a snapshot
    only when it carries a MATCHING workspace stamp — an unstamped snapshot (built by the
    offline/CLI/all-tenants/routine path) is NOT readable cross-tenant by guessing its id.
    The offline path (``workspace_id=None``) keeps reading any snapshot, byte-unchanged.
    """
    storage = StorageService()
    ctx = ContextService(storage_service=storage)
    app = ContextAppService(context_service=ctx, storage=storage)
    spec = ContextBuildSpec(keys=[ContextSpecKey(key="company", required=False)])

    async def _scenario():
        stamped = await app.build_snapshot(
            subject_id="s1", build_spec=spec, metadata={"workspace_id": "w1"}
        )
        same = await app.get_snapshot(stamped.snapshot_id, workspace_id="w1")
        foreign = await app.get_snapshot(stamped.snapshot_id, workspace_id="w2")
        # An unstamped (offline/CLI-built) snapshot is NOT readable by a tenant-bound caller
        # (fail closed) — it belongs to no workspace, so a tenant cannot claim it by id.
        legacy = await app.build_snapshot(subject_id="s2", build_spec=spec)
        legacy_tenant_bound = await app.get_snapshot(
            legacy.snapshot_id, workspace_id="w9"
        )
        # ...but the offline path (no workspace supplied) still reads it — byte-unchanged.
        legacy_offline = await app.get_snapshot(legacy.snapshot_id, workspace_id=None)
        missing = await app.get_snapshot("nope", workspace_id="w1")
        return same, foreign, legacy_tenant_bound, legacy_offline, missing

    same, foreign, legacy_tenant_bound, legacy_offline, missing = run_async(
        _scenario()
    )
    assert same is not None
    assert foreign is None
    assert legacy_tenant_bound is None  # fail closed: unstamped not readable cross-tenant
    assert legacy_offline is not None  # offline path unchanged
    assert missing is None


def test_recommendation_update_workspace_isolation() -> None:
    """update_status rejects a recommendation from another workspace (returns None)."""
    storage = StorageService()
    rec_app = RecommendationAppService(storage=storage)
    run = RunRecord(
        workspace_id="w1",
        subject_id="s1",
        output_structured={"recommendations": [{"kind": "trade", "title": "Buy"}]},
    )

    async def _scenario():
        items = await rec_app.extract_from_run(run)
        rid = items[0].recommendation_id
        # Foreign workspace cannot mutate.
        foreign = await rec_app.update_status(
            rid, status=RecommendationStatus.ACCEPTED, workspace_id="w2"
        )
        # Correct workspace succeeds.
        ok = await rec_app.update_status(
            rid, status=RecommendationStatus.ACCEPTED, workspace_id="w1"
        )
        return foreign, ok

    foreign, ok = run_async(_scenario())
    assert foreign is None
    assert ok is not None
    assert ok.status == RecommendationStatus.ACCEPTED


# ------------------------------------------------------------------- AAEO-8
def test_recommendation_list_pagination_and_count() -> None:
    """Recommendation list is created_at-desc ordered, windowed, and capped."""
    storage = StorageService()
    rec_app = RecommendationAppService(storage=storage)

    async def _scenario():
        for i in range(5):
            run = RunRecord(
                workspace_id="w1",
                subject_id="s1",
                output_structured={
                    "recommendations": [{"kind": "trade", "title": f"r{i}"}]
                },
            )
            await rec_app.extract_from_run(run)
        page1 = await rec_app.list(workspace_id="w1", limit=2, offset=0)
        page2 = await rec_app.list(workspace_id="w1", limit=2, offset=2)
        total = await rec_app.count(workspace_id="w1")
        # An absurd limit is clamped (no rows beyond what exists, no error).
        capped = await rec_app.list(workspace_id="w1", limit=MAX_PAGE_LIMIT + 5000)
        return page1, page2, total, capped

    page1, page2, total, capped = run_async(_scenario())
    assert total == 5
    assert len(page1) == 2 and len(page2) == 2
    # Newest-first ordering with no overlap between pages.
    ids1 = {r.recommendation_id for r in page1}
    ids2 = {r.recommendation_id for r in page2}
    assert ids1.isdisjoint(ids2)
    assert len(capped) == 5


# ------------------------------------------------------------------- AAEO-15
def test_dashboard_folds_evaluation_scores() -> None:
    """The dashboard summary folds in the latest evaluation aggregate (AAEO-15)."""
    from himmy.services.evaluation.models import (
        EvaluationCase,
        EvaluationSuite,
    )
    from himmy.services.evaluation.service import EvaluationService

    storage = StorageService()
    dashboard = DashboardQueryService(storage=storage)
    eval_svc = EvaluationService(storage_service=storage)

    async def _scenario():
        case = EvaluationCase(
            expected_output={"answer": "yes"}, metric_weights={"accuracy": 1.0}
        )
        suite = EvaluationSuite(name="qa", cases=[case])
        await eval_svc.run_suite(
            suite=suite, actual_outputs={case.case_id: {"answer": "yes"}}
        )
        # The offline / single-tenant path (no tenant-bound principal) keeps the lenient
        # eval-tile filter, so the unstamped run scored above stays visible — the
        # byte-unchanged zero-config behaviour. (A tenant-bound caller passes
        # all_tenants=False and would strict-scope it out; see scope-r3.)
        return await dashboard.summary(
            subject_id="s1", workspace_id="w1", all_tenants=True
        )

    summary = run_async(_scenario())
    assert summary["evaluation"]["total"] == 1
    assert summary["evaluation"]["latest_aggregate"] == 1.0
    assert summary["evaluation"]["latest_suite_name"] == "qa"


def test_dashboard_evaluation_empty_when_no_runs() -> None:
    """The dashboard reports a zeroed evaluation tile when no eval runs exist."""
    storage = StorageService()
    dashboard = DashboardQueryService(storage=storage)
    summary = run_async(dashboard.summary(subject_id="s1", workspace_id="w1"))
    assert summary["evaluation"]["total"] == 0
    assert summary["evaluation"]["latest_aggregate"] is None
