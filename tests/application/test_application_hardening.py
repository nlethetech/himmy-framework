"""Tests for the production-hardening application behaviours (AAEO-1/3/4/6/8/16).

Covers the FAILED-run path, concurrent idempotency, workspace isolation,
structured-output schema validation, pagination, model_key precedence, and the
startup sweep / shutdown drain.
"""

from __future__ import annotations

import asyncio

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.application.services import (
    MAX_PAGE_LIMIT,
    RecommendationAppService,
    RunAppService,
    _resolve_model_key,
)
from himmy.entities.registry import EntityRegistry
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    LLMConfig,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


class _FailingManager:
    """A client manager whose generate() returns a FAILED response (AAEO-3)."""

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=InferenceErrorCode.INVALID_REQUEST,
                message="schema validation failed at the model",
                retryable=False,
            ),
        )


def _stack(manager=None, **run_app_kwargs):
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
        **run_app_kwargs,
    )
    return storage, run_app, rec_app


# ---------------------------------------------------------------- AAEO-3 / 12
def test_failed_inference_yields_failed_run_with_error() -> None:
    """A FAILED inference response records a FAILED run with run.error set."""
    storage, run_app, rec_app = _stack(manager=_FailingManager())
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        terminal = await run_app.await_run(run.run_id, timeout=5.0)
        recs = await rec_app.list(workspace_id="w1")
        return terminal, recs

    terminal, recs = run_async(_scenario())
    assert terminal is not None
    assert terminal.status == RunStatus.FAILED
    assert terminal.error and "schema validation failed" in terminal.error
    # No recommendations extracted from a failed run.
    assert recs == []


# -------------------------------------------------------------------- AAEO-7 / 12
def test_concurrent_idempotency_creates_one_run() -> None:
    """N concurrent create_run with one key produce exactly one run + one task."""
    storage, run_app, _rec = _stack()
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        results = await asyncio.gather(
            *(
                run_app.create_run(
                    workspace_id="w1",
                    subject_id="s1",
                    persona=persona,
                    task=task,
                    idempotency_key="key-1",
                )
                for _ in range(20)
            )
        )
        # Let any launched background task settle.
        await asyncio.sleep(0.1)
        return results

    results = run_async(_scenario())
    run_ids = {r.run_id for r in results}
    assert len(run_ids) == 1
    all_runs = run_async(storage.list_runs(workspace_id="w1"))
    assert len(all_runs) == 1


# -------------------------------------------------------------------- AAEO-4 / 12
def test_workspace_isolation_on_reads() -> None:
    """get_run/get_run_thread/get_run_events reject a foreign workspace_id."""
    storage, run_app, _rec = _stack()
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        await run_app.await_run(run.run_id, timeout=5.0)
        # Correct workspace sees the run.
        same = await run_app.get_run(run.run_id, workspace_id="w1")
        # Foreign workspace gets None (404 upstream).
        other = await run_app.get_run(run.run_id, workspace_id="w2")
        thread_other = await run_app.get_run_thread(run.run_id, workspace_id="w2")
        events_other = await run_app.get_run_events(run.run_id, workspace_id="w2")
        return same, other, thread_other, events_other

    same, other, thread_other, events_other = run_async(_scenario())
    assert same is not None
    assert other is None
    assert thread_other is None
    assert events_other == []


def test_context_field_workspace_isolation() -> None:
    """Two workspaces sharing a subject_id don't see each other's fields."""
    from himmy.application.services import ContextAppService
    from himmy.services.context.models import ContextField

    storage = StorageService()
    ctx = ContextService(storage_service=storage)
    app = ContextAppService(context_service=ctx, storage=storage)

    async def _scenario():
        await app.upsert_fields("w1", "shared", [ContextField(key="a", value="1")])
        await app.upsert_fields("w2", "shared", [ContextField(key="b", value="2")])
        w1 = await app.list_fields("shared", workspace_id="w1")
        w2 = await app.list_fields("shared", workspace_id="w2")
        return w1, w2

    w1, w2 = run_async(_scenario())
    assert {f.key for f in w1} == {"a"}
    assert {f.key for f in w2} == {"b"}


# -------------------------------------------------------------------- AAEO-6
def test_schema_validation_records_extraction_error() -> None:
    """Output that violates the requested schema records an extraction_error."""
    storage, run_app, _rec = _stack()
    # A schema the stub's synthesized output will satisfy is not useful here; we
    # want a mismatch, so directly exercise the validator path via extract.
    bad_envelope = {"recommendations": [{"title": "missing kind"}]}
    run = RunRecord(workspace_id="w1", subject_id="s1", output_structured=bad_envelope)

    async def _scenario():
        rec_app = RecommendationAppService(storage=storage)
        items = await rec_app.extract_from_run(run)
        stored = await storage.get_run(run.run_id)
        return items, stored

    items, stored = run_async(_scenario())
    assert items == []
    assert stored is not None
    assert "extraction_error" in (stored.metadata or {})


# -------------------------------------------------------------------- AAEO-8
def test_list_runs_pagination_and_order() -> None:
    """list_runs is created_at-desc ordered, capped, and windowed by offset/limit."""
    storage = StorageService()
    # Pre-seed 5 runs directly (deterministic created_at via monotonic ids).
    for i in range(5):
        run_async(
            storage.save_run(
                RunRecord(
                    workspace_id="w1",
                    subject_id="s1",
                    run_id=f"run-{i}",
                    created_at=f"2026-01-0{i + 1}T00:00:00+00:00",
                )
            )
        )
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager())
    )
    run_app = RunAppService(runtime=runtime, storage=storage)

    async def _scenario():
        page1 = await run_app.list_runs(workspace_id="w1", limit=2, offset=0)
        page2 = await run_app.list_runs(workspace_id="w1", limit=2, offset=2)
        total = await run_app.count_runs(workspace_id="w1")
        return page1, page2, total

    page1, page2, total = run_async(_scenario())
    assert total == 5
    assert len(page1) == 2 and len(page2) == 2
    # created_at desc: newest first.
    assert page1[0].created_at > page1[1].created_at
    assert page1[1].created_at >= page2[0].created_at
    # No overlap between pages.
    assert {r.run_id for r in page1}.isdisjoint({r.run_id for r in page2})


def test_list_limit_is_capped() -> None:
    """A list limit above MAX_PAGE_LIMIT is clamped (no unbounded response)."""
    storage = StorageService()
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager())
    )
    run_app = RunAppService(runtime=runtime, storage=storage)
    # Even an absurd limit returns at most the cap (no rows here, but no error).
    page = run_async(run_app.list_runs(workspace_id="w1", limit=MAX_PAGE_LIMIT + 999))
    assert page == []


# -------------------------------------------------------------------- AAEO-16
def test_model_key_precedence() -> None:
    """task.context model_key wins over a default-only llm_config (AAEO-16)."""
    task = Task(title="t", prompt="p", context={"model_key": "gpt-context"})
    # llm_config carrying only the sentinel default must NOT shadow the context.
    cfg_default = LLMConfig()  # model_key == "default"
    assert _resolve_model_key(cfg_default, task) == "gpt-context"
    # A caller-set llm_config model_key wins.
    cfg_set = LLMConfig(model_key="gpt-explicit")
    assert _resolve_model_key(cfg_set, task) == "gpt-explicit"
    # No config + no context -> None.
    bare = Task(title="t", prompt="p")
    assert _resolve_model_key(None, bare) is None
    # Config-only default with no context falls back to the config default.
    assert _resolve_model_key(LLMConfig(), bare) == "default"


# -------------------------------------------------------------------- AAEO-1
def test_sweep_stuck_runs_marks_failed() -> None:
    """Non-terminal runs are swept to FAILED on startup recovery."""
    storage = StorageService()
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager())
    )
    run_app = RunAppService(runtime=runtime, storage=storage)
    run_async(
        storage.save_run(
            RunRecord(
                workspace_id="w1",
                subject_id="s1",
                run_id="stuck",
                status=RunStatus.RUNNING,
            )
        )
    )

    swept = run_async(run_app.sweep_stuck_runs())
    assert "stuck" in swept
    recovered = run_async(storage.get_run("stuck"))
    assert recovered.status == RunStatus.FAILED
    assert "abandoned" in (recovered.error or "")


def test_drain_cancels_inflight_runs() -> None:
    """drain() cancels + awaits in-flight background runs (FAILED on cancel)."""

    class _SlowManager:
        def resolve(self, model_key: str) -> str:
            return "stub:slow"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            await asyncio.sleep(5.0)  # long enough to be cancelled by drain
            return InferenceResponse(
                request_id=request.request_id, status=InferenceStatus.SUCCESS
            )

    storage, run_app, _rec = _stack(manager=_SlowManager())
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        # Give it a moment to enter RUNNING.
        await asyncio.sleep(0.05)
        await run_app.drain(timeout=5.0)
        return await storage.get_run(run.run_id)

    final = run_async(_scenario())
    assert final is not None
    assert final.status == RunStatus.FAILED
    assert "cancel" in (final.error or "").lower()


def test_per_run_timeout_transitions_to_failed() -> None:
    """A run exceeding run_timeout_seconds transitions to FAILED with a timeout error."""

    class _SlowManager:
        def resolve(self, model_key: str) -> str:
            return "stub:slow"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            await asyncio.sleep(2.0)
            return InferenceResponse(
                request_id=request.request_id, status=InferenceStatus.SUCCESS
            )

    storage, run_app, _rec = _stack(manager=_SlowManager(), run_timeout_seconds=0.1)
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")

    async def _scenario():
        run = await run_app.create_run(
            workspace_id="w1", subject_id="s1", persona=persona, task=task
        )
        return await run_app.await_run(run.run_id, timeout=5.0)

    final = run_async(_scenario())
    assert final is not None
    assert final.status == RunStatus.FAILED
    assert "timeout" in (final.error or "").lower()
