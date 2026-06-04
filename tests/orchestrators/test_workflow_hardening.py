"""Expanded RO hardening tests for WorkflowOrchestrator.

Complements ``test_workflow.py`` by locking in production recovery/ordering
behaviors the base suite only partially asserts: sequential-tool ORDER preserved
across more than two tools (RO-7), workflow fail-fast when a step's tool is
missing / inference fails (RO-4), resume from a failed step with accumulated
state (RO-3), per-step retry attempt counting + non-retry default (RO-3),
idempotency caching only for completed runs (RO-3), the total-timeout
cancellation path emitting WORKFLOW_FINISHED(cancelled) (RO-1), and the
on_event lifecycle ordering (RO-6).

Offline-first: everything runs on the in-memory StorageService + StubClientManager.
"""

from __future__ import annotations

import asyncio

from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType
from himmy.entities.registry import EntityRegistry
from himmy.orchestrators import Workflow, WorkflowOrchestrator, WorkflowStep
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


# --------------------------------------------------------------- local helpers
def _runtime_with_tools(tool_names: list[str] | None = None):
    """A runtime wired with a few deterministic local tools + storage."""
    storage = StorageService()
    registry = EntityRegistry()
    tool_registry = ToolRegistry()
    names = tool_names or ["alpha", "beta", "gamma"]
    for name in names:
        register_local_tool(
            tool_registry,
            name=name,
            handler=lambda args, _n=name: {"tool": _n},
            args_json_schema={"type": "object", "properties": {}},
        )
    tool_service = ToolService(tool_registry, event_sink=storage)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        tool_service=tool_service,
        entity_registry=registry,
    )
    return runtime, storage


def _runtime_with_manager(manager):
    storage = StorageService()
    return (
        SingleAgentRuntime(
            inference_service=InferenceService(manager, event_sink=storage),
            memory_store=storage,
        ),
        storage,
    )


class _AlwaysFailManager:
    """A manager that always returns a FAILED response (no raise)."""

    def __init__(self, *, message: str = "down") -> None:
        self._message = message
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request):  # noqa: ANN001
        self.calls += 1
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                message=self._message,
                retryable=False,
            ),
        )


class _SlowManager:
    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request):  # noqa: ANN001
        await asyncio.sleep(5.0)
        raise AssertionError("slow manager should have been cancelled")


# ------------------------------------------------------- RO-7 sequential ORDER
def test_sequential_tools_preserves_order_across_three_tools() -> None:
    """RO-7: a 3-tool sequential step calls tools in EXACTLY the declared order.

    The 2-tool base test could pass by coincidence of set iteration; three
    distinct tools make order the load-bearing assertion of WORKFLOW mode.
    """
    runtime, _storage = _runtime_with_tools(["alpha", "beta", "gamma"])
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="seq3",
        steps=[
            WorkflowStep(
                name="pipeline",
                subtask="Analyze {topic}",
                tool_names=["gamma", "alpha", "beta"],  # deliberately not sorted
                sequential_tools=True,
            )
        ],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona, initial_state={"topic": "x"}))
    assert result.status == "completed"
    step = result.step_results[0]
    assert all(isinstance(c, ToolCallRecord) for c in step.tool_calls)
    called_in_order = [c.tool_name for c in step.tool_calls]
    assert called_in_order == ["gamma", "alpha", "beta"]
    # One return per call, in the same order.
    assert [r.tool_name for r in step.tool_returns] == ["gamma", "alpha", "beta"]


def test_sequential_tools_with_two_names_runs_one_per_iteration() -> None:
    """RO-7: sequential mode runs exactly one tool per workflow iteration."""
    runtime, _storage = _runtime_with_tools(["alpha", "beta"])
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="seq2",
        steps=[
            WorkflowStep(
                name="p",
                subtask="do {x}",
                tool_names=["alpha", "beta"],
                sequential_tools=True,
            )
        ],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona, initial_state={"x": "y"}))
    step = result.step_results[0]
    assert [c.tool_name for c in step.tool_calls] == ["alpha", "beta"]


# ----------------------------------------------------- RO-4 fail-fast on failure
def test_workflow_stops_on_first_failed_step_by_default() -> None:
    """RO-4: stop_on_step_failure stops the loop at the failing step.

    A 3-step workflow whose 1st step fails leaves step 2/3 unexecuted, status
    'failed', and next_index pointing at the failed step.
    """
    manager = _AlwaysFailManager()
    runtime, _storage = _runtime_with_manager(manager)
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="ff",
        steps=[
            WorkflowStep(name="s0", subtask="a"),
            WorkflowStep(name="s1", subtask="b"),
            WorkflowStep(name="s2", subtask="c"),
        ],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "failed"
    assert len(result.step_results) == 1  # stopped after the first failed step
    assert result.step_results[0].status == "failed"
    assert result.next_index == 0


def test_workflow_continues_past_failure_when_not_stopping() -> None:
    """RO-4: stop_on_step_failure=False runs all steps; failures still recorded."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="cont",
        steps=[
            WorkflowStep(name="ok0", subtask="fine"),
            WorkflowStep(name="bad", subtask="needs {nope}"),  # missing-key failure
            WorkflowStep(name="ok2", subtask="fine again"),
        ],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona, stop_on_step_failure=False))
    assert len(result.step_results) == 3
    statuses = [r.status for r in result.step_results]
    assert statuses == ["completed", "failed", "completed"]
    assert result.status == "partial"  # some completed, some failed


def test_failed_inference_does_not_phantom_succeed_step() -> None:
    """RO-4: a FAILED inference becomes a failed step carrying the provider error."""
    manager = _AlwaysFailManager(message="provider gone")
    runtime, _storage = _runtime_with_manager(manager)
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(name="bad", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "failed"
    assert result.step_results[0].status == "failed"
    assert "provider gone" in (result.step_results[0].error or "")


# --------------------------------------------------------------- RO-3 recovery
def test_resume_from_failed_step_preserves_accumulated_state() -> None:
    """RO-3: a partial workflow resumes from next_index with prior state intact."""

    class _FailNth:
        def __init__(self, *, fail_on: int) -> None:
            self._fail_on = fail_on
            self.calls = 0

        def resolve(self, model_key: str) -> str:
            return f"stub:{model_key}"

        async def generate(self, request):  # noqa: ANN001
            self.calls += 1
            if self.calls == self._fail_on:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                        message="blip",
                        retryable=False,
                    ),
                )
            return await StubClientManager().generate(request)

    storage = StorageService()
    manager = _FailNth(fail_on=2)  # s0 ok, s1 fails
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(manager, event_sink=storage),
        memory_store=storage,
    )
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="three",
        steps=[
            WorkflowStep(name="s0", subtask="a", output_key="o0"),
            WorkflowStep(name="s1", subtask="b", output_key="o1"),
            WorkflowStep(name="s2", subtask="c", output_key="o2"),
        ],
    )
    persona = Persona(name="W")
    first = run_async(orch.run(workflow, persona))
    assert first.status == "partial"
    assert first.next_index == 1
    assert "o0" in first.final_state
    assert "o1" not in first.final_state

    second = run_async(orch.run(workflow, persona, resume=first))
    assert second.start_index == 1
    assert second.status in ("completed", "partial")
    # Prior output preserved, and the resumed steps add their outputs.
    assert "o0" in second.final_state
    assert "o1" in second.final_state and "o2" in second.final_state


def test_resume_does_not_rerun_already_completed_steps() -> None:
    """RO-3: resuming begins at next_index, not from step 0."""

    class _CountingFailNth:
        def __init__(self, *, fail_on: int) -> None:
            self._fail_on = fail_on
            self.calls = 0

        def resolve(self, model_key: str) -> str:
            return f"stub:{model_key}"

        async def generate(self, request):  # noqa: ANN001
            self.calls += 1
            if self.calls == self._fail_on:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                        message="x",
                        retryable=False,
                    ),
                )
            return await StubClientManager().generate(request)

    storage = StorageService()
    manager = _CountingFailNth(fail_on=2)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(manager, event_sink=storage),
        memory_store=storage,
    )
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="resume",
        steps=[
            WorkflowStep(name="s0", subtask="a", output_key="o0"),
            WorkflowStep(name="s1", subtask="b", output_key="o1"),
        ],
    )
    persona = Persona(name="W")
    first = run_async(orch.run(workflow, persona))
    calls_after_first = manager.calls  # s0 ok (1) + s1 fail (2) = 2
    assert calls_after_first == 2
    run_async(orch.run(workflow, persona, resume=first))
    # Resume only re-runs s1 -> exactly one more generate call.
    assert manager.calls == calls_after_first + 1


def test_per_step_retry_counts_attempts() -> None:
    """RO-3: max_step_retries retries a transient failure; attempts is recorded."""

    class _FlakyManager:
        def __init__(self, *, fail_first: int) -> None:
            self._remaining = fail_first

        def resolve(self, model_key: str) -> str:
            return f"stub:{model_key}"

        async def generate(self, request):  # noqa: ANN001
            if self._remaining > 0:
                self._remaining -= 1
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                        message="transient",
                        retryable=False,
                    ),
                )
            return await StubClientManager().generate(request)

    runtime, _storage = _runtime_with_manager(_FlakyManager(fail_first=2))
    orch = WorkflowOrchestrator(
        runtime,
        max_step_retries=3,
        retry_base_delay_seconds=0.0,
        retry_jitter_seconds=0.0,
    )
    workflow = Workflow(name="retry", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "completed"
    assert result.step_results[0].attempts == 3  # 2 failures + 1 success


def test_no_retry_by_default_records_single_attempt() -> None:
    """RO-3: with the default max_step_retries=0 a failed step has attempts==1."""
    runtime, _storage = _runtime_with_manager(_AlwaysFailManager())
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(name="r", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "failed"
    assert result.step_results[0].attempts == 1


# --------------------------------------------------------- RO-3 idempotency
def test_idempotency_caches_only_completed_results() -> None:
    """RO-3: a completed workflow is cached + reused; a re-run is the same object."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(name="idem", steps=[WorkflowStep(name="s", subtask="do {x}")])
    persona = Persona(name="W")
    r1 = run_async(
        orch.run(workflow, persona, initial_state={"x": "1"}, idempotency_key="K")
    )
    r2 = run_async(
        orch.run(workflow, persona, initial_state={"x": "2"}, idempotency_key="K")
    )
    assert r1.status == "completed"
    assert r2 is r1  # cache hit, not re-run


def test_idempotency_does_not_cache_failed_results() -> None:
    """RO-3: a failed workflow is NOT cached; a later run with the key re-executes."""
    storage = StorageService()
    # Manager fails the first generate, then succeeds on subsequent calls.
    state = {"calls": 0}

    class _FlipManager:
        def resolve(self, model_key: str) -> str:
            return f"stub:{model_key}"

        async def generate(self, request):  # noqa: ANN001
            state["calls"] += 1
            if state["calls"] == 1:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                        message="first fails",
                        retryable=False,
                    ),
                )
            return await StubClientManager().generate(request)

    runtime = SingleAgentRuntime(
        inference_service=InferenceService(_FlipManager(), event_sink=storage),
        memory_store=storage,
    )
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(name="idem2", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    first = run_async(orch.run(workflow, persona, idempotency_key="K2"))
    assert first.status == "failed"
    second = run_async(orch.run(workflow, persona, idempotency_key="K2"))
    # The failed result was not cached, so the workflow ran again and now passes.
    assert second.status == "completed"
    assert second is not first


# ----------------------------------------------------------------- RO-1 timeout
def test_total_timeout_cancels_and_emits_workflow_finished_cancelled() -> None:
    """RO-1: a workflow exceeding total_timeout raises + emits FINISHED(cancelled)."""
    storage = StorageService()
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(_SlowManager(), event_sink=storage),
        memory_store=storage,
    )
    orch = WorkflowOrchestrator(runtime, total_timeout_seconds=0.05)
    workflow = Workflow(name="slow", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    raised = False
    try:
        run_async(orch.run(workflow, persona))
    except asyncio.CancelledError:
        raised = True
    assert raised
    events = run_async(storage.list_events())
    finished = [e for e in events if e.event_type == EventType.WORKFLOW_FINISHED]
    assert finished and finished[-1].error == "cancelled"
    assert finished[-1].payload.get("status") == "cancelled"


def test_per_step_timeout_fails_step_without_killing_workflow() -> None:
    """RO-1: a per-step timeout fails just that step (does not raise the run)."""
    storage = StorageService()
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(_SlowManager(), event_sink=storage),
        memory_store=storage,
    )
    orch = WorkflowOrchestrator(runtime, step_timeout_seconds=0.05)
    workflow = Workflow(name="st", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    # The step's timeout surfaces as a failed step, not a raised run.
    assert result.status == "failed"
    assert result.step_results[0].status == "failed"


# ----------------------------------------------------------------- RO-6 events
def test_orchestrator_on_event_lifecycle_order() -> None:
    """RO-6: on_event sees STARTED first, FINISHED last, with a STEP in between."""
    seen: list = []

    async def listener(event) -> None:  # noqa: ANN001
        seen.append(event.event_type)

    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime, on_event=listener)
    workflow = Workflow(name="evt", steps=[WorkflowStep(name="s", subtask="do {x}")])
    persona = Persona(name="W")
    run_async(orch.run(workflow, persona, initial_state={"x": "y"}))
    assert seen[0] == EventType.WORKFLOW_STARTED
    assert seen[-1] == EventType.WORKFLOW_FINISHED
    assert EventType.WORKFLOW_STEP_COMPLETED in seen


def test_step_completed_event_per_step() -> None:
    """RO-6: one WORKFLOW_STEP_COMPLETED event is emitted per step."""
    runtime, storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="multi",
        steps=[
            WorkflowStep(name="s0", subtask="a {x}", output_key="o0"),
            WorkflowStep(name="s1", subtask="b {o0}", output_key="o1"),
        ],
    )
    persona = Persona(name="W")
    run_async(orch.run(workflow, persona, initial_state={"x": "1"}))
    events = run_async(storage.list_events())
    step_events = [
        e for e in events if e.event_type == EventType.WORKFLOW_STEP_COMPLETED
    ]
    assert len(step_events) == 2
    names = [e.payload.get("step_name") for e in step_events]
    assert names == ["s0", "s1"]


def test_state_threads_output_key_into_next_step() -> None:
    """RO-4: a step's output_key feeds the next step's {placeholder} via state."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="thread",
        steps=[
            WorkflowStep(name="draft", subtask="Draft {topic}", output_key="draft"),
            WorkflowStep(name="refine", subtask="Refine: {draft}", output_key="final"),
        ],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona, initial_state={"topic": "ACME"}))
    assert result.status == "completed"
    assert "draft" in result.final_state
    assert "final" in result.final_state


def test_missing_state_key_is_clean_failure_not_keyerror() -> None:
    """RO-4: a subtask referencing an unknown key fails the step (no KeyError)."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="bad", steps=[WorkflowStep(name="oops", subtask="Use {missing}")]
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "failed"
    assert result.step_results[0].status == "failed"
    assert "missing" in (result.step_results[0].error or "")
