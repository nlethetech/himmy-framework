"""Tests for the orchestrators kernel: workflow stepping, state threading, tools."""

from __future__ import annotations

from opensims.agents.personas.persona import Persona
from opensims.core.events import EventType
from opensims.entities.registry import EntityRegistry
from opensims.orchestrators import (
    Workflow,
    WorkflowOrchestrator,
    WorkflowStep,
)
from opensims.runtime.single_agent import SingleAgentRuntime
from opensims.services.inference.client_manager import StubClientManager
from opensims.services.inference.models import ResponseFormat
from opensims.services.inference.service import InferenceService
from opensims.services.storage.service import StorageService
from opensims.services.tools.registry import ToolRegistry, register_local_tool
from opensims.services.tools.service import ToolService
from tests.conftest import run_async


def _runtime_with_tools():
    storage = StorageService()
    registry = EntityRegistry()
    tool_registry = ToolRegistry()

    register_local_tool(
        tool_registry,
        name="count_words",
        handler=lambda args: {"count": len(str(args.get("text", "")).split())},
        args_json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
        },
    )
    register_local_tool(
        tool_registry,
        name="score_sentiment",
        handler=lambda args: {"sentiment": "positive"},
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


def test_workflow_runs_all_steps_and_threads_state() -> None:
    """Each step's output_key feeds the next step's subtask via state.format()."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="analyze",
        steps=[
            WorkflowStep(
                name="draft", subtask="Draft a note about {topic}", output_key="draft"
            ),
            WorkflowStep(name="refine", subtask="Refine: {draft}", output_key="final"),
        ],
    )
    persona = Persona(name="Writer")
    result = run_async(orch.run(workflow, persona, initial_state={"topic": "ACME"}))
    assert result.status == "completed"
    assert len(result.step_results) == 2
    assert all(r.status == "completed" for r in result.step_results)
    assert "draft" in result.final_state
    assert "final" in result.final_state
    assert result.thread_id is not None


def test_missing_state_key_yields_failed_step_not_keyerror() -> None:
    """A subtask referencing an unknown state key fails the step cleanly."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="bad",
        steps=[WorkflowStep(name="oops", subtask="Use {missing_key}")],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "failed"
    assert result.step_results[0].status == "failed"
    assert "missing_key" in (result.step_results[0].error or "")


def test_sequential_tools_step_drives_one_tool_per_call() -> None:
    """A sequential-tools step aggregates one tool call per workflow iteration."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="seq",
        steps=[
            WorkflowStep(
                name="pipeline",
                subtask="Analyze {topic}",
                tool_names=["count_words", "score_sentiment"],
                sequential_tools=True,
            )
        ],
    )
    persona = Persona(name="W")
    result = run_async(
        orch.run(workflow, persona, initial_state={"topic": "the market"})
    )
    assert result.status == "completed"
    step = result.step_results[0]
    # RO-4: tool_calls are real typed ToolCallRecord objects, not dicts.
    from opensims.services.inference.models import ToolCallRecord

    assert all(isinstance(c, ToolCallRecord) for c in step.tool_calls)
    # RO-7: the call ORDER must equal the declared tool_names sequence — this is
    # the load-bearing guarantee of WORKFLOW mode, not just set-membership.
    called_in_order = [c.tool_name for c in step.tool_calls]
    assert called_in_order == ["count_words", "score_sentiment"]


def test_workflow_emits_lifecycle_events() -> None:
    """The orchestrator emits WORKFLOW_STARTED/STEP_COMPLETED/FINISHED."""
    runtime, storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="evt",
        steps=[WorkflowStep(name="s1", subtask="do {x}")],
    )
    persona = Persona(name="W")
    run_async(orch.run(workflow, persona, initial_state={"x": "thing"}))
    events = run_async(storage.list_events())
    types = {e.event_type for e in events}
    assert EventType.WORKFLOW_STARTED in types
    assert EventType.WORKFLOW_STEP_COMPLETED in types
    assert EventType.WORKFLOW_FINISHED in types


def test_structured_step_sets_state_from_structured_output() -> None:
    """A structured step stores its parsed structured output under output_key."""
    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    schema = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }
    workflow = Workflow(
        name="report",
        steps=[
            WorkflowStep(
                name="summarize",
                subtask="Summarize {topic}",
                response_format=ResponseFormat.STRUCTURED_OUTPUT,
                output_json_schema=schema,
                output_key="report",
            )
        ],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona, initial_state={"topic": "ACME"}))
    assert result.status == "completed"
    report = result.final_state["report"]
    assert isinstance(report, dict)
    assert "summary" in report


# --------------------------------------------------------------------- RO tests
class _FlakyManager:
    """Fails the first ``fail_first`` generate calls, then succeeds (RO-3)."""

    def __init__(self, *, fail_first: int = 1) -> None:
        self._remaining = fail_first
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request):  # noqa: ANN001
        from opensims.services.inference.client_manager import StubClientManager
        from opensims.services.inference.models import (
            InferenceError,
            InferenceErrorCode,
            InferenceResponse,
            InferenceStatus,
        )

        self.calls += 1
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


def _runtime_with_manager(manager):
    from opensims.runtime.single_agent import SingleAgentRuntime
    from opensims.services.inference.service import InferenceService

    storage = StorageService()
    return (
        SingleAgentRuntime(
            inference_service=InferenceService(manager, event_sink=storage),
            memory_store=storage,
        ),
        storage,
    )


def test_workflow_result_carries_typed_tool_records() -> None:
    """RO-4: single-step tool_calls/returns are typed ToolCallRecord objects."""
    from opensims.services.inference.models import ToolCallRecord

    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(
        name="tooly",
        steps=[
            WorkflowStep(
                name="lookup",
                subtask="look at {topic}",
                tool_names=["count_words"],
            )
        ],
    )
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona, initial_state={"topic": "x"}))
    assert result.status == "completed"
    calls = result.step_results[0].tool_calls
    assert calls and all(isinstance(c, ToolCallRecord) for c in calls)
    assert calls[0].tool_name == "count_words"


def test_workflow_resume_from_failed_step() -> None:
    """RO-3: a failed workflow can resume from next_index with accumulated state."""
    from opensims.runtime.single_agent import SingleAgentRuntime
    from opensims.services.inference.service import InferenceService

    # Manager that fails exactly the 2nd generate call, succeeds afterward.
    class _FailNth:
        def __init__(self, *, fail_on: int) -> None:
            self._fail_on = fail_on
            self.calls = 0

        def resolve(self, model_key: str) -> str:
            return f"stub:{model_key}"

        async def generate(self, request):  # noqa: ANN001
            from opensims.services.inference.client_manager import (
                StubClientManager,
            )
            from opensims.services.inference.models import (
                InferenceError,
                InferenceErrorCode,
                InferenceResponse,
                InferenceStatus,
            )

            self.calls += 1
            if self.calls == self._fail_on:
                return InferenceResponse(
                    request_id=request.request_id,
                    status=InferenceStatus.FAILED,
                    error=InferenceError(
                        code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                        message="down",
                        retryable=False,
                    ),
                )
            return await StubClientManager().generate(request)

    storage = StorageService()
    manager = _FailNth(fail_on=2)
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
    assert first.status == "partial"  # s0 done, s1 failed
    assert first.next_index == 1
    assert "o0" in first.final_state

    # Resume from the failed step; the manager now succeeds.
    second = run_async(orch.run(workflow, persona, resume=first))
    assert second.status in ("completed", "partial")
    assert second.start_index == 1
    assert "o1" in second.final_state and "o2" in second.final_state


def test_per_step_retry_recovers_transient_failure() -> None:
    """RO-3: max_step_retries retries a transient failure and completes the step."""
    manager = _FlakyManager(fail_first=1)
    runtime, _storage = _runtime_with_manager(manager)
    orch = WorkflowOrchestrator(
        runtime,
        max_step_retries=2,
        retry_base_delay_seconds=0.0,
        retry_jitter_seconds=0.0,
    )
    workflow = Workflow(name="retry", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "completed"
    assert result.step_results[0].attempts == 2


def test_idempotency_key_short_circuits_completed_workflow() -> None:
    """RO-3: a completed workflow with an idempotency_key is cached + reused."""
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
    # Same object returned from cache — second call did not re-run.
    assert r2 is r1


def test_orchestrator_on_event_callback() -> None:
    """RO-6: a caller-facing on_event callback gets workflow lifecycle events."""
    seen: list = []

    async def listener(event) -> None:  # noqa: ANN001
        seen.append(event.event_type)

    runtime, _storage = _runtime_with_tools()
    orch = WorkflowOrchestrator(runtime, on_event=listener)
    workflow = Workflow(name="evt", steps=[WorkflowStep(name="s", subtask="do {x}")])
    persona = Persona(name="W")
    run_async(orch.run(workflow, persona, initial_state={"x": "y"}))
    assert EventType.WORKFLOW_STARTED in seen
    assert EventType.WORKFLOW_STEP_COMPLETED in seen
    assert EventType.WORKFLOW_FINISHED in seen


def test_failed_inference_fails_workflow_step() -> None:
    """RO-4: a FAILED run becomes a failed step instead of a phantom success."""
    manager = _FlakyManager(fail_first=99)  # always fails
    runtime, _storage = _runtime_with_manager(manager)
    orch = WorkflowOrchestrator(runtime)
    workflow = Workflow(name="bad", steps=[WorkflowStep(name="s", subtask="do")])
    persona = Persona(name="W")
    result = run_async(orch.run(workflow, persona))
    assert result.status == "failed"
    assert result.step_results[0].status == "failed"
    assert "transient" in (result.step_results[0].error or "")
