"""Tests for the runtime kernel: SingleAgentRuntime.run_task end-to-end."""

from __future__ import annotations

from opensims.agents.base_agent.task import Task
from opensims.agents.base_agent.thread import MessageRole
from opensims.agents.personas.persona import Persona
from opensims.core.events import EventType
from opensims.entities.registry import EntityRegistry
from opensims.runtime import SingleAgentRuntime
from opensims.services.context.service import ContextService
from opensims.services.inference.client_manager import StubClientManager
from opensims.services.inference.models import LLMConfig, ResponseFormat
from opensims.services.inference.service import InferenceService
from opensims.services.storage.service import StorageService
from opensims.services.tools.registry import ToolRegistry, register_local_tool
from opensims.services.tools.service import ToolService
from tests.conftest import run_async


def _runtime(**overrides):
    storage = overrides.pop("storage", None) or StorageService()
    registry = overrides.pop("registry", None) or EntityRegistry()
    inference = InferenceService(StubClientManager(), event_sink=storage)
    context = ContextService(storage_service=storage, entity_registry=registry)
    rt = SingleAgentRuntime(
        inference_service=inference,
        memory_store=storage,
        context_service=context,
        entity_registry=registry,
        **overrides,
    )
    return rt, storage, registry


def test_basic_run_appends_system_user_assistant() -> None:
    """A first run appends SYSTEM, USER, then ASSISTANT messages with metadata."""
    rt, storage, _registry = _runtime()
    persona = Persona(
        name="Analyst", description="careful", metadata={"role": "analyst"}
    )
    task = Task(title="brief", prompt="Summarize ACME.")
    thread = run_async(rt.run_task(persona, task))

    roles = [m.role for m in thread.messages]
    assert MessageRole.SYSTEM in roles
    assert MessageRole.USER in roles
    assert thread.last_message is not None
    assert thread.last_message.role == MessageRole.ASSISTANT
    assert thread.last_message.content  # non-empty echo
    # Assistant metadata carries model/latency provenance.
    md = thread.last_message.metadata
    assert "latency_ms" in md
    assert md["status"] == "SUCCESS"
    # Thread was persisted.
    assert run_async(storage.load_thread(thread.thread_id)) is not None


def test_run_emits_full_event_sequence() -> None:
    """A run emits AGENT_RUN_STARTED, INFERENCE_*, AGENT_RUN_FINISHED to the sink."""
    rt, storage, _registry = _runtime()
    persona = Persona(name="A")
    task = Task(title="t", prompt="p")
    thread = run_async(rt.run_task(persona, task))
    events = run_async(storage.list_events())
    types = {e.event_type for e in events}
    assert EventType.AGENT_RUN_STARTED in types
    assert EventType.INFERENCE_REQUESTED in types
    assert EventType.INFERENCE_SUCCEEDED in types
    assert EventType.AGENT_RUN_FINISHED in types
    # trace_id is thread:task.
    started = next(e for e in events if e.event_type == EventType.AGENT_RUN_STARTED)
    assert started.trace_id == f"{thread.thread_id}:{task.task_id}"


def test_run_registers_lineage_entities() -> None:
    """A run projects persona/prompt/thread entities and links them."""
    rt, _storage, registry = _runtime()
    persona = Persona(name="A")
    task = Task(title="t", prompt="p")
    run_async(rt.run_task(persona, task))
    assert registry.list_by_kind("persona")
    assert registry.list_by_kind("prompt")
    assert registry.list_by_kind("chat_thread")
    # The thread record links to the persona via uses_persona.
    thread_records = registry.list_by_kind("chat_thread")
    relations = {
        link.relation
        for tr in thread_records
        for link in registry.links_from(tr.record_id)
    }
    assert "uses_persona" in relations


def test_tool_calling_appends_tool_rows() -> None:
    """With a tool service + AUTO_TOOLS, TOOL rows appear before the assistant turn."""
    storage = StorageService()
    registry = EntityRegistry()
    tool_registry = ToolRegistry()

    def lookup(args: dict) -> dict:
        return {"price": 100}

    register_local_tool(
        tool_registry,
        name="get_price",
        handler=lookup,
        args_json_schema={"type": "object", "properties": {}},
    )
    tool_service = ToolService(tool_registry, event_sink=storage)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        tool_service=tool_service,
        entity_registry=registry,
    )
    persona = Persona(name="A")
    task = Task(
        title="t",
        prompt="price please",
        context={"tool_names": ["get_price"], "response_format": "AUTO_TOOLS"},
    )
    thread = run_async(rt.run_task(persona, task))
    tool_rows = [m for m in thread.messages if m.role == MessageRole.TOOL]
    assert len(tool_rows) == 1
    assert tool_rows[0].metadata["tool_name"] == "get_price"
    assert tool_rows[0].metadata["tool_outcome"] == "success"


def test_structured_output_via_llm_config() -> None:
    """STRUCTURED_OUTPUT through LLMConfig fills the assistant with JSON output."""
    rt, _storage, _registry = _runtime()
    persona = Persona(name="A")
    task = Task(title="t", prompt="give a rec")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "score": {"type": "number"}},
        "required": ["title"],
    }
    cfg = LLMConfig(
        response_format=ResponseFormat.STRUCTURED_OUTPUT, output_json_schema=schema
    )
    thread = run_async(rt.run_task(persona, task, llm_config=cfg))
    import json

    parsed = json.loads(thread.last_message.content)
    assert "title" in parsed


def test_minimal_runtime_with_only_inference() -> None:
    """The runtime works with only inference_service (all else None)."""
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        prompt_manager=None,
    )
    persona = Persona(name="A")
    task = Task(title="t", prompt="hi")
    thread = run_async(rt.run_task(persona, task))
    assert thread.last_message is not None
    assert thread.last_message.role == MessageRole.ASSISTANT


def test_second_turn_reuses_thread_and_skips_system() -> None:
    """A second run on the same thread does not append another SYSTEM message."""
    rt, _storage, _registry = _runtime()
    persona = Persona(name="A")
    t1 = Task(title="t1", prompt="first")
    thread = run_async(rt.run_task(persona, t1))
    system_count_1 = sum(1 for m in thread.messages if m.role == MessageRole.SYSTEM)
    t2 = Task(title="t2", prompt="second")
    thread = run_async(rt.run_task(persona, t2, thread=thread))
    system_count_2 = sum(1 for m in thread.messages if m.role == MessageRole.SYSTEM)
    assert system_count_1 == system_count_2 == 1
    # Two assistant turns now exist.
    assert sum(1 for m in thread.messages if m.role == MessageRole.ASSISTANT) == 2


# --------------------------------------------------------------------- RO tests
class _FailingClientManager:
    """A client manager whose ``generate`` returns a FAILED response (no raise)."""

    def __init__(self, *, code: str = "INVALID_REQUEST") -> None:
        self._code = code

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request):  # noqa: ANN001
        from opensims.services.inference.models import (
            InferenceError,
            InferenceErrorCode,
            InferenceResponse,
            InferenceStatus,
        )

        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=InferenceErrorCode(self._code), message="boom", retryable=False
            ),
        )


def test_tool_path_emits_tool_called_and_completed_events() -> None:
    """RO-2: the tool path emits TOOL_CALLED + TOOL_COMPLETED to the sink."""
    storage = StorageService()
    tool_registry = ToolRegistry()
    register_local_tool(
        tool_registry,
        name="get_price",
        handler=lambda args: {"price": 100},
        args_json_schema={"type": "object", "properties": {}},
    )
    tool_service = ToolService(tool_registry, event_sink=storage)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        tool_service=tool_service,
    )
    persona = Persona(name="A")
    task = Task(
        title="t",
        prompt="price please",
        context={"tool_names": ["get_price"], "response_format": "AUTO_TOOLS"},
    )
    run_async(rt.run_task(persona, task))
    events = run_async(storage.list_events())
    # The tool SERVICE also emits its own TOOL_* events (without a trace_id);
    # RO-2 asserts the RUNTIME's events, which carry the run's trace_id +
    # request_id and link the exchange to the run.
    called = [
        e
        for e in events
        if e.event_type == EventType.TOOL_CALLED and e.trace_id is not None
    ]
    completed = [
        e
        for e in events
        if e.event_type == EventType.TOOL_COMPLETED and e.trace_id is not None
    ]
    assert len(called) == 1
    assert called[0].tool_call_id is not None
    assert called[0].request_id is not None
    assert called[0].payload["tool_name"] == "get_price"
    assert called[0].payload["tool_args"] is not None
    assert len(completed) == 1
    assert completed[0].payload["tool_outcome"] == "success"
    # The call/return events share the tool_call_id of the TOOL row.
    assert called[0].tool_call_id == completed[0].tool_call_id


def test_thread_version_increments_with_and_without_registry() -> None:
    """RO-8: thread.version bumps on the 2nd turn even with no entity_registry."""
    # With a registry.
    rt, _storage, _registry = _runtime()
    persona = Persona(name="A")
    thread = run_async(rt.run_task(persona, Task(title="t1", prompt="a")))
    assert thread.version == 1
    thread = run_async(
        rt.run_task(persona, Task(title="t2", prompt="b"), thread=thread)
    )
    assert thread.version == 2

    # Without a registry (memory store still present so save uses the version).
    storage = StorageService()
    rt2 = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=storage,
    )
    th = run_async(rt2.run_task(persona, Task(title="u1", prompt="a")))
    assert th.version == 1
    th = run_async(rt2.run_task(persona, Task(title="u2", prompt="b"), thread=th))
    assert th.version == 2


def test_failed_inference_emits_failed_event_and_stamps_metadata() -> None:
    """RO-6/AAEO-3 seam: a FAILED response stamps status/error on the message."""
    storage = StorageService()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_FailingClientManager(), event_sink=storage),
        memory_store=storage,
    )
    persona = Persona(name="A")
    result = run_async(rt.run_task_detailed(persona, Task(title="t", prompt="p")))
    assert result.status == "FAILED"
    assert result.error == "boom"
    assert result.error_code == "INVALID_REQUEST"
    assert not result.succeeded
    last = result.thread.last_message
    assert last.metadata["status"] == "FAILED"
    assert last.metadata["error"] == "boom"
    events = run_async(storage.list_events())
    types = [e.event_type for e in events]
    assert EventType.INFERENCE_FAILED in types
    finished = next(e for e in events if e.event_type == EventType.AGENT_RUN_FINISHED)
    assert finished.error == "boom"


def test_run_task_detailed_exposes_typed_run_result() -> None:
    """RO-5: run_task_detailed returns status/cost/structured + typed records."""
    from opensims.runtime import RunResult

    rt, _storage, _registry = _runtime()
    persona = Persona(name="A")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
    }
    cfg = LLMConfig(
        response_format=ResponseFormat.STRUCTURED_OUTPUT, output_json_schema=schema
    )
    result = run_async(
        rt.run_task_detailed(persona, Task(title="t", prompt="rec"), llm_config=cfg)
    )
    assert isinstance(result, RunResult)
    assert result.succeeded
    assert isinstance(result.output_structured, dict)
    assert "title" in result.output_structured
    assert result.request_id is not None


def test_on_event_callback_receives_run_events() -> None:
    """RO-6: a caller-facing on_event callback receives streamed run events."""
    seen: list = []

    async def listener(event) -> None:  # noqa: ANN001
        seen.append(event.event_type)

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        on_event=listener,
    )
    persona = Persona(name="A")
    run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    assert EventType.AGENT_RUN_STARTED in seen
    assert EventType.AGENT_RUN_FINISHED in seen


def test_workflow_format_without_tool_service_fails_fast() -> None:
    """RO-9: WORKFLOW response_format with no tool_service raises a clear error."""
    from opensims.core.errors import OpenSimsError
    from opensims.services.inference.models import WorkflowDefinition, WorkflowState

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
    )
    persona = Persona(name="A")
    state = WorkflowState(definition=WorkflowDefinition(steps=["do_thing"]))
    cfg = LLMConfig(response_format=ResponseFormat.WORKFLOW, workflow=state)
    try:
        run_async(rt.run_task(persona, Task(title="t", prompt="p"), llm_config=cfg))
    except OpenSimsError as exc:
        assert "tool_service" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected OpenSimsError for WORKFLOW without tool_service")


def test_cancelled_run_emits_terminal_event_and_saves() -> None:
    """RO-1: a run that exceeds its deadline still emits a terminal event + saves."""
    import asyncio

    class _SlowManager:
        def resolve(self, model_key: str) -> str:
            return f"stub:{model_key}"

        async def generate(self, request):  # noqa: ANN001
            await asyncio.sleep(5.0)
            raise AssertionError("should have been cancelled")  # pragma: no cover

    storage = StorageService()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_SlowManager(), event_sink=storage),
        memory_store=storage,
    )
    persona = Persona(name="A")
    raised = False
    try:
        run_async(
            rt.run_task(persona, Task(title="t", prompt="p"), deadline_seconds=0.05)
        )
    except asyncio.CancelledError:
        raised = True
    assert raised
    events = run_async(storage.list_events())
    finished = [e for e in events if e.event_type == EventType.AGENT_RUN_FINISHED]
    assert finished and finished[-1].error == "cancelled"


def test_strict_snapshot_raises_when_requested_but_missing() -> None:
    """RO-11: strict_snapshot raises when a requested snapshot can't be resolved."""
    from opensims.core.errors import OpenSimsError

    storage = StorageService()
    context = ContextService(storage_service=storage)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=storage,
        context_service=context,
        strict_snapshot=True,
    )
    persona = Persona(name="A")
    try:
        run_async(
            rt.run_task(
                persona, Task(title="t", prompt="p"), snapshot_id="does-not-exist"
            )
        )
    except OpenSimsError:
        pass
    else:  # pragma: no cover - must raise in strict mode
        raise AssertionError("expected OpenSimsError for missing strict snapshot")
