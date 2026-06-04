"""Expanded RO hardening tests for SingleAgentRuntime.run_task / run_task_detailed.

These complement ``test_runtime.py`` by locking in production behaviors that the
base suite only partially asserts: the TOOL_FAILED leg of the tool-event contract
(RO-2), inference-level timeout vs. deadline cancellation (RO-1), the non-strict
snapshot diagnostic on the audit trail (RO-11), thread.version stamping into the
saved/registered record without a registry (RO-8), on_event ordering and
fan-out (RO-6), and the partial-thread persistence guarantee on cancellation.

Offline-first: everything runs on the in-memory StorageService + StubClientManager
with no provider/DB; there is nothing to skip here.
"""

from __future__ import annotations

import asyncio
import json

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread, MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.core.events import EventType
from himmy.entities.registry import EntityRegistry
from himmy.runtime import RunResult, SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceResponse,
    InferenceStatus,
    LLMConfig,
    ResponseFormat,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


# --------------------------------------------------------------- local helpers
def _storage_runtime(**overrides):
    """A runtime sinking everything to one in-memory StorageService."""
    storage = StorageService()
    inference = InferenceService(StubClientManager(), event_sink=storage)
    rt = SingleAgentRuntime(
        inference_service=inference, memory_store=storage, **overrides
    )
    return rt, storage


class _SlowManager:
    """A manager whose generate sleeps long enough to trip any small deadline."""

    def __init__(self, *, sleep_seconds: float = 5.0) -> None:
        self._sleep = sleep_seconds

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request):  # noqa: ANN001
        await asyncio.sleep(self._sleep)
        raise AssertionError("slow manager should have been cancelled/timed out")


# ------------------------------------------------------------------ RO-2 tools
def test_tool_failed_event_emitted_when_handler_raises() -> None:
    """RO-2: a raising tool handler yields a 'failed' row + a TOOL_FAILED event.

    Locks the negative leg of the tool-event contract the base suite only
    asserts for the success path: the runtime must emit TOOL_CALLED then
    TOOL_FAILED (not TOOL_COMPLETED) keyed on the return's outcome.
    """
    storage = StorageService()
    tool_registry = ToolRegistry()

    def raising(args: dict) -> dict:
        raise RuntimeError("kaboom")

    register_local_tool(
        tool_registry,
        name="boom",
        handler=raising,
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
        prompt="go",
        context={"tool_names": ["boom"], "response_format": "AUTO_TOOLS"},
    )
    thread = run_async(rt.run_task(persona, task))

    tool_rows = [m for m in thread.messages if m.role == MessageRole.TOOL]
    assert len(tool_rows) == 1
    assert tool_rows[0].metadata["tool_outcome"] == "failed"

    events = run_async(storage.list_events())
    called = [
        e
        for e in events
        if e.event_type == EventType.TOOL_CALLED and e.trace_id is not None
    ]
    failed = [
        e
        for e in events
        if e.event_type == EventType.TOOL_FAILED and e.trace_id is not None
    ]
    completed = [
        e
        for e in events
        if e.event_type == EventType.TOOL_COMPLETED and e.trace_id is not None
    ]
    assert len(called) == 1
    assert len(failed) == 1
    assert len(completed) == 0
    # The failed return event keeps the call's id and a non-null error.
    assert failed[0].tool_call_id == called[0].tool_call_id
    assert failed[0].error is not None
    assert failed[0].payload["tool_outcome"] == "failed"


def test_tool_events_share_request_id_with_inference() -> None:
    """RO-2: the runtime tool events carry the same request_id as the inference leg."""
    storage = StorageService()
    tool_registry = ToolRegistry()
    register_local_tool(
        tool_registry,
        name="get_price",
        handler=lambda args: {"price": 1},
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
        prompt="price",
        context={"tool_names": ["get_price"], "response_format": "AUTO_TOOLS"},
    )
    run_async(rt.run_task(persona, task))
    events = run_async(storage.list_events())
    requested = next(e for e in events if e.event_type == EventType.INFERENCE_REQUESTED)
    called = next(
        e
        for e in events
        if e.event_type == EventType.TOOL_CALLED and e.trace_id is not None
    )
    assert called.request_id == requested.request_id


# ----------------------------------------------------------------- RO-1 timing
def test_inference_level_timeout_surfaces_as_failed_not_cancellation() -> None:
    """RO-1: a per-request timeout becomes a FAILED(TIMEOUT) run, NOT a raise.

    Distinguishes the inference-service-level timeout (InferenceService bounds
    request.timeout_seconds and normalizes to a FAILED response) from the
    runtime-level deadline (which raises CancelledError). The base suite covers
    the deadline raise; this locks the FAILED-not-raise path.
    """
    storage = StorageService()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_SlowManager(), event_sink=storage),
        memory_store=storage,
    )
    persona = Persona(name="A")
    cfg = LLMConfig(timeout_seconds=0.05)
    result = run_async(
        rt.run_task_detailed(persona, Task(title="t", prompt="p"), llm_config=cfg)
    )
    assert result.status == "FAILED"
    assert result.error_code == "TIMEOUT"
    assert not result.succeeded
    events = run_async(storage.list_events())
    types = [e.event_type for e in events]
    assert EventType.INFERENCE_FAILED in types
    finished = next(e for e in events if e.event_type == EventType.AGENT_RUN_FINISHED)
    assert finished.error is not None  # carries the timeout message


def test_deadline_cancel_leaves_partial_thread_but_still_saves() -> None:
    """RO-1: a deadline-cancelled run leaves SYSTEM+USER (no ASSISTANT) yet saves.

    Asserts the partial-mutation + persistence guarantee: the thread is appended
    up to USER, no ASSISTANT row is added, and the partial thread is still saved
    so the audit trail is not lost on cancellation.
    """
    storage = StorageService()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_SlowManager(), event_sink=storage),
        memory_store=storage,
    )
    persona = Persona(name="A")
    thread = ChatThread(agent_id=persona.agent_id)
    raised = False
    try:
        run_async(
            rt.run_task(
                persona,
                Task(title="t", prompt="p"),
                thread=thread,
                deadline_seconds=0.05,
            )
        )
    except asyncio.CancelledError:
        raised = True
    assert raised
    roles = {m.role for m in thread.messages}
    assert MessageRole.USER in roles
    assert MessageRole.ASSISTANT not in roles
    # The partial thread is persisted (so a UI can show the in-flight state).
    assert run_async(storage.load_thread(thread.thread_id)) is not None
    events = run_async(storage.list_events())
    finished = [e for e in events if e.event_type == EventType.AGENT_RUN_FINISHED]
    assert finished and finished[-1].error == "cancelled"
    assert finished[-1].payload.get("status") == "CANCELLED"


def test_default_deadline_seconds_constructor_applies() -> None:
    """RO-1: a constructor default_deadline_seconds bounds runs with no per-call arg."""
    storage = StorageService()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_SlowManager(), event_sink=storage),
        memory_store=storage,
        default_deadline_seconds=0.05,
    )
    persona = Persona(name="A")
    raised = False
    try:
        run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    except asyncio.CancelledError:
        raised = True
    assert raised


# ------------------------------------------------------------- RO-11 snapshot
def test_non_strict_missing_snapshot_degrades_with_diagnostic() -> None:
    """RO-11: a missing snapshot in non-strict mode degrades but records the error.

    The run still produces an ASSISTANT message, but the audit trail carries a
    CONTEXT_SNAPSHOT_BUILT event with the error AND the AGENT_RUN_STARTED payload
    carries snapshot_error — so 'requested but unavailable' is distinguishable
    from 'no snapshot requested'.
    """
    storage = StorageService()
    context = ContextService(storage_service=storage)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
        strict_snapshot=False,
    )
    persona = Persona(name="A")
    thread = run_async(
        rt.run_task(persona, Task(title="t", prompt="p"), snapshot_id="missing")
    )
    assert thread.last_message.role == MessageRole.ASSISTANT  # degraded, not crashed
    events = run_async(storage.list_events())
    snap = [e for e in events if e.event_type == EventType.CONTEXT_SNAPSHOT_BUILT]
    assert len(snap) == 1
    assert snap[0].error is not None
    assert snap[0].payload.get("snapshot_error") is not None
    started = next(e for e in events if e.event_type == EventType.AGENT_RUN_STARTED)
    assert started.payload.get("snapshot_error") is not None


def test_no_snapshot_requested_has_no_snapshot_error() -> None:
    """RO-11: an ordinary run (no snapshot requested) carries no snapshot_error."""
    storage = StorageService()
    context = ContextService(storage_service=storage)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=context,
    )
    persona = Persona(name="A")
    run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    events = run_async(storage.list_events())
    started = next(e for e in events if e.event_type == EventType.AGENT_RUN_STARTED)
    assert started.payload.get("snapshot_error") is None
    # No CONTEXT_SNAPSHOT_BUILT diagnostic when nothing was requested.
    assert not [e for e in events if e.event_type == EventType.CONTEXT_SNAPSHOT_BUILT]


def test_strict_snapshot_without_context_service_raises() -> None:
    """RO-11: strict mode + a requested snapshot but no context_service raises."""
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        strict_snapshot=True,
    )
    persona = Persona(name="A")
    try:
        run_async(rt.run_task(persona, Task(title="t", prompt="p"), snapshot_id="x"))
    except HimmyError as exc:
        assert "context_service" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected HimmyError in strict mode")


# ------------------------------------------------------------- RO-8 versioning
def test_thread_version_stamped_into_registered_record_without_registry() -> None:
    """RO-8: the saved thread carries version==2 on the 2nd turn (no registry)."""
    storage = StorageService()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        memory_store=storage,
    )
    persona = Persona(name="A")
    th = run_async(rt.run_task(persona, Task(title="u1", prompt="a")))
    assert th.version == 1
    th = run_async(rt.run_task(persona, Task(title="u2", prompt="b"), thread=th))
    assert th.version == 2
    # The persisted thread reflects the bumped version.
    saved = run_async(storage.load_thread(th.thread_id))
    assert saved is not None
    assert saved.version == 2


def test_thread_version_with_registry_projects_each_version() -> None:
    """RO-8: with a registry, each turn projects a chat_thread record version."""
    storage = StorageService()
    registry = EntityRegistry()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        entity_registry=registry,
    )
    persona = Persona(name="A")
    th = run_async(rt.run_task(persona, Task(title="t1", prompt="a")))
    th = run_async(rt.run_task(persona, Task(title="t2", prompt="b"), thread=th))
    assert th.version == 2
    versions = sorted(r.version for r in registry.list_by_kind("chat_thread"))
    assert 1 in versions and 2 in versions


# ----------------------------------------------------------------- RO-6 events
def test_on_event_receives_events_in_lifecycle_order() -> None:
    """RO-6: the on_event callback observes STARTED before FINISHED, in order."""
    seen: list = []

    async def listener(event) -> None:  # noqa: ANN001
        seen.append(event.event_type)

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        on_event=listener,
    )
    persona = Persona(name="A")
    run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    assert seen[0] == EventType.AGENT_RUN_STARTED
    assert seen[-1] == EventType.AGENT_RUN_FINISHED
    assert EventType.INFERENCE_REQUESTED in seen
    assert EventType.INFERENCE_SUCCEEDED in seen


def test_on_event_list_fans_out_to_every_callback() -> None:
    """RO-6: a list of on_event callbacks all receive the run events."""
    a: list = []
    b: list = []

    async def la(e) -> None:  # noqa: ANN001
        a.append(e.event_type)

    async def lb(e) -> None:  # noqa: ANN001
        b.append(e.event_type)

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        on_event=[la, lb],
    )
    persona = Persona(name="A")
    run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    assert EventType.AGENT_RUN_FINISHED in a
    assert EventType.AGENT_RUN_FINISHED in b


def test_on_event_listener_exception_never_breaks_the_run() -> None:
    """RO-6: a throwing on_event listener is isolated; the run still completes."""

    async def boom(event) -> None:  # noqa: ANN001
        raise RuntimeError("listener exploded")

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        on_event=boom,
    )
    persona = Persona(name="A")
    thread = run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    assert thread.last_message.role == MessageRole.ASSISTANT
    assert thread.last_message.metadata["status"] == "SUCCESS"


def test_add_event_listener_registers_after_construction() -> None:
    """RO-6: add_event_listener wires a callback added after the runtime exists."""
    seen: list = []

    async def listener(event) -> None:  # noqa: ANN001
        seen.append(event.event_type)

    rt = SingleAgentRuntime(inference_service=InferenceService(StubClientManager()))
    rt.add_event_listener(listener)
    persona = Persona(name="A")
    run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    assert EventType.AGENT_RUN_STARTED in seen


# ----------------------------------------------------------- RO-5 RunResult shape
def test_run_task_detailed_carries_tokens_and_request_id_on_success() -> None:
    """RO-5: a successful detailed run exposes tokens/latency/request_id + thread."""
    rt, _storage = _storage_runtime()
    persona = Persona(name="A")
    result = run_async(rt.run_task_detailed(persona, Task(title="t", prompt="hello")))
    assert isinstance(result, RunResult)
    assert result.succeeded
    assert result.status == InferenceStatus.SUCCESS.value
    assert result.input_tokens >= 1
    assert result.output_tokens >= 1
    assert result.request_id is not None
    # trace_id is "<thread_id>:<task_id>" — it begins with the thread id.
    assert result.trace_id is not None
    assert result.trace_id.startswith(f"{result.thread.thread_id}:")
    assert result.output_text


def test_failed_run_detailed_exposes_error_and_back_compat_thread() -> None:
    """RO-5: a FAILED detailed run carries error/code AND a usable thread row."""
    storage = StorageService()

    class _Manager:
        def resolve(self, model_key: str) -> str:
            return f"stub:{model_key}"

        async def generate(self, request):  # noqa: ANN001
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                error=InferenceError(
                    code=InferenceErrorCode.QUOTA,
                    message="over budget",
                    retryable=False,
                ),
            )

    rt = SingleAgentRuntime(
        inference_service=InferenceService(_Manager(), event_sink=storage),
        memory_store=storage,
    )
    persona = Persona(name="A")
    result = run_async(rt.run_task_detailed(persona, Task(title="t", prompt="p")))
    assert result.status == "FAILED"
    assert result.error == "over budget"
    assert result.error_code == "QUOTA"
    # Back-compat: run_task callers still get a thread with the error stamped.
    last = result.thread.last_message
    assert last.role == MessageRole.ASSISTANT
    assert last.metadata["error_code"] == "QUOTA"


# ----------------------------------------------------------- RO-9 workflow guard
def test_workflow_with_unbound_step_tool_fails_fast() -> None:
    """RO-9: WORKFLOW whose step tool isn't registered raises a clear error."""
    from himmy.services.inference.models import WorkflowDefinition, WorkflowState

    storage = StorageService()
    tool_registry = ToolRegistry()
    register_local_tool(
        tool_registry,
        name="other_tool",
        handler=lambda args: {"ok": True},
        args_json_schema={"type": "object", "properties": {}},
    )
    tool_service = ToolService(tool_registry, event_sink=storage)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        tool_service=tool_service,
    )
    persona = Persona(name="A")
    state = WorkflowState(definition=WorkflowDefinition(steps=["unregistered"]))
    cfg = LLMConfig(response_format=ResponseFormat.WORKFLOW, workflow=state)
    try:
        run_async(rt.run_task(persona, Task(title="t", prompt="p"), llm_config=cfg))
    except HimmyError as exc:
        assert "unregistered" in str(exc)
    else:  # pragma: no cover - must raise
        raise AssertionError("expected HimmyError for unbound step tool")


# ------------------------------------------------------- structured / minimal
def test_structured_output_threads_into_run_result() -> None:
    """RO-5: STRUCTURED_OUTPUT parses into both the thread row and RunResult."""
    rt, _storage = _storage_runtime()
    persona = Persona(name="A")
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}, "n": {"type": "integer"}},
        "required": ["title"],
    }
    cfg = LLMConfig(
        response_format=ResponseFormat.STRUCTURED_OUTPUT, output_json_schema=schema
    )
    result = run_async(
        rt.run_task_detailed(persona, Task(title="t", prompt="rec"), llm_config=cfg)
    )
    assert isinstance(result.output_structured, dict)
    assert "title" in result.output_structured
    # The thread row content is the JSON-serialized structured payload.
    parsed = json.loads(result.thread.last_message.content)
    assert "title" in parsed


def test_minimal_runtime_emits_no_events_without_sink() -> None:
    """A bare runtime (inference only) still returns a thread and does not crash."""
    rt = SingleAgentRuntime(inference_service=InferenceService(StubClientManager()))
    persona = Persona(name="A")
    thread = run_async(rt.run_task(persona, Task(title="t", prompt="hi")))
    assert thread.last_message.role == MessageRole.ASSISTANT
    assert thread.last_message.content
