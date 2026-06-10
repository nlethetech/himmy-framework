"""Observability through the RUNTIME seam: real runs driving span/event emission.

The existing span tests feed hand-crafted ``RunEvent``s straight into
``emit_event_span``; these drive REAL ``SingleAgentRuntime`` runs (stub
inference, a real ``ToolService``) against a fake ``logfire`` sink and assert
the spans/logs the seam emits carry the correct attributes for run start/end,
inference, tool execution, and the failure paths — plus the plain event-sink
view: ordered ``RunEvent``s with linked trace/request ids.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType, RunEvent
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceResponse,
    InferenceStatus,
)
from himmy.services.inference.service import InferenceService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


def _fresh_module():
    """Re-import the observability module so its module-level state is reset."""
    import himmy.services.observability as obs

    return importlib.reload(obs)


class _FakeSpan:
    """A minimal stand-in for a logfire span (context manager)."""

    def __init__(self, name: str, attributes: dict) -> None:
        self.name = name
        self.attributes = dict(attributes)
        self.exited = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True
        return False

    def set_attribute(self, key, value):
        self.attributes[key] = value


class _FakeLogfire:
    """A fake ``logfire`` module recording span() / info() calls."""

    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self.infos: list[tuple[str, dict]] = []

    def span(self, name, **attributes):
        span = _FakeSpan(name, attributes)
        self.spans.append(span)
        return span

    def info(self, name, **attributes):
        self.infos.append((name, attributes))

    def configure(self, **_):
        pass

    def instrument_pydantic_ai(self):
        pass


class _SinkRecorder:
    """A fake EventSink recording every runtime event in order."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


class _FailingManager:
    """A manager that returns a typed FAILED response (the inference failure path)."""

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request):  # noqa: ANN001
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=InferenceErrorCode.INVALID_REQUEST,
                message="boom",
                retryable=False,
            ),
        )


def _enable(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _FakeLogfire]:
    """Turn the seam on against a fake logfire and return (module, fake)."""
    monkeypatch.setenv("HIMMY_LOGFIRE_ENABLED", "1")
    monkeypatch.delenv("HIMMY_LOGFIRE_INCLUDE_CONTENT", raising=False)
    obs = _fresh_module()
    fake = _FakeLogfire()
    monkeypatch.setitem(importlib.sys.modules, "logfire", fake)
    obs.configure_observability()
    return obs, fake


def _tool_runtime(
    inference_manager: Any = None, *, tool_handler: Any = None
) -> SingleAgentRuntime:
    """A runtime with one ``get_price`` tool bound (no extra event sinks)."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="get_price",
        handler=tool_handler or (lambda args: {"price": 100}),
        args_json_schema={"type": "object", "properties": {}},
    )
    return SingleAgentRuntime(
        inference_service=InferenceService(
            inference_manager or StubClientManager(), max_retries=0
        ),
        tool_service=ToolService(registry),
    )


def _tool_task() -> Task:
    return Task(
        title="t",
        prompt="price please",
        context={"tool_names": ["get_price"], "response_format": "AUTO_TOOLS"},
    )


def test_run_and_tool_spans_carry_correct_attributes(monkeypatch) -> None:
    """A real tool-using run opens/closes the run + tool spans with the right attrs."""
    obs, fake = _enable(monkeypatch)
    rt = _tool_runtime()
    persona = Persona(name="A")
    task = _tool_task()
    result = run_async(rt.run_task_detailed(persona, task))
    assert result.succeeded

    # Exactly two lifecycle spans: the run span and the nested tool span.
    assert [s.name for s in fake.spans] == ["agent_run_started", "tool_called"]
    run_span, tool_span = fake.spans
    assert all(s.exited for s in fake.spans)
    assert obs._OPEN_SPANS == {}

    # Run span: trace/thread/agent identity + run payload attributes.
    trace_id = f"{result.thread.thread_id}:{task.task_id}"
    assert run_span.attributes["himmy.trace_id"] == trace_id
    assert run_span.attributes["himmy.thread_id"] == result.thread.thread_id
    assert run_span.attributes["himmy.agent_id"] == persona.agent_id
    assert run_span.attributes["himmy.payload.model_key"] == "default"
    assert run_span.attributes["himmy.payload.persona_name"] == "A"
    # Close-time attributes were recorded onto the span (status from FINISHED).
    assert run_span.attributes["himmy.payload.status"] == "SUCCESS"

    # Tool span: identity, the tool name, and the close-time outcome.
    assert tool_span.attributes["himmy.trace_id"] == trace_id
    assert tool_span.attributes["himmy.tool_call_id"]
    assert tool_span.attributes["himmy.payload.tool_name"] == "get_price"
    assert tool_span.attributes["himmy.payload.tool_outcome"] == "success"
    # Content gating (default off): tool args never land on span attributes.
    assert "himmy.payload.tool_args" not in tool_span.attributes

    # Inference is logged as point-in-time records with the request threading.
    infos = dict(fake.infos)
    assert "inference_requested" in infos
    assert "inference_succeeded" in infos
    assert infos["inference_requested"]["himmy.request_id"] == result.request_id
    assert infos["inference_requested"]["himmy.payload.model_key"] == "default"
    # Content gating (default off): the rendered prompt is dropped.
    assert "himmy.payload.rendered_prompt" not in infos["inference_requested"]


def test_inference_failure_path_logs_error_and_closes_run_span(monkeypatch) -> None:
    """A FAILED inference emits inference_failed with the error and still closes the run span."""
    _obs, fake = _enable(monkeypatch)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(_FailingManager(), max_retries=0)
    )
    persona = Persona(name="A")
    result = run_async(rt.run_task_detailed(persona, Task(title="t", prompt="p")))
    assert not result.succeeded

    infos = dict(fake.infos)
    assert "inference_failed" in infos
    assert infos["inference_failed"]["himmy.error"] == "boom"
    assert infos["inference_failed"]["himmy.payload.error_code"] == "INVALID_REQUEST"
    # The run span still closed, carrying the failure on its close attributes.
    (run_span,) = fake.spans
    assert run_span.exited
    assert run_span.attributes["himmy.error"] == "boom"
    assert run_span.attributes["himmy.payload.status"] == "FAILED"


def test_tool_failure_closes_tool_span_with_error(monkeypatch) -> None:
    """A failing tool closes its span via TOOL_FAILED with the failure recorded."""

    def _boom(args: dict) -> dict:
        raise RuntimeError("kaboom")

    obs, fake = _enable(monkeypatch)
    rt = _tool_runtime(tool_handler=_boom)
    persona = Persona(name="A")
    result = run_async(rt.run_task_detailed(persona, _tool_task()))
    assert result.succeeded  # the turn completes; the failure is the tool's

    tool_span = next(s for s in fake.spans if s.name == "tool_called")
    assert tool_span.exited
    assert tool_span.attributes["himmy.payload.tool_outcome"] == "failed"
    assert tool_span.attributes["himmy.error"] == "tool outcome: failed"
    assert obs._OPEN_SPANS == {}


def test_content_attributes_present_when_explicitly_enabled(monkeypatch) -> None:
    """With HIMMY_LOGFIRE_INCLUDE_CONTENT on, the rendered prompt reaches the span sink."""
    monkeypatch.setenv("HIMMY_LOGFIRE_ENABLED", "1")
    monkeypatch.setenv("HIMMY_LOGFIRE_INCLUDE_CONTENT", "true")
    obs = _fresh_module()
    fake = _FakeLogfire()
    monkeypatch.setitem(importlib.sys.modules, "logfire", fake)
    obs.configure_observability()

    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), max_retries=0)
    )
    run_async(rt.run_task(Persona(name="A"), Task(title="t", prompt="secret prompt")))

    infos = dict(fake.infos)
    assert (
        "secret prompt" in infos["inference_requested"]["himmy.payload.rendered_prompt"]
    )


def test_event_sink_receives_ordered_linked_run_events(monkeypatch) -> None:
    """The plain EventSink seam gets the full, ordered, trace-linked event series."""
    monkeypatch.delenv("HIMMY_LOGFIRE_ENABLED", raising=False)
    sink = _SinkRecorder()
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="get_price",
        handler=lambda args: {"price": 100},
        args_json_schema={"type": "object", "properties": {}},
    )
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), max_retries=0),
        memory_store=sink,
        tool_service=ToolService(registry),
    )
    persona = Persona(name="A")
    task = _tool_task()
    result = run_async(rt.run_task_detailed(persona, task))
    assert result.succeeded

    assert [e.event_type for e in sink.events] == [
        EventType.AGENT_RUN_STARTED,
        EventType.INFERENCE_REQUESTED,
        EventType.INFERENCE_SUCCEEDED,
        EventType.TOOL_CALLED,
        EventType.TOOL_COMPLETED,
        EventType.AGENT_RUN_FINISHED,
    ]
    # Every event is linked to the same run trace.
    trace_id = f"{result.thread.thread_id}:{task.task_id}"
    assert {e.trace_id for e in sink.events} == {trace_id}
    assert {e.agent_id for e in sink.events} == {persona.agent_id}
    # The inference + tool events thread the same request id.
    requested, succeeded, called, completed = sink.events[1:5]
    assert requested.request_id == result.request_id
    assert succeeded.request_id == requested.request_id
    assert called.request_id == requested.request_id
    assert completed.tool_call_id == called.tool_call_id
    assert completed.payload["tool_name"] == "get_price"
    assert completed.payload["tool_outcome"] == "success"
    # The terminal event carries the typed status.
    assert sink.events[-1].payload["status"] == "SUCCESS"
    assert sink.events[-1].error is None
