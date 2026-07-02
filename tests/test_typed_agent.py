"""Behavioral tests for the write-time typed Agent facade (himmy.typed_agent).

These exercise the real runtime loop: signature-derived tool schemas, the offline
structured-output path, deps injection into tools, output validation + bounded
repair retries on a *real* runtime (with a controllable client manager), and the
audited TYPED_OUTPUT_* provenance transitions on the EntityRecord spine.

Tests are plain ``def test_*`` functions that drive the async API via the shared
:func:`run_async` helper (the suite does not assume ``pytest-asyncio``).
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, Field

from himmy import build_runtime
from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.services.inference.models import (
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ResponseFormat,
    ToolCallRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.typed_agent import RunContext, TypedAgent, compile_tool_schema
from tests.conftest import run_async


# --------------------------------------------------------------------------- #
# Output / deps models used across tests                                       #
# --------------------------------------------------------------------------- #
class Weather(BaseModel):
    city: str
    summary: str = Field(min_length=1)
    temp_c: int


class Deps(BaseModel):
    api_key: str


# --------------------------------------------------------------------------- #
# A controllable client manager: returns a queued sequence of structured       #
# payloads so we can drive an invalid -> valid retry deterministically.        #
# --------------------------------------------------------------------------- #
class ScriptedClientManager:
    """A ClientManager that returns pre-scripted structured payloads in order.

    Each STRUCTURED_OUTPUT request pops the next scripted payload; once exhausted
    it repeats the last one. Non-structured requests echo. This lets a test force a
    schema-invalid structured payload first, then a valid one, to prove the typed
    agent's validate-and-retry behavior on the real runtime loop.
    """

    def __init__(self, payloads: list[Any]) -> None:
        self.provider_name = "scripted"
        self._payloads = list(payloads)
        self._index = 0
        self.structured_calls = 0

    def resolve(self, model_key: str) -> str:
        return f"scripted:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        response = InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            model_path=self.resolve(request.model_key),
            provider_name=self.provider_name,
            latency_ms=1.0,
        )
        if request.response_format == ResponseFormat.STRUCTURED_OUTPUT:
            self.structured_calls += 1
            payload = self._payloads[min(self._index, len(self._payloads) - 1)]
            self._index += 1
            response.output_structured = payload
            response.output_text = str(payload)
        else:
            response.output_text = "[scripted] ack"
        return response


class ToolThenStructuredManager:
    """Mimics a real provider: call bound tools first, then emit structured output.

    On a turn where tools are bound and no tool result is yet on the thread, it
    invokes each bound tool through the request's executor (so the typed agent's
    deps-injecting handler actually runs) and returns the tool-call exchange with
    NO final text — driving the runtime's agent loop to continue. On the following
    turn (tool result present) it returns the scripted structured payload.
    """

    def __init__(self, payload: dict[str, Any]) -> None:
        self.provider_name = "tool-then-structured"
        self._payload = payload
        self.tool_turns = 0

    def resolve(self, model_key: str) -> str:
        return f"tts:{model_key}"

    @staticmethod
    def _has_tool_result(request: InferenceRequest) -> bool:
        return any(m.role == "tool" for m in request.messages)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        response = InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            model_path=self.resolve(request.model_key),
            provider_name=self.provider_name,
            latency_ms=1.0,
        )
        if request.bound_tools and not self._has_tool_result(request):
            self.tool_turns += 1
            for tool in request.bound_tools:
                args = {"city": "Lhasa"}
                call = ToolCallRecord(
                    tool_call_id=f"tc_{tool.name}", tool_name=tool.name, args=args
                )
                response.tool_calls.append(call)
                if request.tool_executor is not None:
                    ret = await request.tool_executor(tool.name, args)
                    response.tool_returns.append(
                        ret.model_copy(update={"tool_call_id": call.tool_call_id})
                    )
            response.output_text = ""
            return response
        response.output_structured = self._payload
        response.output_text = str(self._payload)
        return response


def _scripted_runtime(
    payloads: list[Any],
    *,
    on_event: Any = None,
) -> tuple[Any, StorageService, ScriptedClientManager]:
    """Build a runtime whose inference is the scripted manager (shared storage)."""
    storage = StorageService()
    manager = ScriptedClientManager(payloads)
    inference = InferenceService(manager, event_sink=storage)
    runtime, _inf, _tools = build_runtime(
        inference=inference,
        storage=storage,
        on_event=on_event,
    )
    return runtime, storage, manager


# --------------------------------------------------------------------------- #
# compile_tool_schema                                                          #
# --------------------------------------------------------------------------- #
def test_compile_tool_schema_from_signature() -> None:
    """A typed function's signature compiles to a JSON Schema; ctx is excluded."""

    async def lookup(
        ctx: RunContext[Deps], city: str, units: str = "c"
    ) -> dict[str, Any]:
        return {}

    schema, takes_context = compile_tool_schema(lookup)
    assert takes_context is True
    props = schema["properties"]
    assert set(props) == {"city", "units"}  # ctx is NOT advertised
    assert props["city"]["type"] == "string"
    assert schema["required"] == ["city"]  # units has a default -> optional


def test_compile_tool_schema_without_context() -> None:
    """A tool with no RunContext param compiles all params as args."""

    async def add(a: int, b: int) -> int:
        return a + b

    schema, takes_context = compile_tool_schema(add)
    assert takes_context is False
    assert set(schema["properties"]) == {"a", "b"}
    assert sorted(schema["required"]) == ["a", "b"]


# --------------------------------------------------------------------------- #
# Offline default: zero-config, no keys, no network                            #
# --------------------------------------------------------------------------- #
def test_offline_default_returns_validated_output() -> None:
    """A typed agent runs end-to-end on the offline stub and returns OutputT."""
    runtime, _inf, _tools = build_runtime()
    agent: TypedAgent[Deps, Weather] = TypedAgent(
        runtime, output_type=Weather, name="weather"
    )
    result = run_async(agent.run("Weather in Pokhara?", deps=Deps(api_key="k")))
    assert isinstance(result.output, Weather)
    assert result.validation_attempts == 1
    assert result.repaired is False
    # The output schema is the pydantic model's schema, sent to the model.
    assert set(agent.output_schema["properties"]) == {"city", "summary", "temp_c"}


def test_tool_receives_typed_deps() -> None:
    """A registered tool sees the exact typed deps via RunContext at call time.

    Uses a provider that calls tools first then emits structured output (the real
    agentic flow); the offline stub cannot do both in one turn, so this asserts the
    deps-injection seam against a tool-then-structured manager.
    """
    storage = StorageService()
    manager = ToolThenStructuredManager(
        {"city": "Lhasa", "summary": "Snowy", "temp_c": -3}
    )
    inference = InferenceService(manager, event_sink=storage)
    runtime, _inf, _tools = build_runtime(inference=inference, storage=storage)
    agent: TypedAgent[Deps, Weather] = TypedAgent(runtime, output_type=Weather)
    seen: dict[str, Any] = {}

    @agent.tool
    async def lookup(ctx: RunContext[Deps], city: str) -> dict[str, Any]:
        """Look up weather."""
        seen["api_key"] = ctx.deps.api_key
        seen["city"] = city
        return {"city": city, "temp": 19}

    result = run_async(agent.run("Weather in Lhasa?", deps=Deps(api_key="topsecret")))
    assert seen["api_key"] == "topsecret"  # typed deps reached the tool
    assert seen["city"] == "Lhasa"
    assert manager.tool_turns == 1  # the loop ran a tool turn before structured output
    assert result.output.summary == "Snowy"


def test_typed_tool_read_only_not_asserted_by_default() -> None:
    """A typed tool is NOT declared authoritative read-only just because it is typed.

    Regression: a mutating ``@agent.tool`` (e.g. a writer) must stay fail-CLOSED at
    the parallel/retry/authz gate. Left unset, ``read_only`` is name-inferred and
    NON-authoritative, so it is not hoisted into a concurrent read-batch, not
    re-fired on timeout, and does not waive its ``:write`` grant.
    """
    runtime, _inf, tools = build_runtime()
    agent: TypedAgent[Deps, Weather] = TypedAgent(runtime, output_type=Weather)

    @agent.tool
    async def create_order(ctx: RunContext[Deps], sku: str) -> dict[str, Any]:
        """Create an order (mutates state)."""
        return {"sku": sku}

    bound = agent._bind_tools(Deps(api_key="k"))
    try:
        definition = tools.registry.get("create_order")
        assert definition is not None
        assert definition.read_only is None  # name-inferred, not asserted True
        assert definition.read_only_authoritative is False  # NOT the trust anchor
    finally:
        agent._unbind_tools(bound)


def test_typed_tool_read_only_declared_true_is_authoritative() -> None:
    """An explicit ``read_only=True`` typed tool IS authoritative (opt-in trust)."""
    runtime, _inf, tools = build_runtime()
    agent: TypedAgent[Deps, Weather] = TypedAgent(runtime, output_type=Weather)

    @agent.tool(read_only=True)
    async def lookup(ctx: RunContext[Deps], city: str) -> dict[str, Any]:
        """Look up weather (no side effect)."""
        return {"city": city}

    bound = agent._bind_tools(Deps(api_key="k"))
    try:
        definition = tools.registry.get("lookup")
        assert definition is not None
        assert definition.read_only is True
        assert definition.read_only_authoritative is True
    finally:
        agent._unbind_tools(bound)


def test_typed_tool_read_only_declared_false_is_writer() -> None:
    """An explicit ``read_only=False`` typed tool is a declared writer (non-authoritative)."""
    runtime, _inf, tools = build_runtime()
    agent: TypedAgent[Deps, Weather] = TypedAgent(runtime, output_type=Weather)

    @agent.tool(read_only=False)
    async def send_email(ctx: RunContext[Deps], to: str) -> dict[str, Any]:
        """Send an email (mutates state)."""
        return {"to": to}

    bound = agent._bind_tools(Deps(api_key="k"))
    try:
        definition = tools.registry.get("send_email")
        assert definition is not None
        assert definition.read_only is False
        assert definition.read_only_authoritative is False
    finally:
        agent._unbind_tools(bound)


def test_tool_is_unbound_after_run() -> None:
    """Tools registered for a run are removed from the shared registry afterwards."""
    runtime, _inf, tools = build_runtime()
    agent: TypedAgent[Deps, Weather] = TypedAgent(runtime, output_type=Weather)

    @agent.tool
    async def lookup(ctx: RunContext[Deps], city: str) -> dict[str, Any]:
        return {"city": city}

    assert tools.registry.get("lookup") is None
    run_async(agent.run("x", deps=Deps(api_key="k")))
    # After the run the closed-over handler is gone (no cross-run leakage).
    assert tools.registry.get("lookup") is None


# --------------------------------------------------------------------------- #
# Validation + bounded repair retries on the real runtime loop                 #
# --------------------------------------------------------------------------- #
def test_invalid_then_valid_output_repairs() -> None:
    """An invalid structured payload triggers a bounded retry that then validates."""
    bad = {"city": "Kathmandu", "summary": "", "temp_c": "not-an-int"}
    good = {"city": "Kathmandu", "summary": "Clear skies", "temp_c": 24}
    runtime, _storage, manager = _scripted_runtime([bad, good])
    agent: TypedAgent[Deps, Weather] = TypedAgent(
        runtime, output_type=Weather, output_retries=2
    )
    result = run_async(agent.run("Weather?", deps=Deps(api_key="k")))
    assert isinstance(result.output, Weather)
    assert result.output.temp_c == 24
    assert result.output.summary == "Clear skies"
    assert result.validation_attempts == 2
    assert result.repaired is True
    assert manager.structured_calls == 2  # one bad turn, one repaired turn


def test_exhausted_retries_raises() -> None:
    """Persistently invalid output raises HimmyError after the retry budget."""
    bad = {"city": "X", "summary": "", "temp_c": "nope"}
    runtime, _storage, manager = _scripted_runtime([bad])
    agent: TypedAgent[Deps, Weather] = TypedAgent(
        runtime, output_type=Weather, output_retries=1
    )
    with pytest.raises(HimmyError) as excinfo:
        run_async(agent.run("Weather?", deps=Deps(api_key="k")))
    assert "did not validate" in str(excinfo.value)
    # output_retries=1 -> first attempt + one retry = 2 structured calls.
    assert manager.structured_calls == 2


def test_zero_retries_raises_on_first_failure() -> None:
    """output_retries=0 means a single attempt: invalid output raises immediately."""
    bad = {"city": "X", "summary": "", "temp_c": "nope"}
    runtime, _storage, manager = _scripted_runtime([bad, bad])
    agent: TypedAgent[Deps, Weather] = TypedAgent(
        runtime, output_type=Weather, output_retries=0
    )
    with pytest.raises(HimmyError):
        run_async(agent.run("Weather?", deps=Deps(api_key="k")))
    assert manager.structured_calls == 1


# --------------------------------------------------------------------------- #
# Provenance: TYPED_OUTPUT_* events flow through the audit spine               #
# --------------------------------------------------------------------------- #
def test_validated_event_emitted_and_audited() -> None:
    """A successful run emits TYPED_OUTPUT_VALIDATED through the event stream."""
    captured: list[RunEvent] = []

    async def on_event(event: RunEvent) -> None:
        captured.append(event)

    runtime, _inf, _tools = build_runtime(on_event=on_event)
    agent: TypedAgent[Deps, Weather] = TypedAgent(runtime, output_type=Weather)
    run_async(agent.run("Weather?", deps=Deps(api_key="k")))

    validated = [
        e for e in captured if e.event_type == EventType.TYPED_OUTPUT_VALIDATED
    ]
    assert len(validated) == 1
    assert validated[0].payload["output_type"] == "Weather"
    assert validated[0].agent_id == agent.persona.agent_id


def test_repair_event_emitted_on_retry() -> None:
    """A repaired run emits TYPED_OUTPUT_REPAIRED before the retry turn."""
    captured: list[RunEvent] = []

    async def on_event(event: RunEvent) -> None:
        captured.append(event)

    bad = {"city": "X", "summary": "", "temp_c": "nope"}
    good = {"city": "X", "summary": "Sunny", "temp_c": 30}
    runtime, _storage, _manager = _scripted_runtime([bad, good], on_event=on_event)
    agent: TypedAgent[Deps, Weather] = TypedAgent(
        runtime, output_type=Weather, output_retries=2
    )
    run_async(agent.run("Weather?", deps=Deps(api_key="k")))

    repaired = [e for e in captured if e.event_type == EventType.TYPED_OUTPUT_REPAIRED]
    validated = [
        e for e in captured if e.event_type == EventType.TYPED_OUTPUT_VALIDATED
    ]
    assert len(repaired) == 1
    assert repaired[0].error  # the validation failure is recorded on the event
    assert len(validated) == 1


def test_typed_events_registered_on_entity_spine() -> None:
    """TYPED_OUTPUT_VALIDATED is projected into the EntityRecord audit spine."""
    runtime, _inf, _tools = build_runtime()
    registry = runtime.entity_registry
    assert registry is not None
    agent: TypedAgent[Deps, Weather] = TypedAgent(runtime, output_type=Weather)
    run_async(agent.run("Weather?", deps=Deps(api_key="k")))

    run_events = registry.list_by_kind("run_event")
    typed = [
        r
        for r in run_events
        if r.payload.get("event_type") == EventType.TYPED_OUTPUT_VALIDATED.value
    ]
    assert typed, "expected a TYPED_OUTPUT_VALIDATED run_event record on the spine"


# --------------------------------------------------------------------------- #
# Construction guards                                                          #
# --------------------------------------------------------------------------- #
def test_non_basemodel_output_type_rejected() -> None:
    """output_type must be a pydantic BaseModel subclass."""
    runtime, _inf, _tools = build_runtime()
    with pytest.raises(HimmyError):
        TypedAgent(runtime, output_type=dict)  # type: ignore[type-var]


def test_negative_retries_rejected() -> None:
    """output_retries must be non-negative."""
    runtime, _inf, _tools = build_runtime()
    with pytest.raises(HimmyError):
        TypedAgent(runtime, output_type=Weather, output_retries=-1)


def test_lazy_top_level_export() -> None:
    """TypedAgent is exposed at the package top level (lazy)."""
    import himmy

    assert himmy.TypedAgent is TypedAgent
    assert himmy.RunContext is RunContext
