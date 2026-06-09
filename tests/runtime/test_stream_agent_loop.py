"""Tests for the streamed multi-turn tool loop (stream_agent_loop).

These exercise the opt-in async generator that streams text deltas + tool-call /
tool-result / turn-end events THROUGH the whole tool loop (not just one turn),
plus a terminal ``done`` delta carrying the :class:`AgentLoopResult`. Everything
runs offline against the deterministic :class:`StubClientManager` / a small
scripted manager, so the assertions are byte-stable.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.runtime import (
    AgentLoopResult,
    InMemoryCheckpointStore,
    SingleAgentRuntime,
)
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService, StreamDelta
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


# --------------------------------------------------------------------------- helpers
async def _collect(gen: Any) -> list[StreamDelta]:
    """Drain an async generator of StreamDeltas into a list."""
    out: list[StreamDelta] = []
    async for delta in gen:
        out.append(delta)
    return out


def _text(deltas: list[StreamDelta]) -> str:
    """Reassemble the streamed text (text deltas have event_type=None)."""
    return "".join(d.delta for d in deltas if d.event_type is None and not d.done)


def _events(deltas: list[StreamDelta], kind: str) -> list[StreamDelta]:
    return [d for d in deltas if d.event_type == kind]


def _result(deltas: list[StreamDelta]) -> AgentLoopResult:
    done = [d for d in deltas if d.done]
    assert len(done) == 1, "exactly one terminal done delta"
    result = done[0].event_payload["result"]  # type: ignore[index]
    assert isinstance(result, AgentLoopResult)
    return result


class _ScriptedManager:
    """A client manager returning a fixed sequence of turns (last repeats).

    No ``generate_stream`` — so ``run_stream`` uses the deterministic offline
    fallback that buffers via ``run`` then chunks at 24 chars. A spec with
    ``tools=True`` carries a tool call/return (loop continues); ``False`` is a
    final answer.
    """

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self._specs = specs
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return f"scripted:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        spec = self._specs[min(self.calls, len(self._specs) - 1)]
        self.calls += 1
        cid = f"c{self.calls}"
        has_tools = bool(spec.get("tools"))
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=spec.get("text", ""),
            tool_calls=(
                [ToolCallRecord(tool_call_id=cid, tool_name="probe", args={})]
                if has_tools
                else []
            ),
            tool_returns=(
                [
                    ToolReturnRecord(
                        tool_call_id=cid, tool_name="probe", content={"ok": True}
                    )
                ]
                if has_tools
                else []
            ),
            cost=float(spec.get("cost", 0.0)),
            input_tokens=1,
            output_tokens=1,
        )


def _scripted_runtime(specs: list[dict[str, Any]]) -> SingleAgentRuntime:
    return SingleAgentRuntime(
        inference_service=InferenceService(_ScriptedManager(specs)),
        memory_store=StorageService(),
    )


def _stub_runtime_with_tool(*, requires_approval: bool = False) -> SingleAgentRuntime:
    """A runtime wired to the real StubClientManager + one registered tool."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="lookup",
        handler=lambda args: {"value": 42},
        args_json_schema={"type": "object", "properties": {}},
        requires_approval=requires_approval,
    )
    storage = StorageService()
    return SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        tool_service=ToolService(registry, event_sink=storage),
        checkpoint_store=InMemoryCheckpointStore(),
    )


def _go() -> tuple[Persona, Task]:
    return Persona(name="analyst"), Task(title="t", prompt="investigate")


# --------------------------------------------------------------------------- tests
def test_single_turn_no_tools_streams_text_and_turn_end() -> None:
    """A first-turn final answer streams text deltas, a turn_end, and a done."""
    rt = _scripted_runtime([{"tools": False, "text": "the final answer is 42"}])
    persona, task = _go()
    deltas = run_async(_collect(rt.stream_agent_loop(persona, task)))

    assert _text(deltas) == "the final answer is 42"
    assert len(_events(deltas, "turn_end")) == 1
    # No tools were called, so no tool events.
    assert _events(deltas, "tool_call") == []
    assert _events(deltas, "tool_result") == []
    result = _result(deltas)
    assert result.turn_count == 1
    assert result.stopped_reason == "final"
    assert result.final.output_text == "the final answer is 42"


def test_reassembled_text_equals_full_response() -> None:
    """The streamed text deltas concatenate back to the full reply (24-char chunks)."""
    long_text = "x" * 100  # forces multiple 24-char chunks
    rt = _scripted_runtime([{"tools": False, "text": long_text}])
    persona, task = _go()
    deltas = run_async(_collect(rt.stream_agent_loop(persona, task)))

    text_deltas = [d for d in deltas if d.event_type is None and not d.done]
    # Stub/fallback chunk size is 24 chars.
    assert all(len(d.delta) <= 24 for d in text_deltas)
    assert len(text_deltas) > 1  # really chunked, not one blob
    assert _text(deltas) == long_text


def test_two_turn_with_tools_yields_ordered_events() -> None:
    """A tool turn then a final answer yields text -> tool_call -> tool_result -> turn_end."""
    # turn 1 calls a tool; turn 2 answers (no tools).
    rt = _scripted_runtime(
        [{"tools": True, "text": "let me look that up"}, {"tools": False, "text": "ok"}]
    )
    persona, task = _go()
    deltas = run_async(_collect(rt.stream_agent_loop(persona, task)))

    # The ordered, non-done event tags across the whole loop.
    tags: list[str] = []
    for d in deltas:
        if d.done:
            continue
        if d.event_type is None:
            tags.append("text")
        else:
            tags.append(d.event_type)

    # First turn: text, then its tool_call/tool_result, then turn_end.
    first_text = tags.index("text")
    first_call = tags.index("tool_call")
    first_result = tags.index("tool_result")
    first_turn_end = tags.index("turn_end")
    assert first_text < first_call < first_result < first_turn_end

    # Exactly one tool exchange happened (turn 1 only).
    assert len(_events(deltas, "tool_call")) == 1
    assert len(_events(deltas, "tool_result")) == 1
    assert len(_events(deltas, "turn_end")) == 2  # two turns

    call = _events(deltas, "tool_call")[0]
    res = _events(deltas, "tool_result")[0]
    assert call.event_payload["tool_name"] == "probe"  # type: ignore[index]
    assert res.event_payload["tool_name"] == "probe"  # type: ignore[index]
    assert res.event_payload["outcome"] == "success"  # type: ignore[index]

    result = _result(deltas)
    assert result.turn_count == 2
    assert result.stopped_reason == "final"


def test_real_stub_tools_replayed_onto_thread() -> None:
    """The first streamed turn's tool messages land on the thread (loop can continue)."""
    rt = _stub_runtime_with_tool()
    persona = Persona(name="analyst")
    task = Task(
        title="t",
        prompt="look it up",
        context={"tool_names": ["lookup"], "response_format": "AUTO_TOOLS"},
    )
    deltas = run_async(_collect(rt.stream_agent_loop(persona, task)))

    result = _result(deltas)
    tool_msgs = [m for m in result.thread.messages if m.role == MessageRole.TOOL]
    assert tool_msgs, "the streamed first turn replayed its tool exchange"
    # A tool_call + tool_result pair was surfaced for the real stub tool.
    calls = _events(deltas, "tool_call")
    assert any(c.event_payload["tool_name"] == "lookup" for c in calls)  # type: ignore[index]


def test_max_turns_honored_during_streaming() -> None:
    """A model that always calls tools is bounded by max_turns while streaming."""
    rt = _scripted_runtime([{"tools": True, "text": "act"}])  # always tools
    persona, task = _go()
    deltas = run_async(_collect(rt.stream_agent_loop(persona, task, max_turns=3)))
    result = _result(deltas)
    assert result.turn_count == 3
    assert result.stopped_reason == "max_turns"
    assert len(_events(deltas, "turn_end")) == 3


def test_cost_budget_honored_during_streaming() -> None:
    """Accrued cost across streamed turns trips the budget ceiling."""
    rt = _scripted_runtime([{"tools": True, "text": "act", "cost": 1.0}])
    persona, task = _go()
    deltas = run_async(
        _collect(rt.stream_agent_loop(persona, task, max_turns=10, cost_budget=2.5))
    )
    result = _result(deltas)
    assert result.stopped_reason == "budget"
    assert result.total_cost >= 2.5


def test_no_progress_honored_during_streaming() -> None:
    """Two identical tool turns stop with no_progress when opted in."""
    rt = _scripted_runtime([{"tools": True, "text": "spin"}])  # identical every turn
    persona, task = _go()
    deltas = run_async(
        _collect(
            rt.stream_agent_loop(persona, task, max_turns=10, stop_on_no_progress=True)
        )
    )
    result = _result(deltas)
    assert result.stopped_reason == "no_progress"
    # Stopped well before max_turns (after the 2nd identical turn).
    assert result.turn_count < 10


def test_hitl_approval_required_surfaces() -> None:
    """An approval-required tool pauses the streamed loop with a checkpoint."""
    rt = _stub_runtime_with_tool(requires_approval=True)
    persona = Persona(name="treasurer")
    task = Task(
        title="t",
        prompt="wire the money",
        context={"tool_names": ["lookup"], "response_format": "AUTO_TOOLS"},
    )
    deltas = run_async(
        _collect(rt.stream_agent_loop(persona, task, max_turns=5, hitl=True))
    )
    result = _result(deltas)
    assert result.stopped_reason == "awaiting_approval"
    assert result.checkpoint_id is not None


def test_hitl_requires_checkpoint_store() -> None:
    """hitl=True without a checkpoint_store is a configuration error."""
    rt = _scripted_runtime([{"tools": False, "text": "x"}])
    persona, task = _go()
    with pytest.raises(HimmyError):
        run_async(_collect(rt.stream_agent_loop(persona, task, hitl=True)))


def test_rejects_bad_max_turns() -> None:
    """max_turns must be >= 1."""
    rt = _scripted_runtime([{"tools": False, "text": "x"}])
    persona, task = _go()
    with pytest.raises(HimmyError):
        run_async(_collect(rt.stream_agent_loop(persona, task, max_turns=0)))


def test_stub_24_char_chunking_preserved_across_turns() -> None:
    """Continuation turns re-chunk at 24 chars; reassembly equals each turn's text."""
    turn1 = "a" * 50  # > 24 so it chunks
    turn2 = "b" * 60
    rt = _scripted_runtime(
        [{"tools": True, "text": turn1}, {"tools": False, "text": turn2}]
    )
    persona, task = _go()
    deltas = run_async(
        _collect(rt.stream_agent_loop(persona, task, synthesize_empty=False))
    )

    text_deltas = [d for d in deltas if d.event_type is None and not d.done]
    assert all(len(d.delta) <= 24 for d in text_deltas)
    # Both turns' text appear, in order, fully reassembled.
    assert _text(deltas) == turn1 + turn2
    result = _result(deltas)
    assert result.turn_count == 2


def test_run_agent_loop_unchanged_by_streaming_surface() -> None:
    """The non-streaming loop still returns the identical typed outcome."""
    rt = _scripted_runtime(
        [{"tools": True, "text": "act"}, {"tools": False, "text": "done"}]
    )
    persona, task = _go()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=6))
    assert result.turn_count == 2
    assert result.stopped_reason == "final"
    assert result.final.output_text == "done"
