"""Tests for agent-loop termination: final_answer tool + no-progress guard + streaming."""

from __future__ import annotations

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.runtime.termination import (
    calls_signature,
    final_answer_text,
    is_no_progress,
    register_final_answer_tool,
)
from himmy.services.inference.models import ToolCallRecord
from himmy.services.tools.registry import ToolRegistry
from tests.conftest import run_async


class _FakeResult:
    """Minimal stand-in exposing .tool_calls for the helper functions."""

    def __init__(self, calls: list[ToolCallRecord]) -> None:
        self.tool_calls = calls


def test_final_answer_text_extracts_answer() -> None:
    call = ToolCallRecord(
        tool_call_id="1", tool_name="final_answer", args={"answer": "done"}
    )
    assert final_answer_text(_FakeResult([call])) == "done"
    assert final_answer_text(_FakeResult([])) is None


def test_no_progress_detects_repeated_calls() -> None:
    c = ToolCallRecord(tool_call_id="1", tool_name="t", args={"x": 1})
    r1, r2 = _FakeResult([c]), _FakeResult([c])
    assert is_no_progress([r1, r2]) is True
    # Different args → progress.
    c2 = ToolCallRecord(tool_call_id="2", tool_name="t", args={"x": 2})
    assert is_no_progress([_FakeResult([c]), _FakeResult([c2])]) is False
    # Empty tool calls → not "no progress" (that's a final answer).
    assert is_no_progress([_FakeResult([]), _FakeResult([])]) is False


def test_calls_signature_is_order_insensitive_on_args() -> None:
    a = ToolCallRecord(tool_call_id="1", tool_name="t", args={"a": 1, "b": 2})
    b = ToolCallRecord(tool_call_id="2", tool_name="t", args={"b": 2, "a": 1})
    assert calls_signature(_FakeResult([a])) == calls_signature(_FakeResult([b]))


def test_register_final_answer_tool() -> None:
    registry = ToolRegistry()
    register_final_answer_tool(registry)
    assert registry.get("final_answer") is not None
    out = registry.handler_for("final_answer")({"answer": "hi"})
    assert out == {"answer": "hi"}


def test_loop_stops_on_final_answer() -> None:
    """An agent that only has final_answer ends the loop via that tool."""
    registry = ToolRegistry()
    register_final_answer_tool(registry)
    runtime, _inf, _tools = build_runtime(tool_registry=registry)
    task = Task(title="t", prompt="answer", context={"tool_names": ["final_answer"]})
    loop = run_async(runtime.run_agent_loop(Persona(name="a"), task, max_turns=8))
    assert loop.stopped_reason == "final_answer"


def test_loop_stops_on_no_progress_when_opted_in() -> None:
    """With stop_on_no_progress, a repeated identical tool call halts the loop."""
    registry = ToolRegistry()
    from himmy.services.tools.registry import register_local_tool

    register_local_tool(
        registry,
        name="ping",
        handler=lambda args: {"ok": True},
        description="no-op",
        args_json_schema={"type": "object", "properties": {}},
    )
    runtime, _inf, _tools = build_runtime(tool_registry=registry)
    task = Task(title="t", prompt="go", context={"tool_names": ["ping"]})
    loop = run_async(
        runtime.run_agent_loop(
            Persona(name="a"), task, max_turns=10, stop_on_no_progress=True
        )
    )
    assert loop.stopped_reason == "no_progress"
    assert loop.turn_count < 10


def test_stream_task_chunks_reassemble() -> None:
    """Streaming deltas reassemble into the full response text."""
    runtime, _inf, _tools = build_runtime()

    async def _collect() -> tuple[str, str]:
        parts, final = [], ""
        async for d in runtime.stream_task(
            Persona(name="a"), Task(title="t", prompt="hello world")
        ):
            parts.append(d.delta)
            if d.done and d.response is not None:
                final = d.response.output_text or ""
        return "".join(parts), final

    joined, final = run_async(_collect())
    assert joined == final
    assert final  # non-empty stub reply
