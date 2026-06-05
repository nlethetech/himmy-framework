"""Tests for the faithful stub: terminates after one tool turn; surfaces tool errors."""

from __future__ import annotations

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.services.inference.models import LLMConfig, ResponseFormat
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from tests.conftest import run_async


def test_stub_tool_agent_terminates_after_one_tool_turn() -> None:
    """The stub calls a tool once, then (seeing the result) answers — like a real model."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="echo",
        handler=lambda args: {"ok": True},
        description="echo",
        args_json_schema={"type": "object", "properties": {}},
    )
    runtime, _inf, _tools = build_runtime(tool_registry=registry)
    loop = run_async(
        runtime.run_agent_loop(
            Persona(name="a"),
            Task(title="t", prompt="use echo", context={"tool_names": ["echo"]}),
            max_turns=8,
            llm_config=LLMConfig(response_format=ResponseFormat.AUTO_TOOLS),
        )
    )
    assert loop.stopped_reason == "final"
    assert loop.turn_count == 2
    assert loop.turns[0].tool_calls  # turn 0 called the tool
    assert not loop.turns[1].tool_calls  # turn 1 answered with no tool


def test_failed_tool_surfaces_error_in_thread() -> None:
    """A failing tool produces an `ERROR:` TOOL message the model can see."""

    def boom(args: dict) -> dict:
        raise RuntimeError("kaboom")

    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="boom",
        handler=boom,
        description="always fails",
        args_json_schema={"type": "object", "properties": {}},
    )
    runtime, _inf, _tools = build_runtime(tool_registry=registry)
    loop = run_async(
        runtime.run_agent_loop(
            Persona(name="a"),
            Task(title="t", prompt="use boom", context={"tool_names": ["boom"]}),
            max_turns=4,
            llm_config=LLMConfig(response_format=ResponseFormat.AUTO_TOOLS),
        )
    )
    tool_messages = [m for m in loop.thread.messages if m.role == MessageRole.TOOL]
    assert tool_messages
    assert any("ERROR:" in m.content for m in tool_messages)
