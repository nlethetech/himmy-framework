"""Tests for the runtime-owned agentic loop (run_agent_loop)."""

from __future__ import annotations

from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


class _ScriptedManager:
    """A client manager returning a fixed sequence of turns (last repeats).

    Each spec is ``{"tools": bool, "text": str, "cost": float}``; a turn with
    ``tools=True`` carries a tool call/return (so the loop continues), ``False``
    is a final answer.
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


def _runtime(manager: _ScriptedManager, on_event: Any = None) -> SingleAgentRuntime:
    return SingleAgentRuntime(
        inference_service=InferenceService(manager),
        memory_store=StorageService(),
        on_event=on_event,
    )


def _go() -> tuple[Persona, Task]:
    return Persona(name="analyst"), Task(title="t", prompt="investigate")


def test_loop_stops_when_model_answers_without_tools() -> None:
    """Two tool turns then a final answer ends the loop with reason 'final'."""
    rt = _runtime(
        _ScriptedManager(
            [{"tools": True, "text": "act"}, {"tools": False, "text": "done"}]
        )
    )
    persona, task = _go()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=6))
    assert result.turn_count == 2
    assert result.stopped_reason == "final"
    assert result.final.output_text == "done"


def test_loop_bounds_at_max_turns() -> None:
    """A model that always calls tools is bounded by max_turns."""
    rt = _runtime(_ScriptedManager([{"tools": True, "text": "act"}]))  # always tools
    persona, task = _go()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=3))
    assert result.turn_count == 3
    assert result.stopped_reason == "max_turns"


def test_loop_single_turn_when_no_tools() -> None:
    """A first-turn final answer runs exactly one turn."""
    rt = _runtime(_ScriptedManager([{"tools": False, "text": "answer"}]))
    persona, task = _go()
    result = run_async(rt.run_agent_loop(persona, task))
    assert result.turn_count == 1
    assert result.stopped_reason == "final"
    assert result.final.output_text == "answer"


def test_loop_stops_on_cost_budget() -> None:
    """Accrued cost across turns trips the budget ceiling."""
    rt = _runtime(_ScriptedManager([{"tools": True, "text": "act", "cost": 1.0}]))
    persona, task = _go()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=10, cost_budget=2.5))
    assert result.stopped_reason == "budget"
    assert result.total_cost >= 2.5


def test_loop_rejects_bad_max_turns() -> None:
    """max_turns must be >= 1."""
    rt = _runtime(_ScriptedManager([{"tools": False, "text": "x"}]))
    persona, task = _go()
    with pytest.raises(HimmyError):
        run_async(rt.run_agent_loop(persona, task, max_turns=0))


def test_loop_emits_a_turn_completed_event_per_turn() -> None:
    """Each model turn emits an AGENT_TURN_COMPLETED event with its index."""
    events: list[RunEvent] = []

    async def collect(event: RunEvent) -> None:
        events.append(event)

    rt = _runtime(
        _ScriptedManager([{"tools": True, "text": "a"}, {"tools": False, "text": "b"}]),
        on_event=collect,
    )
    persona, task = _go()
    run_async(rt.run_agent_loop(persona, task, max_turns=6))
    completed = [e for e in events if e.event_type == EventType.AGENT_TURN_COMPLETED]
    assert [e.payload["turn"] for e in completed] == [1, 2]
