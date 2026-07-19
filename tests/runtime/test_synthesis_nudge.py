"""Tier 1.1 — the synthesis nudge: a forced final turn when a tool loop empties out."""

from __future__ import annotations

from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
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


class _Scripted:
    """Returns a fixed sequence of turns; ``tools=True`` carries a call+return."""

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self._specs = specs
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return "scripted"

    async def generate(self, request: Any) -> InferenceResponse:
        spec = self._specs[min(self.calls, len(self._specs) - 1)]
        self.calls += 1
        cid = f"c{self.calls}"
        has = bool(spec.get("tools"))
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=spec.get("text", ""),
            tool_calls=[ToolCallRecord(tool_call_id=cid, tool_name="probe", args={})]
            if has
            else [],
            tool_returns=[
                ToolReturnRecord(tool_call_id=cid, tool_name="probe", content={"ok": 1})
            ]
            if has
            else [],
            input_tokens=1,
            output_tokens=1,
        )


def _rt(specs: list[dict[str, Any]]) -> tuple[SingleAgentRuntime, _Scripted]:
    mgr = _Scripted(specs)
    return (
        SingleAgentRuntime(
            inference_service=InferenceService(mgr), memory_store=StorageService()
        ),
        mgr,
    )


def _go():
    return Persona(name="a"), Task(title="t", prompt="answer me")


def test_synthesizes_when_loop_ends_empty_after_tools() -> None:
    # turn1: tool call (empty) -> turn2: no tools + EMPTY -> nudge -> answer.
    rt, mgr = _rt(
        [
            {"tools": True, "text": ""},
            {"tools": False, "text": ""},
            {"tools": False, "text": "There are 30 chickens."},
        ]
    )
    persona, task = _go()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=6))
    assert result.stopped_reason == "synthesized"
    assert result.final.output_text == "There are 30 chickens."
    assert mgr.calls == 3  # the extra synthesis turn ran


def test_no_synthesis_when_already_answered() -> None:
    rt, mgr = _rt([{"tools": True, "text": ""}, {"tools": False, "text": "done"}])
    result = run_async(rt.run_agent_loop(_go()[0], _go()[1], max_turns=6))
    assert result.stopped_reason == "final"
    assert result.final.output_text == "done"
    assert mgr.calls == 2  # no extra turn


def test_no_synthesis_when_no_tools_were_used() -> None:
    # Empty answer but the model never used a tool → nothing to synthesize from.
    rt, mgr = _rt([{"tools": False, "text": ""}])
    result = run_async(rt.run_agent_loop(_go()[0], _go()[1], max_turns=6))
    assert result.stopped_reason == "final"
    assert mgr.calls == 1


def test_synthesis_can_be_disabled() -> None:
    rt, mgr = _rt([{"tools": True, "text": ""}, {"tools": False, "text": ""}])
    result = run_async(
        rt.run_agent_loop(_go()[0], _go()[1], max_turns=6, synthesize_empty=False)
    )
    assert result.stopped_reason == "final"
    assert (result.final.output_text or "") == ""
    assert mgr.calls == 2  # no synthesis turn


def test_synthesis_threads_governed_subject_and_model_key() -> None:
    # A governed run pins a subject + custom model_key + output_schema. The rescue
    # turn must KEEP model_key and context_subject_id (else its spine records
    # fail-closed drop / the wrong model answers) but DROP output_schema (the
    # free-text rescue answer must not be forced through structured validation).
    rt, mgr = _rt(
        [
            {"tools": True, "text": ""},
            {"tools": False, "text": ""},
            {"tools": False, "text": "42 chickens."},
        ]
    )
    seen: list[dict[str, Any]] = []
    original = rt.run_task_detailed

    async def _spy(persona, task, thread=None, **kw):  # type: ignore[no-untyped-def]
        seen.append({"title": task.title, "context": dict(task.context or {})})
        return await original(persona, task, thread=thread, **kw)

    rt.run_task_detailed = _spy  # type: ignore[method-assign]
    persona = Persona(name="a")
    task = Task(
        title="t",
        prompt="answer me",
        context={
            "context_subject_id": "subj-1",
            "model_key": "custom-key",
            "output_schema": {"type": "object"},
        },
    )
    result = run_async(rt.run_agent_loop(persona, task, max_turns=6))
    assert result.stopped_reason == "synthesized"
    nudge = next(s for s in seen if s["title"] == "synthesis")
    assert nudge["context"]["context_subject_id"] == "subj-1"
    assert nudge["context"]["model_key"] == "custom-key"
    assert nudge["context"]["tool_names"] == []
    assert "output_schema" not in nudge["context"]
