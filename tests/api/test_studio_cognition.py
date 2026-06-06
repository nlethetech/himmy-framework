"""Tests for Studio's live cognition stream (think → act → observe).

Covers the event → frame mapping (:class:`_Cognition`), read/write intent tagging,
agent/delegate tracking, and round-trip persistence of the recorded steps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_service as ss
from himmy.api.studio_runs import reset_run_store
from himmy.core.events import EventType, RunEvent


def _tool_event(name: str, *, outcome: str = "success", **payload) -> RunEvent:
    et = EventType.TOOL_COMPLETED if outcome == "success" else EventType.TOOL_FAILED
    return RunEvent(
        event_type=et,
        payload={"tool_name": name, "tool_outcome": outcome, **payload},
    )


def test_tool_frame_carries_args_result_intent_latency() -> None:
    cog = ss._Cognition({"egg_totals": True}, "livestock")
    frames = cog.frames(
        _tool_event(
            "egg_totals",
            tool_args={"days": 7},
            result="245 eggs",
            latency_ms=12.5,
        )
    )
    assert len(frames) == 1
    f = frames[0]
    assert f["type"] == "tool"
    assert f["name"] == "egg_totals"
    assert f["args"] == {"days": 7}
    assert f["result"] == "245 eggs"
    assert f["read_only"] is True
    assert f["latency_ms"] == 12.5
    # the step is recorded for persistence too
    assert [s["kind"] for s in cog.steps] == ["tool"]
    assert cog.tools_used == ["egg_totals"]


def test_write_intent_falls_back_to_name_heuristic() -> None:
    # Registry has no read_only flag → classify by name (log_* is a write).
    cog = ss._Cognition({}, "livestock")
    frame = cog.frames(_tool_event("log_eggs", tool_args={"count": 12}))[0]
    assert frame["read_only"] is False


def test_failed_tool_marks_outcome() -> None:
    cog = ss._Cognition({"pond_status": True}, "pond")
    frame = cog.frames(_tool_event("pond_status", outcome="error"))[0]
    assert frame["outcome"] == "error"


def test_synthetic_orchestration_tools_are_hidden() -> None:
    cog = ss._Cognition({}, "manager")
    assert cog.frames(_tool_event("ask_livestock")) == []
    assert cog.frames(_tool_event("transfer_to_pond")) == []
    assert cog.frames(_tool_event("final_answer")) == []
    assert cog.steps == []


def test_agent_switch_and_delegate_are_tracked() -> None:
    cog = ss._Cognition({}, "manager")
    # a worker starts running
    fr = cog.frames(
        RunEvent(
            event_type=EventType.AGENT_RUN_STARTED,
            payload={"persona_name": "livestock", "model_key": "ollama:qwen"},
        )
    )
    assert fr == [{"type": "agent", "name": "livestock", "model": "ollama:qwen"}]
    assert cog.active == "livestock"
    # the manager delegates to that worker
    fr = cog.frames(
        RunEvent(
            event_type=EventType.AGENT_DELEGATED,
            payload={"worker": "livestock", "task": "count eggs", "answer": "245"},
        )
    )
    assert fr[0]["type"] == "delegate"
    assert fr[0]["worker"] == "livestock"
    assert cog.delegate_answers == [("livestock", "245")]
    kinds = [s["kind"] for s in cog.steps]
    assert kinds == ["agent", "delegate"]


def test_recorded_steps_round_trip_through_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    reset_run_store()
    cog = ss._Cognition({"egg_totals": True}, "livestock")
    cog.frames(_tool_event("egg_totals", tool_args={"days": 7}, result="245"))

    spec = type(
        "S", (), {"name": "livestock", "provider": "ollama", "model": "default"}
    )()
    ss._record_run(
        run_id="run-cog-1",
        spec=spec,
        agent_path="livestock.yaml",
        provider="ollama",
        model=None,
        prompt="how many eggs?",
        history=[],
        output="245 eggs",
        tools=cog.tools_used,
        events=[],
        steps=cog.steps,
        status="ok",
        error=None,
        duration_ms=1.0,
        thread_id="t1",
    )

    from himmy.api.studio_runs import get_run_store

    run = get_run_store().get("run-cog-1")
    assert run is not None
    assert len(run.steps) == 1
    step = run.steps[0]
    assert step.kind == "tool"
    assert step.name == "egg_totals"
    assert step.args == {"days": 7}
    assert step.read_only is True
