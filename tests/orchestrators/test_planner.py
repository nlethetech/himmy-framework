"""Tests for the planner + reflection orchestrators (offline stub)."""

from __future__ import annotations

from himmy import build_runtime
from himmy.agents.personas.persona import Persona
from himmy.orchestrators import PlannerOrchestrator, reflect
from himmy.orchestrators.planner import _steps_from_text
from tests.conftest import run_async


def test_steps_from_text_parses_numbered_and_bulleted() -> None:
    text = "Here is the plan:\n1. First step\n2) Second step\n- Third step\n* Fourth"
    steps = _steps_from_text(text)
    assert steps == ["First step", "Second step", "Third step", "Fourth"]
    assert _steps_from_text(None) == []
    assert _steps_from_text("no steps here") == []


def test_planner_runs_plan_execute_synthesize() -> None:
    """The planner produces a plan, executes steps, and synthesizes an answer."""
    runtime, _inf, _tools = build_runtime()
    result = run_async(
        PlannerOrchestrator(runtime, max_steps=3).run(
            "summarize permaculture", Persona(name="planner")
        )
    )
    assert result.goal == "summarize permaculture"
    assert isinstance(result.plan, list)
    assert len(result.step_results) == len(result.plan)
    assert result.output_text  # a synthesized final answer
    # the shared thread accumulated plan + steps + synthesis turns
    assert len(result.thread.messages) > 2


def test_planner_respects_max_steps() -> None:
    runtime, _inf, _tools = build_runtime()
    result = run_async(PlannerOrchestrator(runtime, max_steps=2).run("do a thing"))
    assert len(result.plan) <= 2


def test_reflect_returns_improved_text() -> None:
    runtime, _inf, _tools = build_runtime()
    improved = run_async(reflect(runtime, "a rough draft about bees"))
    assert isinstance(improved, str)
    assert improved  # non-empty
