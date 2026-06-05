"""Plan-and-execute orchestration: decompose a goal, work the steps, then synthesize.

A reactive tool loop answers in one shot; a planner first asks the model for a *plan* (an
ordered list of steps), executes each step over a shared thread (so later steps see
earlier results), and produces a final synthesis. This makes multi-step goals tractable
and is the agentic counterpart to :class:`~himmy.orchestrators.MultiAgentOrchestrator`.
It runs on any provider — including the offline stub, which returns a deterministic plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread
from himmy.agents.personas.persona import Persona
from himmy.runtime.single_agent import RunResult, SingleAgentRuntime
from himmy.services.inference.models import LLMConfig, ResponseFormat

_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ordered steps that accomplish the goal.",
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}

# Matches a numbered or bulleted step line (fallback when structured output is absent,
# e.g. on providers like Ollama that don't honor output_json_schema).
_STEP_LINE_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+(.*\S)")


def _steps_from_text(text: str | None) -> list[str]:
    """Parse numbered/bulleted step lines from a plain-text plan (provider fallback)."""
    if not text:
        return []
    return [
        m.group(1).strip()
        for line in text.splitlines()
        if (m := _STEP_LINE_RE.match(line))
    ]


@dataclass
class PlanResult:
    """The outcome of a plan-and-execute run."""

    thread: ChatThread
    goal: str
    plan: list[str] = field(default_factory=list)
    step_results: list[RunResult] = field(default_factory=list)
    output_text: str | None = None
    total_cost: float = 0.0


class PlannerOrchestrator:
    """Plan a goal into steps, execute each, and synthesize a final answer."""

    def __init__(
        self,
        runtime: SingleAgentRuntime,
        *,
        max_steps: int = 5,
        default_temperature: float = 0.2,
    ) -> None:
        """Wire the runtime; ``max_steps`` caps how many plan steps are executed."""
        self._runtime = runtime
        self._max_steps = max_steps
        self._temperature = default_temperature

    async def run(
        self,
        goal: str,
        persona: Persona | None = None,
        *,
        tool_names: list[str] | None = None,
        model_key: str = "default",
    ) -> PlanResult:
        """Plan ``goal``, execute the steps, and return the synthesized answer."""
        persona = persona or Persona(
            name="planner", description="You plan, then execute, then summarize."
        )
        thread = ChatThread(agent_id=persona.agent_id)

        # 1. Plan: ask for an ordered list of steps (structured output).
        plan_result = await self._runtime.run_task_detailed(
            persona,
            Task(
                title="plan",
                prompt=f"Make a short, ordered plan (max {self._max_steps} steps) "
                f"to accomplish this goal:\n{goal}",
                context={"output_schema": _PLAN_SCHEMA, "model_key": model_key},
            ),
            thread=thread,
            llm_config=LLMConfig(
                model_key=model_key,
                output_json_schema=_PLAN_SCHEMA,
                temperature=self._temperature,
            ),
        )
        structured = plan_result.output_structured or {}
        plan = [str(s) for s in structured.get("steps", [])]
        if not plan:
            # Providers without structured-output support (e.g. Ollama) plan in text.
            plan = _steps_from_text(plan_result.output_text)
        plan = plan[: self._max_steps]

        # 2. Execute each step over the shared thread (later steps see earlier results).
        step_results: list[RunResult] = []
        ctx: dict[str, Any] = {"model_key": model_key}
        if tool_names:
            ctx["tool_names"] = tool_names
        cfg = LLMConfig(
            model_key=model_key,
            temperature=self._temperature,
            response_format=ResponseFormat.AUTO_TOOLS if tool_names else None,
        )
        for index, step in enumerate(plan, start=1):
            result = await self._runtime.run_task_detailed(
                persona,
                Task(
                    title=f"step-{index}", prompt=f"Step {index}: {step}", context=ctx
                ),
                thread=thread,
                llm_config=cfg,
            )
            step_results.append(result)

        # 3. Synthesize a final answer from the work done.
        final = await self._runtime.run_task_detailed(
            persona,
            Task(
                title="synthesize",
                prompt=f"Using the steps above, give the final answer to: {goal}",
                context={"model_key": model_key},
            ),
            thread=thread,
            llm_config=LLMConfig(model_key=model_key, temperature=self._temperature),
        )

        all_results = [plan_result, *step_results, final]
        return PlanResult(
            thread=thread,
            goal=goal,
            plan=plan,
            step_results=step_results,
            output_text=final.output_text,
            total_cost=sum(r.cost for r in all_results),
        )


__all__ = ["PlannerOrchestrator", "PlanResult"]
