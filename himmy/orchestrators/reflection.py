"""Reflection: a critique-and-revise pass that improves a draft answer.

A single reflection turn asks the model to critique its own draft against optional
criteria and return an improved version — the cheapest quality lift available, usable
standalone or as a final step in a plan-and-execute run.
"""

from __future__ import annotations

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread
from himmy.agents.personas.persona import Persona
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.inference.models import LLMConfig


async def reflect(
    runtime: SingleAgentRuntime,
    draft: str,
    *,
    persona: Persona | None = None,
    criteria: str | None = None,
    model_key: str = "default",
) -> str:
    """Critique ``draft`` and return an improved version (one model turn)."""
    persona = persona or Persona(
        name="reviewer",
        description="You critique a draft and return an improved version.",
    )
    rubric = criteria or "accuracy, clarity, and completeness"
    result = await runtime.run_task_detailed(
        persona,
        Task(
            title="reflect",
            prompt=(
                f"Critique the following draft for {rubric}, then return ONLY the "
                f"improved final version (no preamble):\n\n{draft}"
            ),
            context={"model_key": model_key},
        ),
        thread=ChatThread(agent_id=persona.agent_id),
        llm_config=LLMConfig(model_key=model_key, temperature=0.2),
    )
    return result.output_text or draft


__all__ = ["reflect"]
