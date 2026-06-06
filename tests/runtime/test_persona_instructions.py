"""Regression: a persona's instructions must reach the system prompt.

A long-standing bug rendered the background as ``description or instructions``, so the
moment a persona had a description, every instruction was silently dropped — agents
(and team members) never saw their own directives. This pins the fix.
"""

from __future__ import annotations

import asyncio

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.services.inference import InferenceService, StubClientManager


def _system_prompt(persona: Persona) -> str:
    runtime, _inf, _tools = build_runtime(
        inference=InferenceService(StubClientManager())
    )
    result = asyncio.run(
        runtime.run_task_detailed(persona, Task(title="t", prompt="hi"))
    )
    system = [m for m in result.thread.messages if m.role.value == "system"]
    assert system, "no system message rendered"
    return system[0].content


def test_instructions_reach_prompt_even_with_description() -> None:
    persona = Persona(
        name="agent",
        description="A concise helper.",
        instructions=[
            "NEVER record data without confirming the details first.",
            "Ask one clarifying question when a request is ambiguous.",
        ],
    )
    prompt = _system_prompt(persona)
    assert "A concise helper." in prompt  # description still present
    assert "NEVER record data without confirming the details first." in prompt
    assert "Ask one clarifying question when a request is ambiguous." in prompt


def test_instructions_present_without_description() -> None:
    persona = Persona(
        name="agent",
        description="",
        instructions=["Only answer in haiku."],
    )
    assert "Only answer in haiku." in _system_prompt(persona)
