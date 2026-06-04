"""Example 01: basic chat — a persona + task answered by the SingleAgentRuntime.

Runs offline against the deterministic stub by default. Prints the assistant's
reply plus the latency/model metadata stamped on the message.

    python examples/01_basic_chat.py
"""

from __future__ import annotations

import asyncio
import os
import sys

# Make the sibling ``_runtime`` module importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runtime import build_runtime  # noqa: E402

from opensims.agents.base_agent.task import Task  # noqa: E402
from opensims.agents.personas.persona import Persona  # noqa: E402


async def main() -> None:
    """Build the runtime, run one task, and print the answer + metadata."""
    runtime, _inference, _tools = build_runtime()

    persona = Persona(
        name="market-research-analyst",
        description="You are a market research analyst specializing in tech.",
        instructions=["Provide actionable insights backed by clear reasoning."],
        metadata={"role": "Research Analyst"},
    )
    task = Task(
        title="semiconductor-evaluation",
        prompt="Three most important factors when evaluating a semiconductor stock?",
    )

    thread = await runtime.run_task(persona, task)
    last = thread.last_message
    assert last is not None

    print("=== Example 01: basic chat ===")
    print(f"persona : {persona.name} (role={persona.role})")
    print(f"task    : {task.title}")
    print("-" * 60)
    print("assistant reply:")
    print(last.content)
    print("-" * 60)
    meta = last.metadata
    print(f"model_path   : {meta.get('model_path')}")
    print(f"provider     : {meta.get('provider_name')}")
    print(f"latency_ms   : {meta.get('latency_ms')}")
    print(f"input_tokens : {meta.get('input_tokens')}")
    print(f"output_tokens: {meta.get('output_tokens')}")
    print(f"messages     : {len(thread.messages)} on thread {thread.thread_id}")


if __name__ == "__main__":
    asyncio.run(main())
