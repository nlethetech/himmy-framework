"""Example 10: a REAL tool-using agent (calculator) — runs on the stub or a live model.

Unlike the stub-only examples, this one is provider-selectable so you can watch a real
local model actually call a tool, get the result, and answer:

    python examples/10_real_tool_agent.py                                  # offline stub
    HIMMY_EXAMPLE_PROVIDER=ollama HIMMY_EXAMPLE_MODEL=qwen2.5:3b-instruct \\
        python examples/10_real_tool_agent.py                              # real model

The agent has the `utils` toolkit pack (calculator, current_time). With a real model it
emits a `calculator` tool call, the runtime executes it, and the model answers using the
result — `stopped_reason="final"`.
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from himmy import build_runtime  # noqa: E402
from himmy.agents.base_agent.task import Task  # noqa: E402
from himmy.agents.personas.persona import Persona  # noqa: E402
from himmy.cli.provider import build_inference_for  # noqa: E402
from himmy.services.inference.models import LLMConfig, ResponseFormat  # noqa: E402
from himmy.services.tools.registry import ToolRegistry  # noqa: E402
from himmy.toolkit import ToolkitConfig, register_packs  # noqa: E402


async def main() -> None:
    """Run a calculator-using agent on the configured provider."""
    provider = os.environ.get("HIMMY_EXAMPLE_PROVIDER")  # None → offline stub
    model = os.environ.get("HIMMY_EXAMPLE_MODEL")

    registry = ToolRegistry()
    register_packs(registry, ["utils"], ToolkitConfig())
    inference = build_inference_for(provider, model)
    runtime, _inference, _tools = build_runtime(
        inference=inference, tool_registry=registry
    )

    persona = Persona(
        name="math-helper",
        description="You solve arithmetic using the calculator tool, then state the answer.",
    )
    task = Task(
        title="calc",
        prompt="What is 17 multiplied by 23? Use the calculator tool, then give the answer.",
        context={"tool_names": ["calculator"]},
    )
    loop = await runtime.run_agent_loop(
        persona,
        task,
        max_turns=4,
        llm_config=LLMConfig(response_format=ResponseFormat.AUTO_TOOLS),
    )

    print(f"=== Example 10: real tool agent (provider={provider or 'stub'}) ===")
    print(f"stopped_reason : {loop.stopped_reason}   turns: {loop.turn_count}")
    for i, turn in enumerate(loop.turns):
        calls = [(c.tool_name, c.args) for c in turn.tool_calls]
        print(f"  turn {i}: tool_calls={calls}")
    print(f"answer         : {loop.final.output_text}")


if __name__ == "__main__":
    asyncio.run(main())
