"""Example 11: a web-research agent — search the live web, then summarize.

    python examples/11_web_research.py                                     # offline stub
    HIMMY_EXAMPLE_PROVIDER=ollama HIMMY_EXAMPLE_MODEL=qwen2.5:3b-instruct \\
        python examples/11_web_research.py                                 # real model + web

With a real model this calls `web_search` (keyless DuckDuckGo), the runtime executes it
against the live web, and the model summarizes the result. Needs network for the real
path; the stub path is deterministic and offline.
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
    """Run a web_search-using research agent on the configured provider."""
    provider = os.environ.get("HIMMY_EXAMPLE_PROVIDER")
    model = os.environ.get("HIMMY_EXAMPLE_MODEL")
    topic = os.environ.get("HIMMY_EXAMPLE_TOPIC", "what a permaculture food forest is")

    registry = ToolRegistry()
    register_packs(registry, ["web"], ToolkitConfig())
    inference = build_inference_for(provider, model)
    runtime, _inference, _tools = build_runtime(
        inference=inference, tool_registry=registry
    )

    persona = Persona(
        name="researcher",
        description="You research a topic with web_search, then summarize in one sentence.",
    )
    task = Task(
        title="research",
        prompt=f"Use web_search to find {topic}, then summarize it in one sentence.",
        context={"tool_names": ["web_search", "web_fetch"]},
    )
    loop = await runtime.run_agent_loop(
        persona,
        task,
        max_turns=5,
        llm_config=LLMConfig(response_format=ResponseFormat.AUTO_TOOLS),
    )

    print(f"=== Example 11: web research (provider={provider or 'stub'}) ===")
    print(f"stopped_reason : {loop.stopped_reason}   turns: {loop.turn_count}")
    for i, turn in enumerate(loop.turns):
        print(f"  turn {i}: tools={[c.tool_name for c in turn.tool_calls]}")
    print(f"summary        : {loop.final.output_text}")


if __name__ == "__main__":
    asyncio.run(main())
