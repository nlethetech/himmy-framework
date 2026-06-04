"""Example 02: tool calling — register local tools and watch TOOL rows appear.

Two local function tools are registered on the shared ToolService. The task names
them via ``context['tool_names']`` and requests ``AUTO_TOOLS``; the runtime binds
them and the stub executes each one, replaying TOOL messages onto the thread.

    python examples/02_tool_calling.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runtime import build_runtime  # noqa: E402

from himmy.agents.base_agent.task import Task  # noqa: E402
from himmy.agents.base_agent.thread import MessageRole  # noqa: E402
from himmy.agents.personas.persona import Persona  # noqa: E402
from himmy.services.inference.models import LLMConfig, ResponseFormat  # noqa: E402
from himmy.services.tools.registry import register_local_tool  # noqa: E402


def _word_count(args: dict) -> dict:
    """Local tool: count the words in ``args['text']``."""
    text = str(args.get("text", ""))
    return {"word_count": len(text.split())}


def _to_upper(args: dict) -> dict:
    """Local tool: uppercase ``args['text']``."""
    text = str(args.get("text", ""))
    return {"upper": text.upper()}


async def main() -> None:
    """Register tools, run a tool-calling task, and print the TOOL exchanges."""
    runtime, _inference, tools = build_runtime()

    register_local_tool(
        tools.registry,
        name="word_count",
        handler=_word_count,
        description="Count words in a piece of text.",
        args_json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
        output_json_schema={
            "type": "object",
            "properties": {"word_count": {"type": "integer"}},
        },
    )
    register_local_tool(
        tools.registry,
        name="to_upper",
        handler=_to_upper,
        description="Uppercase a piece of text.",
        args_json_schema={
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )

    persona = Persona(
        name="text-utility-agent",
        description="You help users analyze text using tools.",
        metadata={"role": "Text Utility"},
    )
    task = Task(
        title="analyze-text",
        prompt="Analyze this text: 'Himmy makes agents composable and auditable.'",
        context={"tool_names": ["word_count", "to_upper"]},
    )

    thread = await runtime.run_task(
        persona, task, llm_config=LLMConfig(response_format=ResponseFormat.AUTO_TOOLS)
    )

    print("=== Example 02: tool calling ===")
    tool_rows = [m for m in thread.messages if m.role == MessageRole.TOOL]
    print(f"tools registered : {[d.name for d in tools.registry.list()]}")
    print(f"TOOL rows on thread: {len(tool_rows)}")
    print("-" * 60)
    for row in tool_rows:
        print(
            f"  [{row.metadata.get('tool_name')}] "
            f"outcome={row.metadata.get('tool_outcome')} "
            f"args={row.metadata.get('tool_args')} -> {row.content}"
        )
    print("-" * 60)
    last = thread.last_message
    assert last is not None
    print("assistant reply:")
    print(last.content)


if __name__ == "__main__":
    asyncio.run(main())
