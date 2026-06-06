#!/usr/bin/env python3
"""Chat with your own documents — 100% offline, no API keys, no cloud.

This ingests the Markdown files in ./docs into himmy's local knowledge base (a local
hashing embedder — nothing is downloaded, nothing leaves your machine), then answers
your questions with a local model via Ollama, grounded strictly in those docs.

    python chat.py                      # interactive
    python chat.py "How many PTO days?" # one-shot

Requirements: Ollama running with a small model pulled, e.g.
    ollama pull qwen2.5:3b-instruct
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.cli.provider import build_inference_for
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit import ToolkitConfig, register_packs

DOCS = Path(__file__).parent / "docs"
MODEL = "qwen2.5:3b-instruct"  # any local Ollama model works

PERSONA = Persona(
    name="doc-chat",
    description=(
        "You answer questions strictly from the user's ingested documents. Always call "
        "kb_search first, then answer only from what it returns, and name the source "
        "file. If the documents do not cover the question, say so plainly."
    ),
)


async def _ingest(registry: ToolRegistry) -> int:
    """Load every Markdown doc into the local knowledge base. Returns the count."""
    ingest = registry.handler_for("kb_ingest")
    count = 0
    for path in sorted(DOCS.glob("*.md")):
        await ingest(
            {
                "text": path.read_text(encoding="utf-8"),
                "title": path.stem,
                "source_uri": path.name,
            }
        )
        count += 1
    return count


async def _answer(runtime: object, question: str) -> str:
    task = Task(title="q", prompt=question, context={"tool_names": ["kb_search"]})
    loop = await runtime.run_agent_loop(PERSONA, task, max_turns=4)  # type: ignore[attr-defined]
    return (loop.final.output_text or "").strip() if loop.final else ""


async def main() -> None:
    registry = ToolRegistry()
    register_packs(registry, ["knowledge"], ToolkitConfig.from_env())
    n = await _ingest(registry)
    print(f"✓ ingested {n} documents into a local knowledge base — offline, no keys.\n")

    runtime, *_ = build_runtime(
        inference=build_inference_for("ollama", MODEL), tool_registry=registry
    )

    if len(sys.argv) > 1:  # one-shot mode
        question = " ".join(sys.argv[1:])
        print(f"you   › {question}")
        print(f"agent › {await _answer(runtime, question)}")
        return

    print("Ask about the documents (blank line to quit).")
    while True:
        try:
            question = input("\nyou   › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            break
        print(f"agent › {await _answer(runtime, question)}")


if __name__ == "__main__":
    asyncio.run(main())
