#!/usr/bin/env python3
"""Offline company-handbook helpdesk — a richer himmy example.

Builds on examples/local-doc-chat with the things that make a *real* assistant:

* a reusable **skill** (skills/helpdesk.yaml) defines the agent's behaviour — no code;
* a **multi-turn conversation** with memory, so follow-ups ("…and for part-time?") work;
* **grounded + cited** answers, and an honest "that's not in the handbook" when it isn't.

Everything runs locally: docs are embedded with a local hashing embedder (no download),
and answers come from a local model via Ollama. Nothing leaves your machine.

    python helpdesk.py        # run a scripted demo conversation
    python helpdesk.py -i     # interactive chat (keeps memory across turns)

Requires Ollama with a 3B+ model: `ollama pull qwen2.5:3b-instruct`
(tiny models skip the search and hallucinate — see the README).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.cli.provider import build_inference_for
from himmy.services.tools.registry import ToolRegistry
from himmy.skills import load_skill_file
from himmy.toolkit import ToolkitConfig, register_packs

HERE = Path(__file__).parent
MODEL = os.environ.get("HELPDESK_MODEL", "qwen2.5:3b-instruct")

# A scripted conversation that shows retrieval, memory (turn 2 refers back), specific
# facts, and an honest "not in the handbook" (the last one is deliberately out of scope).
DEMO = [
    "How many paid vacation days do full-time employees get?",
    "And what about part-time employees?",  # needs memory of the previous turn
    "What's the daily meal limit when I travel for work?",
    "Can I plug my own USB drive into my work laptop?",
    "What's the company's policy on bringing pets to the office?",  # not in the docs
]


def _persona_from_skill() -> Persona:
    """Turn the YAML skill into the agent's persona (behaviour as config, not code)."""
    skill = load_skill_file(HERE / "skills" / "helpdesk.yaml")
    background = "\n".join([skill.description, *(f"- {i}" for i in skill.instructions)])
    return Persona(name=skill.name, description=background)


async def _ingest(registry: ToolRegistry) -> int:
    ingest = registry.handler_for("kb_ingest")
    count = 0
    for path in sorted((HERE / "docs").glob("*.md")):
        await ingest(
            {
                "text": path.read_text(encoding="utf-8"),
                "title": path.stem,
                "source_uri": path.name,
            }
        )
        count += 1
    return count


async def main() -> None:
    persona = _persona_from_skill()
    registry = ToolRegistry()
    register_packs(registry, ["knowledge"], ToolkitConfig.from_env())
    n = await _ingest(registry)
    print(f"✓ helpdesk ready — {n} handbook docs in a local KB ({MODEL}, offline).\n")

    runtime, *_ = build_runtime(
        inference=build_inference_for("ollama", MODEL), tool_registry=registry
    )

    interactive = "-i" in sys.argv or "--interactive" in sys.argv
    thread = None  # one thread across the whole conversation → the agent remembers

    async def ask(question: str) -> None:
        nonlocal thread
        task = Task(title="q", prompt=question, context={"tool_names": ["kb_search"]})
        loop = await runtime.run_agent_loop(persona, task, thread, max_turns=4)  # type: ignore[attr-defined]
        thread = loop.thread  # carry the conversation forward
        print(f"you   › {question}")
        print(f"agent › {(loop.final.output_text or '').strip()}\n")

    if interactive:
        print(
            "Ask the helpdesk anything (blank line to quit). It remembers the thread."
        )
        while True:
            try:
                q = input("you   › ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                break
            task = Task(title="q", prompt=q, context={"tool_names": ["kb_search"]})
            loop = await runtime.run_agent_loop(persona, task, thread, max_turns=4)  # type: ignore[attr-defined]
            thread = loop.thread
            print(f"agent › {(loop.final.output_text or '').strip()}\n")
        return

    for q in DEMO:
        await ask(q)


if __name__ == "__main__":
    asyncio.run(main())
