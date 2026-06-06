#!/usr/bin/env python3
"""Validate the helpdesk agent against known-good answers — before you ship it.

A specialised agent is only trustworthy if you can prove it answers correctly. This is a
tiny "golden test" harness: a fixed set of questions, each with a fact that MUST appear in
the answer. Run it offline against your local model and get a pass-rate.

    python goldens.py

(For the full framework version of this idea, see `himmy eval` + a suite.yaml.)
"""

from __future__ import annotations

import asyncio

from helpdesk import _ingest, _persona_from_skill  # reuse the agent's setup

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.cli.provider import build_inference_for
from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit import ToolkitConfig, register_packs

# (question, accepted answer forms). A pass = the answer contains ANY of the forms, so the
# test measures CORRECTNESS, not formatting ("4%" and "4 percent" are both right).
GOLDENS = [
    ("How many vacation days do full-time employees get?", ["20"]),
    ("What is the daily meal reimbursement limit when traveling?", ["60"]),
    ("What mileage rate does the company reimburse?", ["0.67", "67 cent"]),
    ("Where do I report a phishing email?", ["security@company.com"]),
    ("How much does the company match into my 401k?", ["4%", "4 percent", "4 %"]),
    ("What is the annual home-office stipend?", ["500"]),
    ("How many weeks of paid parental leave are there?", ["12"]),
]


async def main() -> None:
    persona = _persona_from_skill()
    registry = ToolRegistry()
    register_packs(registry, ["knowledge"], ToolkitConfig.from_env())
    await _ingest(registry)
    runtime, *_ = build_runtime(
        inference=build_inference_for("ollama", "qwen2.5:3b-instruct"),
        tool_registry=registry,
    )

    passed = 0
    for question, accepted in GOLDENS:
        task = Task(title="g", prompt=question, context={"tool_names": ["kb_search"]})
        loop = await runtime.run_agent_loop(persona, task, max_turns=4)
        answer = (loop.final.output_text or "").lower()
        ok = any(form.lower() in answer for form in accepted)
        passed += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {question}  (accept {accepted})")

    print(f"\n{passed}/{len(GOLDENS)} golden answers correct.")
    raise SystemExit(0 if passed == len(GOLDENS) else 1)


if __name__ == "__main__":
    asyncio.run(main())
