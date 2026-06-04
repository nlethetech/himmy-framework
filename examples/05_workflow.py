"""Example 05: workflow — forced sequential tools then a structured report step.

Builds a two-step :class:`Workflow`: step 1 forces one-tool-per-call across
``count_words`` -> ``score_sentiment`` -> ``extract_topics`` (via
``ResponseFormat.WORKFLOW``); step 2 produces a structured report whose output is
threaded into the final state. Fully offline against the deterministic stub.

    python examples/05_workflow.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runtime import build_runtime  # noqa: E402

from opensims.agents.personas.persona import Persona  # noqa: E402
from opensims.orchestrators.workflow import (  # noqa: E402
    Workflow,
    WorkflowOrchestrator,
    WorkflowStep,
)
from opensims.services.inference.models import ResponseFormat  # noqa: E402
from opensims.services.tools.registry import register_local_tool  # noqa: E402


def _count_words(args: dict) -> dict:
    """Tool: count words in ``args['text']``."""
    return {"word_count": len(str(args.get("text", "")).split())}


def _score_sentiment(args: dict) -> dict:
    """Tool: a trivial deterministic sentiment score for ``args['text']``."""
    text = str(args.get("text", "")).lower()
    score = 0.5 + 0.1 * text.count("good") - 0.1 * text.count("bad")
    return {"sentiment": max(0.0, min(1.0, score))}


def _extract_topics(args: dict) -> dict:
    """Tool: extract a few topic-ish tokens from ``args['text']``."""
    words = [w.strip(".,") for w in str(args.get("text", "")).split() if len(w) > 5]
    return {"topics": words[:3]}


async def main() -> None:
    """Register the workflow tools, run the workflow, and print step results."""
    runtime, _inference, tools = build_runtime()

    args_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    for name, handler in (
        ("count_words", _count_words),
        ("score_sentiment", _score_sentiment),
        ("extract_topics", _extract_topics),
    ):
        register_local_tool(
            tools.registry,
            name=name,
            handler=handler,
            description=f"{name} over a text input.",
            args_json_schema=args_schema,
        )

    persona = Persona(
        name="content-analyst",
        description="You analyze content with tools and write structured reports.",
        metadata={"role": "Content Analyst"},
    )

    workflow = Workflow(
        name="content-analysis",
        description="Analyze text with sequential tools, then write a report.",
        steps=[
            WorkflowStep(
                name="analyze",
                subtask="Analyze this text with the tools: {text}",
                tool_names=["count_words", "score_sentiment", "extract_topics"],
                sequential_tools=True,
                output_key="analysis",
            ),
            WorkflowStep(
                name="report",
                subtask="Write a short report given this analysis: {analysis}",
                response_format=ResponseFormat.STRUCTURED_OUTPUT,
                output_json_schema={
                    "type": "object",
                    "properties": {
                        "headline": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["headline", "summary"],
                },
                output_key="report",
            ),
        ],
    )

    orchestrator = WorkflowOrchestrator(runtime)
    result = await orchestrator.run(
        workflow,
        persona,
        initial_state={"text": "OpenSims is a good, composable agent framework."},
    )

    print("=== Example 05: workflow ===")
    print(f"workflow: {result.workflow_name}  status={result.status}")
    print(f"thread  : {result.thread_id}")
    print("-" * 60)
    for step in result.step_results:
        print(f"  step '{step.step_name}': {step.status}")
        if step.tool_calls:
            # step.tool_calls are typed ToolCallRecord objects (RO-4).
            names = [getattr(c, "tool_name", None) for c in step.tool_calls]
            print(f"     tools called: {names}")
        if step.output_text:
            print(f"     output_text : {step.output_text}")
        if step.error:
            print(f"     error       : {step.error}")
    print("-" * 60)
    print("final_state keys:", list(result.final_state.keys()))
    print("report:", result.final_state.get("report"))


if __name__ == "__main__":
    asyncio.run(main())
