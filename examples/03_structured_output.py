"""Example 03: structured output — STRUCTURED_OUTPUT against a JSON schema.

Drives the run with ``LLMConfig(response_format=STRUCTURED_OUTPUT,
output_json_schema=RecommendationEnvelope schema)``. The offline stub synthesizes
a schema-valid instance (seeding prose fields from the prompt); we parse it back
into a :class:`RecommendationEnvelope` and print the recommendations.

    python examples/03_structured_output.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runtime import build_runtime  # noqa: E402

from opensims.agents.base_agent.task import Task  # noqa: E402
from opensims.agents.personas.persona import Persona  # noqa: E402
from opensims.application.models import RecommendationEnvelope  # noqa: E402
from opensims.services.inference.models import LLMConfig, ResponseFormat  # noqa: E402


async def main() -> None:
    """Run a structured-output task and parse the result as a RecommendationEnvelope."""
    runtime, _inference, _tools = build_runtime()

    # An inlined RecommendationEnvelope schema. We avoid ``$ref``/top-level
    # ``default: []`` so the offline stub's schema synthesizer actually populates
    # recommendations (seeding the prose fields from the prompt). The result still
    # validates cleanly as a :class:`RecommendationEnvelope`.
    schema = {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string", "default": "portfolio_action"},
                        "title": {"type": "string"},
                        "summary": {"type": "string"},
                        "rationale": {"type": "string"},
                        "confidence": {"type": "number", "default": 0.7},
                    },
                    "required": ["kind", "title"],
                },
            },
            "summary": {"type": "string"},
        },
        "required": ["recommendations"],
    }

    persona = Persona(
        name="portfolio-advisor",
        description="You produce structured, evidence-backed recommendations.",
        metadata={"role": "Portfolio Advisor"},
    )
    task = Task(
        title="rebalance-recommendations",
        prompt="Recommend two portfolio actions for a tech-heavy investor.",
    )

    cfg = LLMConfig(
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema=schema,
    )
    thread = await runtime.run_task(persona, task, llm_config=cfg)

    last = thread.last_message
    assert last is not None
    parsed = json.loads(last.content)
    envelope = RecommendationEnvelope.model_validate(parsed)

    print("=== Example 03: structured output ===")
    print(f"summary: {envelope.summary}")
    print(f"recommendations: {len(envelope.recommendations)}")
    print("-" * 60)
    for i, rec in enumerate(envelope.recommendations, start=1):
        print(f"  {i}. kind={rec.kind!r} title={rec.title!r}")
        print(f"     summary   : {rec.summary}")
        print(f"     rationale : {rec.rationale}")
        print(f"     confidence: {rec.confidence}")
    print("-" * 60)
    print("raw structured output:")
    print(json.dumps(parsed, indent=2)[:600])


if __name__ == "__main__":
    asyncio.run(main())
