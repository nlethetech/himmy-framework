"""Example 04: orchestration team — 3 specialists via run_batch + manager synthesis.

Three specialist requests are run concurrently through
``InferenceService.run_batch`` (bounded concurrency, order preserved). A manager
persona then synthesizes the three specialist findings into one answer via the
runtime. Fully offline against the deterministic stub.

    python examples/04_orchestration_team.py
"""

from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _runtime import build_runtime  # noqa: E402

from himmy.agents.base_agent.task import Task  # noqa: E402
from himmy.agents.personas.persona import Persona  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    BatchInferenceRequest,
    InferenceMessage,
    InferenceRequest,
)

SPECIALISTS = [
    ("technical-analyst", "Assess the technical/chart picture for NVIDIA."),
    ("fundamental-analyst", "Assess the fundamentals/valuation for NVIDIA."),
    ("macro-analyst", "Assess the macro/sector backdrop for NVIDIA."),
]


async def main() -> None:
    """Fan out to three specialists in a batch, then synthesize via a manager."""
    runtime, inference, _tools = build_runtime()

    # --- 1. fan out: one inference request per specialist, run as a batch ---
    requests = [
        InferenceRequest(
            messages=[
                InferenceMessage(role="system", content=f"You are a {role}."),
                InferenceMessage(role="user", content=prompt),
            ]
        )
        for role, prompt in SPECIALISTS
    ]
    batch = await inference.run_batch(
        BatchInferenceRequest(requests=requests, max_concurrency=3)
    )

    print("=== Example 04: orchestration team ===")
    print(
        f"batch: {batch.success_count} succeeded, "
        f"{batch.failure_count} failed in {batch.elapsed_ms:.2f} ms"
    )
    print("-" * 60)
    findings: list[str] = []
    for (role, _prompt), response in zip(SPECIALISTS, batch.responses, strict=False):
        text = response.output_text or ""
        findings.append(f"- {role}: {text}")
        print(f"  [{role}] {text}")
    print("-" * 60)

    # --- 2. manager synthesis: feed the findings back through the runtime ---
    manager = Persona(
        name="research-manager",
        description="You synthesize specialist findings into a single recommendation.",
        metadata={"role": "Research Manager"},
    )
    synthesis_task = Task(
        title="synthesize-findings",
        prompt="Synthesize these specialist findings into one verdict:\n"
        + "\n".join(findings),
    )
    thread = await runtime.run_task(manager, synthesis_task)
    last = thread.last_message
    assert last is not None
    print("manager synthesis:")
    print(last.content)


if __name__ == "__main__":
    asyncio.run(main())
