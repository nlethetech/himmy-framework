"""Prove the chosen seam: drive the REAL HimalayaGPT 0.5b THROUGH himmy's
HimalayaGptClientManager via the generate_fn bridge — for an EN and a NE prompt.

Run with himmy's venv (py3.14):
    .venv/bin/python experiments/himalayagpt/prove_through_himmy.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hgpt_bridge import HimalayaGptWorker  # noqa: E402

from himmy.services.inference.local import HimalayaGptClientManager  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
)
from himmy.services.inference.tool_formats import format_for  # noqa: E402


async def main() -> None:
    worker = HimalayaGptWorker()
    worker.start()
    print("[worker] READY (model loaded in py3.12 venv)")

    # The manager is the ONLY thing the benchmark talks to. We inject the bridge's
    # generate_fn — the documented injection seam. tool_call_format pins the grammar.
    manager = HimalayaGptClientManager(
        generate_fn=worker.generate_fn(max_new_tokens=64, temperature=0.0),
        tool_call_format="hermes_chatml_xml",
    )

    # Show the resolved model id and the format the registry picks for it.
    resolved = manager.resolve("default")
    fmt = format_for(resolved, manager._tool_call_format)
    print(f"[manager] resolved model id = {resolved}")
    print(f"[manager] tool-call format  = {fmt.name} (override='{manager._tool_call_format}')")

    for label, prompt in [("EN", "What is 2+2?"), ("NE", "नेपालको राजधानी के हो?")]:
        req = InferenceRequest(
            model_key="default",
            messages=[InferenceMessage(role="user", content=prompt)],
        )
        resp = await manager.generate(req)
        assert resp.status is InferenceStatus.SUCCESS, resp.error
        print(f"\n=== [{label}] via himmy manager ===")
        print("prompt :", prompt)
        print("output :", repr(resp.output_text))
        print(f"latency: {resp.latency_ms:.0f} ms  provider={resp.provider_name}")

    worker.close()
    print("\n[done] real generation through HimalayaGptClientManager succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
