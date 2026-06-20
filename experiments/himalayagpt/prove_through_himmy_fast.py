"""Prove the FAST seam: drive the REAL HimalayaGPT 0.5b Q8 GGUF (Metal, via the
nanochat llama.cpp fork) THROUGH himmy's HimalayaGptClientManager using the
llama-server-backed generate_fn bridge — for an EN and a NE prompt.

Identical to prove_through_himmy.py except the generate_fn comes from
HimalayaGptFastBridge (llama-server /completion) instead of the slow transformers
worker. The manager + boosted Hermes few-shot format are unchanged.

Run with himmy's venv (py3.14):
    .venv/bin/python experiments/himalayagpt/prove_through_himmy_fast.py
A llama-server must be running (or this spawns one at ngl=11).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hgpt_fast_bridge import HimalayaGptFastBridge  # noqa: E402

from himmy.services.inference.local import HimalayaGptClientManager  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
)
from himmy.services.inference.tool_formats import format_for  # noqa: E402


async def main() -> None:
    # Attach to an already-running server on :8081 if present; else spawn one.
    bridge = HimalayaGptFastBridge(base_url="http://127.0.0.1:8081")
    try:
        bridge.start(timeout_s=5)
        print("[bridge] attached to running llama-server :8081")
    except Exception:
        bridge = HimalayaGptFastBridge()  # spawn our own (ngl=11)
        bridge.start()
        print("[bridge] spawned llama-server (ngl=11)")

    manager = HimalayaGptClientManager(
        generate_fn=bridge.generate_fn(max_new_tokens=64, temperature=0.0),
        tool_call_format="hermes_chatml_xml",
    )

    resolved = manager.resolve("default")
    fmt = format_for(resolved, manager._tool_call_format)
    print(f"[manager] resolved model id = {resolved}")
    print(f"[manager] tool-call format  = {fmt.name}")

    for label, prompt in [("EN", "What is 2+2?"), ("NE", "नेपालको राजधानी के हो?")]:
        req = InferenceRequest(
            model_key="default",
            messages=[InferenceMessage(role="user", content=prompt)],
        )
        t0 = time.perf_counter()
        resp = await manager.generate(req)
        dt = (time.perf_counter() - t0) * 1000.0
        assert resp.status is InferenceStatus.SUCCESS, resp.error
        print(f"\n=== [{label}] via himmy manager (FAST) ===")
        print("prompt :", prompt)
        print("output :", repr(resp.output_text))
        print(f"latency: {dt:.0f} ms  provider={resp.provider_name}")

    print("\n[done] FAST generation through HimalayaGptClientManager succeeded.")


if __name__ == "__main__":
    asyncio.run(main())
