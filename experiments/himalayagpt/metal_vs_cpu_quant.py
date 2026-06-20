"""Isolate the source of the fast/slow divergence: run the SAME Q8 GGUF through himmy
on (a) Metal partial-offload ngl=11 (:8081) and (b) pure CPU ngl=0 (:8082), identical
pure-argmax, full 18-task EN suite, and compare BOTH to the live transformers fp32
selections captured in parity_h2h_18_en.jsonl.

If CPU-Q8 agrees with fp32 much better than Metal-Q8 does, the divergence is the
Metal compute path (the documented nanochat-on-Metal issue), not the Q8 quant.
If CPU-Q8 also disagrees, it is the Q8 quant (or argmax tie-breaking) itself.

Run with himmy venv; both servers up:
    .venv/bin/python experiments/himalayagpt/metal_vs_cpu_quant.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hgpt_fast_bridge import HimalayaGptFastBridge  # noqa: E402
from tool_suite import TASKS, TOOLS, prompt_for  # noqa: E402

from himmy.services.inference.local import HimalayaGptClientManager  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ToolReturnRecord,
)

FORMAT = "hermes_chatml_xml_fewshot"
GEN = {"max_new_tokens": 64, "temperature": 0.0, "repetition_penalty": 1.0}


async def _stub(name, args):
    return ToolReturnRecord(tool_call_id="", tool_name=name, content="ok")


def _first(lst):
    return lst[0] if lst else None


async def _run(url, label):
    bridge = HimalayaGptFastBridge(base_url=url)
    bridge.start(timeout_s=10)
    mgr = HimalayaGptClientManager(generate_fn=bridge.generate_fn(**GEN), tool_call_format=FORMAT)
    out = {}
    for task in TASKS:
        req = InferenceRequest(
            model_key="default",
            messages=[InferenceMessage(role="user", content=prompt_for(task, "en"))],
            bound_tools=TOOLS,
            tool_executor=_stub,
        )
        resp = await mgr.generate(req)
        assert resp.status is InferenceStatus.SUCCESS, resp.error
        out[task["id"]] = (
            _first([c.tool_name for c in resp.tool_calls]),
            _first([c.args for c in resp.tool_calls]),
        )
    print(f"[{label}] done")
    return out


async def main() -> None:
    # fp32 reference from the live head-to-head we just ran
    ref = {}
    for line in (Path(__file__).with_name("parity_h2h_18_en.jsonl")).read_text().splitlines():
        r = json.loads(line)
        ref[r["task"]] = (r["slow_tool"], r["slow_args"])

    metal = await _run("http://127.0.0.1:8081", "metal ngl=11")
    cpu = await _run("http://127.0.0.1:8082", "cpu ngl=0")

    n = len(TASKS)
    m_sel = m_arg = c_sel = c_arg = mc_sel = 0
    print(f"\n{'task':<22}{'fp32':<18}{'metal11':<18}{'cpu0':<18}")
    for task in TASKS:
        rt, ra = ref[task["id"]]
        mt, ma = metal[task["id"]]
        ct, ca = cpu[task["id"]]
        m_sel += mt == rt
        m_arg += (mt == rt) and (ma == ra)
        c_sel += ct == rt
        c_arg += (ct == rt) and (ca == ra)
        mc_sel += mt == ct
        print(f"{task['id']:<22}{str(rt):<18}{str(mt):<18}{str(ct):<18}")

    print("\n" + "=" * 70)
    print(f"vs transformers fp32 reference (selection / args agreement, n={n}):")
    print(f"  Metal ngl=11 Q8 : sel {m_sel}/{n} ({100*m_sel/n:.0f}%)  args {m_arg}/{n} ({100*m_arg/n:.0f}%)")
    print(f"  CPU   ngl=0  Q8 : sel {c_sel}/{n} ({100*c_sel/n:.0f}%)  args {c_arg}/{n} ({100*c_arg/n:.0f}%)")
    print(f"  Metal-vs-CPU same selection: {mc_sel}/{n} ({100*mc_sel/n:.0f}%)")
    print("\nINTERPRETATION:")
    print("  If CPU-Q8 >> Metal-Q8 agreement -> divergence is the Metal compute path.")
    print("  If CPU-Q8 ~ Metal-Q8 (both low) -> divergence is the Q8 quant / argmax ties.")


if __name__ == "__main__":
    asyncio.run(main())
