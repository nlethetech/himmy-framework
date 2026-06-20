"""AUTHORITATIVE parity: full 18-task EN suite, BOTH backends driven LIVE through the
SAME himmy manager + boosted format with the IDENTICAL pure-argmax decoding config.

This is the only valid quant-faithfulness test: fast Q8 GGUF (llama.cpp/Metal) vs slow
transformers fp32, same prompt, same format, same temp=0 / repetition_penalty=1.0
(argmax). Any divergence here is the Q8 quant + Metal path (or argmax tie-breaking),
NOT a decoding-config mismatch.

Run with himmy's venv (py3.14); llama-server up on :8081 (Metal, ngl=11):
    .venv/bin/python experiments/himalayagpt/parity_h2h_18_en.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hgpt_bridge import HimalayaGptWorker  # noqa: E402
from hgpt_fast_bridge import HimalayaGptFastBridge  # noqa: E402
from tool_suite import TASKS, TOOLS, prompt_for  # noqa: E402

from himmy.benchmark.graders import grade_trajectory  # noqa: E402
from himmy.benchmark.trajectory import Trajectory  # noqa: E402
from himmy.services.inference.local import HimalayaGptClientManager  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ToolReturnRecord,
)

FORMAT = "hermes_chatml_xml_fewshot"
GEN = {"max_new_tokens": 64, "temperature": 0.0, "repetition_penalty": 1.0}


async def _stub(name: str, args: dict) -> ToolReturnRecord:
    return ToolReturnRecord(tool_call_id="", tool_name=name, content="ok")


def _first(lst):
    return lst[0] if lst else None


def _grade(task, traj):
    return (
        grade_trajectory({"type": "tool_called", "value": task["expect_tool"]}, traj),
        grade_trajectory(
            {"type": "tool_called_with", "tool": task["expect_tool"], "args": task["expect_args"]},
            traj,
        ),
    )


async def _run_arm(generate_fn, label):
    mgr = HimalayaGptClientManager(generate_fn=generate_fn, tool_call_format=FORMAT)
    out = {}
    for task in TASKS:
        req = InferenceRequest(
            model_key="default",
            messages=[InferenceMessage(role="user", content=prompt_for(task, "en"))],
            bound_tools=TOOLS,
            tool_executor=_stub,
        )
        t0 = time.perf_counter()
        resp = await mgr.generate(req)
        dt = (time.perf_counter() - t0) * 1000.0
        assert resp.status is InferenceStatus.SUCCESS, resp.error
        traj = Trajectory.from_turns([(resp.tool_calls, resp.tool_returns)])
        sel, arg = _grade(task, traj)
        out[task["id"]] = {
            "tool": _first([c.tool_name for c in resp.tool_calls]),
            "args": _first([c.args for c in resp.tool_calls]),
            "sel": sel,
            "arg": arg,
            "ms": dt,
        }
        print(f"  [{label}] {task['id']:<22} {dt:8.0f}ms -> {out[task['id']]['tool']}")
    return out


async def main() -> None:
    fast = HimalayaGptFastBridge(base_url="http://127.0.0.1:8081")
    fast.start(timeout_s=10)
    print("[fast] attached to llama-server :8081 (Metal, ngl=11) -- running FAST arm")
    fast_out = await _run_arm(fast.generate_fn(**GEN), "fast")

    slow = HimalayaGptWorker()
    slow.start()
    print("[slow] transformers fp32 worker READY -- running SLOW arm (this is slow)")
    slow_out = await _run_arm(slow.generate_fn(**GEN), "slow")
    slow.close()

    n = len(TASKS)
    fsel = sum(1 for v in fast_out.values() if v["sel"])
    farg = sum(1 for v in fast_out.values() if v["arg"])
    ssel = sum(1 for v in slow_out.values() if v["sel"])
    sarg = sum(1 for v in slow_out.values() if v["arg"])
    agree_sel = agree_arg = 0
    rows = []
    print("\n" + "=" * 88)
    print(f"{'task':<22}{'fast_tool':<18}{'slow_tool':<18}{'agree_sel':<11}{'agree_arg':<11}")
    for task in TASKS:
        f = fast_out[task["id"]]
        s = slow_out[task["id"]]
        s_ag = f["tool"] == s["tool"]
        a_ag = s_ag and (f["args"] == s["args"])
        agree_sel += s_ag
        agree_arg += a_ag
        print(f"{task['id']:<22}{str(f['tool']):<18}{str(s['tool']):<18}{str(s_ag):<11}{str(a_ag):<11}")
        rows.append(
            {
                "task": task["id"],
                "expect_tool": task["expect_tool"],
                "fast_tool": f["tool"],
                "fast_args": f["args"],
                "fast_sel": f["sel"],
                "fast_arg": f["arg"],
                "slow_tool": s["tool"],
                "slow_args": s["args"],
                "slow_sel": s["sel"],
                "slow_arg": s["arg"],
                "agree_sel": s_ag,
                "agree_arg": a_ag,
                "fast_ms": f["ms"],
                "slow_ms": s["ms"],
            }
        )

    out = Path(__file__).with_name("parity_h2h_18_en.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))

    favg = sum(v["ms"] for v in fast_out.values()) / n
    savg = sum(v["ms"] for v in slow_out.values()) / n
    print("\n" + "=" * 88)
    print(f"ABSOLUTE correctness (EN, n={n}, pure argmax):")
    print(f"  fast Q8/Metal : selection {fsel}/{n} ({100*fsel/n:.0f}%)  args {farg}/{n} ({100*farg/n:.0f}%)")
    print(f"  slow fp32 ref : selection {ssel}/{n} ({100*ssel/n:.0f}%)  args {sarg}/{n} ({100*sarg/n:.0f}%)")
    print(f"PARITY (fast vs slow, identical decode):")
    print(f"  selection agreement {agree_sel}/{n} = {100*agree_sel/n:.1f}%")
    print(f"  args agreement      {agree_arg}/{n} = {100*agree_arg/n:.1f}%")
    print(f"LATENCY: fast avg {favg:.0f} ms/call   slow avg {savg:.0f} ms/call   speedup {savg/favg:.0f}x")
    print(f"raw -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
