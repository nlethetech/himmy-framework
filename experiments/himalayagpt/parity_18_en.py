"""Full 18-task EN tool-calling PARITY check: fast Q8 GGUF (llama.cpp/Metal) through
himmy's HimalayaGptClientManager + boosted format, graded with the shipped EMIT/
SELECTION/ARGS graders, and compared task-by-task against the transformers fp32
reference (the 'new'/boosted arm in ab_results_powered.jsonl, EN).

PURE GREEDY (temp 0, repetition_penalty 1.0 == argmax) so a match is a faithful one,
not a backend decoding artifact.

Run with himmy's venv (py3.14); llama-server must be up on :8081 (Metal, ngl=11):
    .venv/bin/python experiments/himalayagpt/parity_18_en.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

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
REF_PATH = Path(__file__).with_name("ab_results_powered.jsonl")


async def _stub(name: str, args: dict) -> ToolReturnRecord:
    return ToolReturnRecord(tool_call_id="", tool_name=name, content="ok")


def _first(lst):
    return lst[0] if lst else None


def _load_ref() -> dict:
    """transformers fp32 reference: 'new' (boosted) arm, EN, keyed by task_id."""
    ref = {}
    for line in REF_PATH.read_text().splitlines():
        r = json.loads(line)
        if r["arm"] == "new" and r["lang"] == "en":
            ref[r["task_id"]] = {
                "tool": _first(r["emitted_tools"]),
                "args": _first(r["emitted_args"]) if r.get("emitted_args") else None,
                "selection": r["selection"],
                "args_ok": r["args"],
            }
    return ref


def _grade(task, traj) -> dict:
    return {
        "emit": len(traj.calls) > 0,
        "selection": grade_trajectory(
            {"type": "tool_called", "value": task["expect_tool"]}, traj
        ),
        "args": grade_trajectory(
            {
                "type": "tool_called_with",
                "tool": task["expect_tool"],
                "args": task["expect_args"],
            },
            traj,
        ),
    }


async def main() -> None:
    ref = _load_ref()
    bridge = HimalayaGptFastBridge(base_url="http://127.0.0.1:8081")
    bridge.start(timeout_s=10)
    print("[fast] attached to llama-server :8081 (Metal, ngl=11)")

    mgr = HimalayaGptClientManager(
        generate_fn=bridge.generate_fn(**GEN), tool_call_format=FORMAT
    )
    print(f"[manager] model={mgr.resolve('default')} format={mgr._tool_call_format}")

    f_sel = f_arg = 0
    agree_sel = agree_arg = 0
    lats = []
    rows = []
    print(
        f"\n{'task':<22}{'fast_tool':<20}{'ref_tool':<20}"
        f"{'f_sel':<6}{'f_arg':<6}{'agree_sel':<10}{'agree_arg':<10}"
    )
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
        lats.append(dt)
        assert resp.status is InferenceStatus.SUCCESS, resp.error
        traj = Trajectory.from_turns([(resp.tool_calls, resp.tool_returns)])
        g = _grade(task, traj)
        f_sel += g["selection"]
        f_arg += g["args"]

        ftool = _first([c.tool_name for c in resp.tool_calls])
        fargs = _first([c.args for c in resp.tool_calls])
        r = ref.get(task["id"], {})
        rtool = r.get("tool")
        rargs = r.get("args")
        s_agree = ftool == rtool
        a_agree = s_agree and (fargs == rargs)
        agree_sel += s_agree
        agree_arg += a_agree
        print(
            f"{task['id']:<22}{str(ftool):<20}{str(rtool):<20}"
            f"{str(g['selection']):<6}{str(g['args']):<6}"
            f"{str(s_agree):<10}{str(a_agree):<10}"
        )
        rows.append(
            {
                "task": task["id"],
                "expect_tool": task["expect_tool"],
                "expect_args": task["expect_args"],
                "fast_tool": ftool,
                "fast_args": fargs,
                "ref_tool": rtool,
                "ref_args": rargs,
                "fast_selection": g["selection"],
                "fast_args_ok": g["args"],
                "agree_selection": s_agree,
                "agree_args": a_agree,
                "latency_ms": dt,
            }
        )

    n = len(TASKS)
    out = Path(__file__).with_name("parity_18_en.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print("\n" + "=" * 70)
    print(f"FAST Q8/Metal absolute correctness (EN, n={n}):")
    print(f"  selection {f_sel}/{n} = {100*f_sel/n:.1f}%   args {f_arg}/{n} = {100*f_arg/n:.1f}%")
    print("transformers fp32 reference (boosted, EN): selection 10/18, args 8/18")
    print(f"PARITY vs transformers fp32 reference (same-tool / same-args):")
    print(f"  selection agreement {agree_sel}/{n} = {100*agree_sel/n:.1f}%")
    print(f"  args agreement      {agree_arg}/{n} = {100*agree_arg/n:.1f}%")
    print(f"latency/tool-call: avg={sum(lats)/n:.0f} ms  min={min(lats):.0f}  max={max(lats):.0f}")
    print(f"raw -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
