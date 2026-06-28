"""Deterministic correctness proof: same himmy manager + format, PURE GREEDY
(temp 0, NO repetition penalty -> argmax), driven once by the SLOW transformers
fp32 worker and once by the FAST Q8 GGUF (llama.cpp/Metal) bridge. Argmax decoding
is the only apples-to-apples comparison across backends (repetition penalty is
implemented differently by transformers vs llama.cpp, so it injects spurious
divergence on a fragile 0.5b). If the Q8 quant is faithful, greedy tool selection
and args agree.

Subset-able via --task-count to stay within the slow worker's CPU budget.

Run with himmy's venv (py3.14):
    .venv/bin/python experiments/himalayagpt/greedy_head_to_head.py --task-count 6
A llama-server must be up on :8081 (ngl=11) for the fast arm; the slow arm spawns
the transformers worker in ~/.himalayagpt-venv.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hgpt_bridge import HimalayaGptWorker  # noqa: E402
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
# PURE greedy: temp 0, repetition_penalty 1.0 (== argmax). The ONLY decoding both
# backends implement identically. max_new_tokens matches the A/B budget.
GEN = {"max_new_tokens": 64, "temperature": 0.0, "repetition_penalty": 1.0}


async def _stub(name: str, args: dict) -> ToolReturnRecord:
    return ToolReturnRecord(tool_call_id="", tool_name=name, content="ok")


def _first(lst):
    return lst[0] if lst else None


async def _run_arm(generate_fn, tasks, langs):
    mgr = HimalayaGptClientManager(generate_fn=generate_fn, tool_call_format=FORMAT)
    out = {}
    for lang in langs:
        for task in tasks:
            req = InferenceRequest(
                model_key="default",
                messages=[InferenceMessage(role="user", content=prompt_for(task, lang))],
                bound_tools=TOOLS,
                tool_executor=_stub,
            )
            resp = await mgr.generate(req)
            assert resp.status is InferenceStatus.SUCCESS, resp.error
            out[(task["id"], lang)] = (
                _first([c.tool_name for c in resp.tool_calls]),
                _first([c.args for c in resp.tool_calls]),
            )
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-count", type=int, default=6)
    ap.add_argument("--langs", default="en,ne")
    args = ap.parse_args()
    tasks = TASKS[: args.task_count]
    langs = args.langs.split(",")

    print(f"[greedy h2h] {len(tasks)} tasks x {langs} = {len(tasks)*len(langs)} cells")

    fast = HimalayaGptFastBridge(base_url="http://127.0.0.1:8081")
    fast.start(timeout_s=10)
    print("[fast] attached to llama-server :8081 (Metal, ngl=11)")
    fast_out = await _run_arm(fast.generate_fn(**GEN), tasks, langs)
    print("[fast] done")

    slow = HimalayaGptWorker()
    slow.start()
    print("[slow] transformers fp32 worker READY")
    slow_out = await _run_arm(slow.generate_fn(**GEN), tasks, langs)
    slow.close()
    print("[slow] done")

    sel = argm = total = 0
    print("\n  cell                         expect        fast                 slow")
    for lang in langs:
        for task in tasks:
            key = (task["id"], lang)
            ft, fa = fast_out[key]
            st, sa = slow_out[key]
            total += 1
            s_ok = ft == st
            a_ok = s_ok and fa == sa
            sel += s_ok
            argm += a_ok
            flag = "OK " if a_ok else ("sel" if s_ok else "XXX")
            print(f"  [{flag}] {task['id']:<22}{lang}  {task['expect_tool']:<13} "
                  f"{str(ft):<20} {str(st)}")
    print("\n=== PURE-GREEDY agreement (fast Q8/Metal vs slow transformers-fp32) ===")
    print(f"selection: {sel}/{total} = {100*sel/total:.1f}%")
    print(f"args     : {argm}/{total} = {100*argm/total:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
