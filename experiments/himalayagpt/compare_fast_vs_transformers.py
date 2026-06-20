"""Correctness bar: do tool-calling results THROUGH himmy on the Q8 GGUF (Metal,
llama.cpp fork) MATCH the transformers fp32 reference?

The reference is ab_results_powered.jsonl (the transformers worker, arm "new" =
the shipping boosted hermes_chatml_xml_fewshot format). This runs the SAME 18
tasks x {en,ne} through HimalayaGptClientManager with the SAME format, but with
the generate_fn backed by HimalayaGptFastBridge (llama-server /completion, Metal),
and compares, per (task,lang): emitted tool name and emitted args.

Match definition (per the brief: "same tool selected + same args"):
  - SELECTION matches when the fast backend selects the SAME tool the reference
    selected (both lists compared as the first emitted tool, or both NONE).
  - ARGS matches when, given the same tool, the emitted args dicts are equal.
We report the per-cell agreement and the overall selection/args agreement, plus
each disagreement so any material divergence is visible (then bf16 is the fallback).

Run with himmy's venv; a llama-server must be up on :8081 (ngl=11) or one is spawned.
    .venv/bin/python experiments/himalayagpt/compare_fast_vs_transformers.py
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

from himmy.services.inference.local import HimalayaGptClientManager  # noqa: E402
from himmy.services.inference.models import (  # noqa: E402
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    ToolReturnRecord,
)

REF = Path(__file__).with_name("ab_results_powered.jsonl")
OUT = Path(__file__).with_name("fast_vs_transformers.jsonl")
# Match the reference run's generation budget exactly (ab_runner.GEN_KWARGS).
GEN_KWARGS = {"max_new_tokens": 64, "temperature": 0.0, "repetition_penalty": 1.15}
ARM = "new"  # the shipping boosted format
FORMAT = "hermes_chatml_xml_fewshot"


async def _stub_executor(name: str, args: dict) -> ToolReturnRecord:
    return ToolReturnRecord(
        tool_call_id="", tool_name=name, content=f"ok:{name}:{json.dumps(args)}"
    )


def _load_reference() -> dict[tuple[str, str], dict]:
    ref: dict[tuple[str, str], dict] = {}
    for line in REF.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("arm") != ARM:
            continue
        ref[(r["task_id"], r["lang"])] = r
    return ref


def _first(lst):
    return lst[0] if lst else None


async def main() -> None:
    ref = _load_reference()
    print(f"[ref] loaded {len(ref)} transformers-fp32 reference cells (arm={ARM})")

    bridge = HimalayaGptFastBridge(base_url="http://127.0.0.1:8081")
    try:
        bridge.start(timeout_s=5)
        print("[bridge] attached to running llama-server :8081 (Metal, ngl=11)")
    except Exception:
        bridge = HimalayaGptFastBridge()
        bridge.start()
        print("[bridge] spawned llama-server (Metal, ngl=11)")

    manager = HimalayaGptClientManager(
        generate_fn=bridge.generate_fn(**GEN_KWARGS),
        tool_call_format=FORMAT,
    )

    sel_agree = args_agree = total = 0
    sel_correct_fast = sel_correct_ref = 0
    disagreements: list[str] = []
    t_start = time.perf_counter()

    with OUT.open("w", encoding="utf-8") as fh:
        for lang in ("en", "ne"):
            for task in TASKS:
                key = (task["id"], lang)
                if key not in ref:
                    continue
                total += 1
                prompt = prompt_for(task, lang)
                req = InferenceRequest(
                    model_key="default",
                    messages=[InferenceMessage(role="user", content=prompt)],
                    bound_tools=TOOLS,
                    tool_executor=_stub_executor,
                )
                resp = await manager.generate(req)
                assert resp.status is InferenceStatus.SUCCESS, resp.error

                fast_tools = [c.tool_name for c in resp.tool_calls]
                fast_args = [c.args for c in resp.tool_calls]
                ref_tools = ref[key]["emitted_tools"]
                ref_args = ref[key]["emitted_args"]
                expect = task["expect_tool"]

                fast_tool0, ref_tool0 = _first(fast_tools), _first(ref_tools)
                sel_match = fast_tool0 == ref_tool0
                args_match = sel_match and (_first(fast_args) == _first(ref_args))
                sel_agree += sel_match
                args_agree += args_match
                sel_correct_fast += fast_tool0 == expect
                sel_correct_ref += ref_tool0 == expect

                rec = {
                    "task_id": task["id"], "lang": lang, "expect_tool": expect,
                    "fast_tools": fast_tools, "fast_args": fast_args,
                    "ref_tools": ref_tools, "ref_args": ref_args,
                    "selection_match": sel_match, "args_match": args_match,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if not args_match:
                    disagreements.append(
                        f"  [{lang}] {task['id']}: fast={fast_tool0}{_first(fast_args)} "
                        f"!= ref={ref_tool0}{_first(ref_args)} (expect={expect})"
                    )

    dt = time.perf_counter() - t_start
    print(f"\n=== FAST(GGUF/Metal) vs transformers-fp32, {total} cells, {dt:.1f}s ===")
    print(f"selection agreement : {sel_agree}/{total} = {100*sel_agree/total:.1f}%")
    print(f"args agreement      : {args_agree}/{total} = {100*args_agree/total:.1f}%")
    print(f"selection-correct (vs gold): fast={sel_correct_fast}/{total}  "
          f"ref={sel_correct_ref}/{total}")
    if disagreements:
        print(f"\n--- {len(disagreements)} arg-level disagreements ---")
        for d in disagreements:
            print(d)
    else:
        print("\n[PASS] fast backend matches transformers reference on every cell.")


if __name__ == "__main__":
    asyncio.run(main())
