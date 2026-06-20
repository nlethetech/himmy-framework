"""A/B runner: zero-shot vs few-shot Hermes manifest on the REAL HimalayaGPT 0.5b.

Design (per the brief):
  * SAME tasks, SAME real model, temp 0 (greedy). The ONLY thing that differs between
    the two arms is the prompting:
      - Arm "zero"     -> tool_call_format="hermes_chatml_xml"          (Hermes manifest, NO few-shot)
      - Arm "fewshot"  -> tool_call_format="hermes_chatml_xml_fewshot"  (Hermes manifest + 2 exemplars + no-code nudge)
  * BOTH arms are driven THROUGH himmy: HimalayaGptClientManager(generate_fn=bridge,
    tool_call_format=ARM) -> manager.generate(InferenceRequest(bound_tools=...)). No
    hand-rolled call, no stubbing — the manager renders the manifest via the format
    registry, the real model generates, and the manager parses <tool_call> via the same
    registry.
  * Graded with the SHIPPED benchmark arg-level grader (himmy.benchmark.graders):
      - EMIT      : any parseable tool_call at all       (len(resp.tool_calls) > 0)
      - SELECTION : grade_trajectory tool_called  -> right tool name present
      - ARGS      : grade_trajectory tool_called_with -> right tool WITH matching args
    Trajectory is built from the manager's real resp.tool_calls / resp.tool_returns.

Each (task, lang, arm) trial is appended as one JSON line to the results file the moment
it completes, so the run is chunkable/resumable: already-done (id, lang, arm) trials are
skipped on restart. Pass --lang / --arm / --task-start / --task-count to chunk so no
single invocation exceeds the shell timeout (CPU 0.5b ~ a few s/trial).

Usage (py3.14 himmy venv):
  .venv/bin/python experiments/himalayagpt/ab_runner.py --lang en --arm zero
  .venv/bin/python experiments/himalayagpt/ab_runner.py --lang ne --arm fewshot --task-start 0 --task-count 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hgpt_bridge import HimalayaGptWorker  # noqa: E402
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

RESULTS = Path(__file__).with_name("ab_results.jsonl")

ARM_FORMAT = {
    "zero": "hermes_chatml_xml",
    "fewshot": "hermes_chatml_xml_fewshot",
}

# Generation budget: temp 0 (greedy), enough tokens to emit a <tool_call> block, repetition
# penalty to suppress the 0.5b's loop-to-EOS. 96 tokens comfortably fits a tool_call line.
GEN_KWARGS = {"max_new_tokens": 96, "temperature": 0.0, "repetition_penalty": 1.15}


async def _stub_executor(name: str, args: dict) -> ToolReturnRecord:
    """A trivial executor so the manager's tool-return path runs (results not graded for
    correctness here — we grade EMIT/SELECTION/ARGS off the parsed call). Returns a fixed
    string so result_contains would work if ever needed; selection/args are off the call."""
    return ToolReturnRecord(
        tool_call_id="", tool_name=name, content=f"ok:{name}:{json.dumps(args)}"
    )


def _load_done() -> set[tuple[str, str, str]]:
    """Set of (task_id, lang, arm) already recorded, for resumable chunked runs."""
    done: set[tuple[str, str, str]] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((rec["task_id"], rec["lang"], rec["arm"]))
    return done


def _grade(task: dict, traj: Trajectory) -> dict:
    """Grade one trajectory with the SHIPPED benchmark graders."""
    emit = len(traj.calls) > 0
    selection = grade_trajectory({"type": "tool_called", "value": task["expect_tool"]}, traj)
    args_ok = grade_trajectory(
        {
            "type": "tool_called_with",
            "tool": task["expect_tool"],
            "args": task["expect_args"],
        },
        traj,
    )
    return {"emit": emit, "selection": selection, "args": args_ok}


async def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "ne"], required=True)
    ap.add_argument("--arm", choices=["zero", "fewshot"], required=True)
    ap.add_argument("--task-start", type=int, default=0)
    ap.add_argument("--task-count", type=int, default=len(TASKS))
    args = ap.parse_args()

    tasks = TASKS[args.task_start : args.task_start + args.task_count]
    done = _load_done()
    todo = [t for t in tasks if (t["id"], args.lang, args.arm) not in done]
    if not todo:
        print(f"[skip] all {len(tasks)} (lang={args.lang}, arm={args.arm}) already done")
        return

    worker = HimalayaGptWorker()
    worker.start()
    print(f"[worker] READY — running {len(todo)} trials (lang={args.lang}, arm={args.arm})")

    # The manager is the ONLY interface to the model. Inject the bridge generate_fn and
    # pin the arm's tool-call format; the format registry renders the manifest + parses.
    manager = HimalayaGptClientManager(
        generate_fn=worker.generate_fn(**GEN_KWARGS),
        tool_call_format=ARM_FORMAT[args.arm],
    )

    with RESULTS.open("a", encoding="utf-8") as fh:
        for task in todo:
            prompt = prompt_for(task, args.lang)
            req = InferenceRequest(
                model_key="default",
                messages=[InferenceMessage(role="user", content=prompt)],
                bound_tools=TOOLS,
                tool_executor=_stub_executor,
            )
            t0 = time.perf_counter()
            resp = await manager.generate(req)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            assert resp.status is InferenceStatus.SUCCESS, resp.error

            traj = Trajectory.from_turns([(resp.tool_calls, resp.tool_returns)])
            grades = _grade(task, traj)
            emitted_names = [c.tool_name for c in resp.tool_calls]
            emitted_args = [c.args for c in resp.tool_calls]

            rec = {
                "task_id": task["id"],
                "lang": args.lang,
                "arm": args.arm,
                "prompt": prompt,
                "expect_tool": task["expect_tool"],
                "expect_args": task["expect_args"],
                "emitted_tools": emitted_names,
                "emitted_args": emitted_args,
                "raw_output": resp.output_text,
                "emit": grades["emit"],
                "selection": grades["selection"],
                "args": grades["args"],
                "latency_ms": round(latency_ms, 1),
                "provider": resp.provider_name,
            }
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            mark = "EMIT" if grades["emit"] else "----"
            sel = "SEL" if grades["selection"] else "   "
            ag = "ARG" if grades["args"] else "   "
            print(
                f"  [{mark} {sel} {ag}] {task['id']:<24} {latency_ms:7.0f}ms "
                f"-> {emitted_names or 'NONE'}"
            )

    worker.close()
    print("[done] chunk complete")


if __name__ == "__main__":
    asyncio.run(run())
