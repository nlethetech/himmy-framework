"""Does bf16 GGUF restore parity with the transformers fp32 reference? (the task's
'try bf16 if Q8 diverges materially' branch). Runs the full 18-task EN suite through
himmy on the bf16 GGUF (CPU, :8083), identical pure-argmax, and compares to BOTH the
live fp32 reference and the Q8 results captured in parity_h2h_18_en.jsonl.
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

GEN = {"max_new_tokens": 64, "temperature": 0.0, "repetition_penalty": 1.0}


async def _stub(name, args):
    return ToolReturnRecord(tool_call_id="", tool_name=name, content="ok")


def _first(lst):
    return lst[0] if lst else None


async def main() -> None:
    h2h = {}
    for line in (Path(__file__).with_name("parity_h2h_18_en.jsonl")).read_text().splitlines():
        r = json.loads(line)
        h2h[r["task"]] = r  # has slow_tool/slow_args (fp32 ref) + fast_tool/fast_args (Q8 metal)

    bridge = HimalayaGptFastBridge(base_url="http://127.0.0.1:8083")
    bridge.start(timeout_s=10)
    mgr = HimalayaGptClientManager(
        generate_fn=bridge.generate_fn(**GEN), tool_call_format="hermes_chatml_xml_fewshot"
    )
    bf16 = {}
    for task in TASKS:
        req = InferenceRequest(
            model_key="default",
            messages=[InferenceMessage(role="user", content=prompt_for(task, "en"))],
            bound_tools=TOOLS,
            tool_executor=_stub,
        )
        resp = await mgr.generate(req)
        assert resp.status is InferenceStatus.SUCCESS, resp.error
        bf16[task["id"]] = (
            _first([c.tool_name for c in resp.tool_calls]),
            _first([c.args for c in resp.tool_calls]),
        )

    n = len(TASKS)
    bf_fp32_sel = bf_fp32_arg = bf_q8_sel = 0
    print(f"{'task':<22}{'fp32':<18}{'Q8_metal':<18}{'bf16_cpu':<18}")
    for task in TASKS:
        r = h2h[task["id"]]
        rt, ra = r["slow_tool"], r["slow_args"]
        qt = r["fast_tool"]
        bt, ba = bf16[task["id"]]
        bf_fp32_sel += bt == rt
        bf_fp32_arg += (bt == rt) and (ba == ra)
        bf_q8_sel += bt == qt
        print(f"{task['id']:<22}{str(rt):<18}{str(qt):<18}{str(bt):<18}")

    print("\n" + "=" * 70)
    print(f"bf16 GGUF (CPU) vs transformers fp32 reference (n={n}):")
    print(f"  selection agreement {bf_fp32_sel}/{n} = {100*bf_fp32_sel/n:.1f}%")
    print(f"  args agreement      {bf_fp32_arg}/{n} = {100*bf_fp32_arg/n:.1f}%")
    print(f"bf16 GGUF vs Q8 GGUF selection agreement: {bf_q8_sel}/{n} = {100*bf_q8_sel/n:.1f}%")
    print("\nReminder: Q8(metal) vs fp32 was selection 7/18 (39%), args 6/18 (33%).")
    print("VERDICT: if bf16 also ~39% vs fp32 -> divergence is GGUF/llama.cpp, not the")
    print("         quant level; bf16 buys no parity. If bf16 >> Q8 -> use bf16.")


if __name__ == "__main__":
    asyncio.run(main())
