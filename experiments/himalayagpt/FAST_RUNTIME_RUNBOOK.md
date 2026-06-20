# HimalayaGPT-0.5b fast runtime (nanochat llama.cpp fork + Metal)

Make the HimalayaGPT-0.5b run ~20-25x faster than the CPU/fp32 transformers path
by serving the **Q8_0 GGUF** on the **nanochat llama.cpp fork** with **Metal GPU
acceleration**, and drive it through himmy's existing `HimalayaGptClientManager`
`generate_fn` seam so tool-calling behaviour is unchanged — just faster.

Binaries and weights live OUTSIDE this repo at `~/himalaya-runtime/` (they are
3rd-party / large). Only the bridge + this runbook live in the repo.

## What was built (one time)

1. **Fork + Metal build** (`~/himalaya-runtime/llama.cpp/`):
   ```sh
   cd ~/himalaya-runtime
   GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/HimalayaAI/llama.cpp.git
   cd llama.cpp
   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
         -DGGML_METAL=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TOOLS=ON
   cmake --build build -j --target llama-cli llama-server
   # -> build/bin/llama-cli, build/bin/llama-server  (Metal + BLAS backends)
   ```
   Upstream llama.cpp rejects nanochat ("NanochatForCausalLM is not supported");
   the fork adds the custom RoPE / input smear / value embeddings / squared-ReLU
   MLP / mid-layer backout / output softcap, AND a nanochat-aware
   `convert_hf_to_gguf.py`.

2. **Q8_0 GGUF** (`~/himalaya-runtime/himalayagpt-0.5b-it-Q8_0.gguf`, 558 MB).
   The prebuilt GGUF repo `lukas-h/selfhosted-himalaya-gpt` is **gated/private**
   (HTTP 401 "Invalid username or password", no token available), so we converted
   the **already-cached** base safetensors (`~/.cache/huggingface/.../himalayagpt-0.5b-it`)
   with the fork's nanochat-aware converter — the auth-free, canonical path that
   produces the same Q8_0:
   ```sh
   SNAP=~/.cache/huggingface/hub/models--himalaya-ai--himalayagpt-0.5b-it/snapshots/*/
   PYTHONPATH=~/himalaya-runtime/llama.cpp/gguf-py ~/.himalayagpt-venv/bin/python \
     ~/himalaya-runtime/llama.cpp/convert_hf_to_gguf.py $SNAP \
     --outtype q8_0 --outfile ~/himalaya-runtime/himalayagpt-0.5b-it-Q8_0.gguf
   ```
   (Never build an **F16** GGUF — squared-ReLU overflows F16 -> NaN. Q8_0 weights
   are fine; bf16 was tested and gives identical tool-calling to Q8 but ~3x slower,
   so Q8 is the choice.)

## CRITICAL Metal cap: `-ngl 11` (NOT 99)

On M3 Pro, **full GPU offload (`-ngl 12`..99) of this GGUF produces degenerate
output on Metal** — a stream of `\x1f` (U+001F) tokens — while CPU (`-ngl 0`) and
partial offload (`-ngl <=11`) are coherent. The fork ships no nanochat-specific
Metal kernels; the **squared-ReLU FFN** (`ggml_relu` -> `ggml_sqr`,
`llama-graph.cpp:LLM_FFN_RELU_SQR`) overflows when layer 12+ runs on the GPU
(the same F16 squared-ReLU overflow the model card warns about, here in the Metal
compute path rather than the weights). The 15-layer model with the first 11 layers
offloaded keeps the overflow-prone tail on CPU and is still **~120-205 tok/s**
(measured), vs the **~6.7 tok/s** CPU/fp32 transformers baseline = **~20-25x**.
Bisected boundary: `-ngl 11` = correct, `-ngl 12` = garbage.

## Run the server (the integration target)

```sh
~/himalaya-runtime/llama.cpp/build/bin/llama-server \
  -m ~/himalaya-runtime/himalayagpt-0.5b-it-Q8_0.gguf \
  --host 127.0.0.1 --port 8081 -ngl 11 -c 4096 --no-warmup
# /health -> {"status":"ok"} in ~2s
```

himmy must use **`/completion`** (raw prompt), NOT `/v1/chat/completions`:
himmy COMPOSES its own nanochat prompt (special tokens `<|bos|><|user_start|>...`
+ the Hermes `<tools>` manifest + few-shot). `/v1/chat` would re-apply the GGUF
chat template and double-wrap it. `/completion` takes the raw prompt verbatim with
`parse_special: true`, temperature 0, and stops on `<|assistant_end|>`.

## Integrate into himmy

`hgpt_fast_bridge.py` is a drop-in replacement for `hgpt_bridge.HimalayaGptWorker`:
same `generate(prompt, **kw) -> str` / `generate_fn(**kw) -> (prompt)->str` surface,
so `HimalayaGptClientManager(generate_fn=...)` + the boosted
`hermes_chatml_xml_fewshot` format are unchanged — only the backend swaps from slow
CPU/fp32 transformers to the Metal Q8 GGUF.

```python
from hgpt_fast_bridge import HimalayaGptFastBridge
from himmy.services.inference.local import HimalayaGptClientManager

bridge = HimalayaGptFastBridge(base_url="http://127.0.0.1:8081")  # attach to running server
bridge.start()
mgr = HimalayaGptClientManager(generate_fn=bridge.generate_fn(max_new_tokens=64, temperature=0.0))
# mgr.generate(InferenceRequest(...)) now runs on Metal
```
(With no `base_url`, the bridge spawns its own `llama-server` at `-ngl 11`.)

## Smoke / verification scripts (run with the himmy venv `.venv/bin/python`)

- `prove_through_himmy_fast.py` — EN + NE free-form generation through the manager on
  the fast backend.
- `prove_toolcall_fast.py` — proves a REAL TOOL CALL (bound tools + executor + shipped
  EMIT/SELECTION/ARGS graders) flows through the manager on the fast backend, EN + NE,
  and reports per-call latency. 3/4 cells fully pass (incl. an NE pass); the one NE miss
  is the 0.5b's documented Nepali arg-fragility (affects both backends), not the GGUF.
- `compare_fast_vs_transformers.py` — 18-task x {en,ne} tool-calling, fast (Q8/Metal)
  vs the transformers-fp32 reference (`ab_results_powered.jsonl`).
- `greedy_head_to_head.py` — PURE GREEDY (argmax) head-to-head, the only
  apples-to-apples cross-backend comparison.

## Correctness finding (honest, full 18-task suite)

- Sanity checks pass: EN "What is 2+2?" -> coherent; NE "नेपालको राजधानी के हो?" ->
  "...काठमाडौं हो।" (**Kathmandu, correct**) on Metal at `-ngl 11`.
- **Tool-calling is NOT faithful to the transformers fp32 reference.** Full 18-task EN
  suite, through himmy's HimalayaGptClientManager (boosted `hermes_chatml_xml_fewshot`),
  identical pure-argmax (temp 0, repetition_penalty 1.0):
  - GGUF/llama.cpp tool-selection **9/18 (50%)** vs transformers fp32 **13/18 (72%)** —
    the fast path is **less accurate**, not equal-or-better.
  - Agreement only **39% selection / 33% args**.
  - Genuine regressions (fp32 right, GGUF wrong): `translate_hello_fr` &
    `translate_thanks_spanish` (fp32 -> translate_text; GGUF -> set_timer);
    `remind_meeting_3pm` (fp32 -> schedule_reminder; GGUF -> convert_currency). Only
    2 GGUF-better cases (stock tasks).
- **Root cause (diagnosed):** NOT quantization (a bf16 GGUF shows the same ~44%/33%
  divergence) and NOT Metal (the same Q8 GGUF on pure CPU, `-ngl 0`, gives the same
  39%/33%). It is a systematic difference between the HimalayaAI llama.cpp fork's
  nanochat implementation (kernels / tokenizer / sampler) and the HF transformers
  reference, amplified by the 0.5b's fragility. Tokenization-vs-numerics not yet isolated.
- **Implication:** the proven prompting-boost numbers (and the team chart) were measured
  on the SLOW transformers path; they do NOT transfer to this fast GGUF runtime as-is.
- Next: isolate whether the divergence is a fixable prompt/tokenization mismatch vs a
  fork bug (signal for the fork maintainers); a tool-calling fine-tune would improve
  robustness on both engines.
