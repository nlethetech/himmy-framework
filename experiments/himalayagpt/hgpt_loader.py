"""Reproducible loader + generator for the REAL HimalayaGPT 0.5b (nanochat arch).

Runs in the dedicated py3.12 venv (~/.himalayagpt-venv) where torch 2.12 /
transformers==4.46.3 / tiktoken / blobfile are installed. The himmy side
(py3.14) NEVER imports torch; it drives this module through hgpt_worker.py's
line protocol via HimalayaGptClientManager(generate_fn=...).

Load recipe (resolves the safetensors-missing-metadata crash):
  - AutoModelForCausalLM.from_pretrained crashes on 4.46.3 because the exported
    model.safetensors has NO metadata header (metadata.get("format") -> AttributeError).
  - FIX: instantiate the model from the remote-code config via `from_config`
    (trust_remote_code), then load weights MANUALLY with safetensors.torch.load_file
    and `model.load_state_dict(strict=False)`. This sidesteps the metadata probe
    entirely and gives a clean load (0 missing / 0 unexpected keys).
  - device=cpu, dtype=float32 (squared-ReLU MLP overflows fp16 -> NaN; the custom
    nanochat ops are not MPS-safe).

Chat format (nanochat): <|bos|><|user_start|>{user}<|user_end|><|assistant_start|>{assistant}<|assistant_end|>
System messages merge into the next user turn. Generation stops on <|assistant_end|>.
The model has NO KV cache (use_cache=False), so decoding re-runs the full prefix
each step; fine for short benchmark completions.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REPO_ID = "himalaya-ai/himalayagpt-0.5b-it"

_SPECIALS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]

_lock = threading.Lock()
_model = None
_tok = None
_ids: dict[str, int] = {}


def load() -> None:
    """Load the model + tokenizer once (idempotent, thread-safe)."""
    global _model, _tok, _ids
    if _model is not None:
        return
    with _lock:
        if _model is not None:
            return
        local = snapshot_download(REPO_ID, local_files_only=True)
        cfg = AutoConfig.from_pretrained(local, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        state_dict = load_file(local + "/model.safetensors", device="cpu")
        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"state_dict mismatch: missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}"
            )
        model = model.to(device="cpu", dtype=torch.float32).eval()
        tok = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
        ids = {t: int(tok._special_to_id[t]) for t in _SPECIALS if t in tok._special_to_id}
        _model, _tok, _ids = model, tok, ids


def _render_turn(prompt: str) -> list[int]:
    """Render one user turn in the nanochat chat format -> token ids."""
    uids = _tok(prompt, add_special_tokens=False)["input_ids"]
    return (
        [_ids["<|bos|>"], _ids["<|user_start|>"]]
        + uids
        + [_ids["<|user_end|>"], _ids["<|assistant_start|>"]]
    )


def _apply_repetition_penalty(logits: torch.Tensor, seq: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty <= 1.0 or seq.numel() == 0:
        return logits
    uniq = torch.unique(seq)
    out = logits.clone()
    sel = out[:, uniq]
    out[:, uniq] = torch.where(sel < 0, sel * penalty, sel / penalty)
    return out


def generate(
    prompt: str,
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.95,
    repetition_penalty: float = 1.15,
    seed: int = 0,
) -> str:
    """Generate a completion for a single user prompt. Greedy when temperature==0."""
    load()
    prompt_ids = _render_turn(prompt)
    ids = torch.tensor([prompt_ids], dtype=torch.long)
    stop = {_ids["<|assistant_end|>"], _ids["<|bos|>"]}
    gen = None
    if temperature > 0:
        gen = torch.Generator()
        gen.manual_seed(seed)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = _model(input_ids=ids).logits[:, -1, :]
            logits = _apply_repetition_penalty(logits, ids[0], repetition_penalty)
            if temperature <= 0:
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    k = min(top_k, logits.size(-1))
                    v, _ = torch.topk(logits, k)
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                if 0 < top_p < 1:
                    sp, si = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sp, dim=-1)
                    drop = cum - sp > top_p
                    sp[drop] = 0.0
                    sp = sp / sp.sum(dim=-1, keepdim=True)
                    choice = torch.multinomial(sp, 1, generator=gen)
                    nxt = si.gather(-1, choice)
                else:
                    nxt = torch.multinomial(probs, 1, generator=gen)
            ids = torch.cat([ids, nxt], dim=1)
            if int(nxt.item()) in stop:
                break
    out = ids[0, len(prompt_ids):]
    text = _tok.decode(out, skip_special_tokens=False)
    return _strip_specials(text).strip()


def _strip_specials(text: str) -> str:
    out = text
    for tok in _SPECIALS:
        out = out.replace(tok, "")
    return out


if __name__ == "__main__":
    load()
    for p in ["What is 2+2?", "नेपालको राजधानी के हो?"]:
        print("\n=== PROMPT:", p, "===")
        print("OUTPUT:", repr(generate(p, max_new_tokens=64, temperature=0.0)))
