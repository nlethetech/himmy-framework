"""Interactive chat CLI for the in-house HimalayaGPT-0.5b-it model.

A tiny REPL that talks to the model through the SAME fast llama.cpp bridge the himmy
experiments use (``hgpt_fast_bridge.HimalayaGptFastBridge``), but composes the model's
NATIVE nanochat chat prompt (``<|bos|><|user_start|>...<|assistant_start|>``) directly.

Why native framing and not himmy's HimalayaGptClientManager here: the manager's no-tool
path flattens turns to plain ``[User]\\n...`` text, which this 0.5b model rambles on
(verified: "What is 2+2?" -> an unrelated essay). Fed its own chat special tokens the
SAME model answers cleanly ("...2 + 2 = 4"). So a chat CLI uses the native format.

The model's trained context is only 2048 tokens, so history is trimmed to a budget.

Prereqs (one time) — see experiments/himalayagpt/FAST_RUNTIME_RUNBOOK.md:
  * the nanochat llama.cpp fork built at ~/himalaya-runtime/llama.cpp/build/bin/llama-server
  * the Q8_0 GGUF at ~/himalaya-runtime/himalayagpt-0.5b-it-Q8_0.gguf

Usage:
  # auto: attach to a llama-server on :8081 if one is up, else spawn one (-ngl 11)
  .venv/bin/python experiments/himalayagpt/himalaya_cli.py

  # talk to an already-running server explicitly
  .venv/bin/python experiments/himalayagpt/himalaya_cli.py --base-url http://127.0.0.1:8081

In-chat commands:  /reset  (clear history)   /exit or /quit   (Ctrl-D / Ctrl-C also exit)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hgpt_fast_bridge import HimalayaGptFastBridge  # noqa: E402

# nanochat chat special tokens (the model's native chat template).
_BOS = "<|bos|>"
_U0, _U1 = "<|user_start|>", "<|user_end|>"
_A0, _A1 = "<|assistant_start|>", "<|assistant_end|>"

#: The model is trained on a 2048-token window; keep the rendered prompt comfortably
#: under it so the newest turns + the reply always fit. ~3.0 chars/token is a safe
#: lower bound for mixed English/Nepali, so this char budget ~= 1500 tokens of history.
_HISTORY_CHAR_BUDGET = 4500


def _render(history: list[tuple[str, str]], user_msg: str) -> str:
    """Render the nanochat chat prompt for ``history`` + a new user turn.

    ``history`` is a list of (user, assistant) turn pairs (oldest first). Oldest pairs
    are dropped until the rendered prompt fits the char budget, so a long session never
    overflows the 2048-token model context.
    """
    turns = list(history)
    while True:
        parts = [_BOS]
        for u, a in turns:
            parts.append(f"{_U0}{u}{_U1}{_A0}{a}{_A1}")
        parts.append(f"{_U0}{user_msg}{_U1}{_A0}")
        prompt = "".join(parts)
        if len(prompt) <= _HISTORY_CHAR_BUDGET or not turns:
            return prompt
        turns.pop(0)  # drop the oldest turn pair and re-render


def _connect(base_url: str | None, *, spawn: bool) -> HimalayaGptFastBridge:
    """Attach to a running llama-server, else (or if --spawn) spawn one at -ngl 11."""
    if not spawn:
        try:
            bridge = HimalayaGptFastBridge(base_url=base_url or "http://127.0.0.1:8081")
            bridge.start(timeout_s=5)
            print(f"[bridge] attached to llama-server at {bridge._base_url}")
            return bridge
        except Exception as exc:  # noqa: BLE001 - fall through to spawning our own
            print(f"[bridge] no server to attach to ({exc}); spawning one…")
    bridge = HimalayaGptFastBridge()  # spawn + own (Metal-safe -ngl 11)
    bridge.start()
    print(f"[bridge] spawned llama-server at {bridge._base_url} (-ngl 11)")
    return bridge


def main() -> None:
    ap = argparse.ArgumentParser(description="Chat with HimalayaGPT-0.5b-it.")
    ap.add_argument(
        "--base-url",
        default=None,
        help="URL of a running llama-server (default: attach to :8081, else spawn).",
    )
    ap.add_argument(
        "--spawn",
        action="store_true",
        help="Always spawn a fresh llama-server instead of attaching to one.",
    )
    ap.add_argument("--max-tokens", type=int, default=256, help="Max tokens per reply.")
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy/most coherent for this 0.5b).",
    )
    args = ap.parse_args()

    bridge = _connect(args.base_url, spawn=args.spawn)
    gen = bridge.generate_fn(
        max_new_tokens=args.max_tokens, temperature=args.temperature
    )

    print("\n  HimalayaGPT-0.5b-it — type a message (English or नेपाली).")
    print("  Commands: /reset  /exit\n")

    history: list[tuple[str, str]] = []
    try:
        while True:
            try:
                user_msg = input("you> ").strip()
            except EOFError:
                print()
                break
            if not user_msg:
                continue
            if user_msg in {"/exit", "/quit"}:
                break
            if user_msg == "/reset":
                history.clear()
                print("[history cleared]\n")
                continue

            prompt = _render(history, user_msg)
            try:
                reply = gen(prompt).strip()
            except Exception as exc:  # noqa: BLE001 - keep the REPL alive on a hiccup
                print(f"[error] {exc}\n")
                continue
            print(f"himalaya> {reply}\n")
            history.append((user_msg, reply))
    except KeyboardInterrupt:
        print()
    finally:
        # Only shut the server down if THIS process spawned it (attached servers stay up).
        bridge.close()
        print("[bye]")


if __name__ == "__main__":
    main()
