"""Persistent stdin/stdout worker that serves HimalayaGPT generations.

Runs in ~/.himalayagpt-venv (py3.12). One model load, then a request/response
loop over stdio using length-free single-line JSON (newlines in payloads are
escaped by json). The himmy side (py3.14) spawns this once and pipes prompts in.

Protocol (one JSON object per line each way):
  request : {"prompt": str, "max_new_tokens": int, "temperature": float, ...}
  response: {"ok": true, "text": str}  |  {"ok": false, "error": str}
A line == "PING" replies "PONG\n" (readiness probe). EOF on stdin -> clean exit.
"""

from __future__ import annotations

import json
import sys

import hgpt_loader


def main() -> None:
    # Warm the model before signaling readiness so the first real request is fast.
    hgpt_loader.load()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        if line == "PING":
            sys.stdout.write("PONG\n")
            sys.stdout.flush()
            continue
        try:
            req = json.loads(line)
            prompt = req.pop("prompt")
            text = hgpt_loader.generate(prompt, **req)
            out = {"ok": True, "text": text}
        except Exception as exc:  # noqa: BLE001 - report, keep serving
            out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
