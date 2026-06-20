"""himmy-side (py3.14) bridge to the HimalayaGPT worker running in the py3.12 venv.

This module imports NO torch/transformers. It spawns hgpt_worker.py with the
himalayagpt venv's python, holds the long-lived process, and exposes a
``generate_fn(prompt) -> str`` callable that is exactly the shape
``HimalayaGptClientManager(generate_fn=...)`` expects. That is the chosen
integration seam: in-process deps in himmy/.venv are NOT viable because
himmy/.venv is Python 3.14 while the model + its custom nanochat tokenizer.pkl
run on Python 3.12 — so we drive the loaded model through the manager's existing
generate_fn injection seam over a subprocess line protocol instead.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable

_VENV_PY = os.path.expanduser("~/.himalayagpt-venv/bin/python")
_WORKER = str(Path(__file__).with_name("hgpt_worker.py"))


class HimalayaGptWorker:
    """Long-lived subprocess that serves nanochat generations over stdio."""

    def __init__(self, *, default_kwargs: dict | None = None) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._default_kwargs = default_kwargs or {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "repetition_penalty": 1.15,
        }

    def start(self) -> None:
        if self._proc is not None:
            return
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        self._proc = subprocess.Popen(
            [_VENV_PY, _WORKER],
            cwd=str(Path(__file__).parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
        )
        # Block until the model is loaded and the worker says READY.
        assert self._proc.stdout is not None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(f"worker died before READY: {err.strip()}")
            if line.strip() == "READY":
                break

    def generate(self, prompt: str, **overrides) -> str:
        with self._lock:
            if self._proc is None:
                self.start()
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            req = {**self._default_kwargs, **overrides, "prompt": prompt}
            self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise RuntimeError(f"worker died during generate: {err.strip()}")
            resp = json.loads(line)
            if not resp.get("ok"):
                raise RuntimeError(f"worker error: {resp.get('error')}")
            return str(resp["text"])

    def generate_fn(self, **overrides) -> Callable[[str], str]:
        """Return a ``(prompt) -> str`` callable for HimalayaGptClientManager."""

        def _fn(prompt: str) -> str:
            return self.generate(prompt, **overrides)

        return _fn

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        finally:
            self._proc = None


_SHARED: HimalayaGptWorker | None = None


def shared_worker() -> HimalayaGptWorker:
    """Process-wide singleton worker (one model load per python process)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = HimalayaGptWorker()
        _SHARED.start()
    return _SHARED
