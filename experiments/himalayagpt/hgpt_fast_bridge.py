"""FAST himmy-side bridge to HimalayaGPT-0.5b via the nanochat llama.cpp fork.

Drop-in replacement for :class:`hgpt_bridge.HimalayaGptWorker`: it exposes the
exact same ``generate(prompt, **overrides) -> str`` / ``generate_fn(**overrides)
-> Callable[[str], str]`` surface that ``HimalayaGptClientManager(generate_fn=...)``
expects, so the manager + the boosted ``hermes_chatml_xml_fewshot`` format (and
therefore the tool-calling behaviour) are byte-for-byte unchanged — only the
backend swaps from slow CPU/fp32 transformers to the Metal-accelerated Q8_0 GGUF.

Why ``/completion`` (raw prompt) and NOT ``/v1/chat/completions``:
  himmy COMPOSES its OWN nanochat prompt — the special tokens
  ``<|bos|><|user_start|>...`` plus the Hermes ``<tools>`` manifest + few-shot
  exemplars — and hands the manager a fully-rendered string. ``/v1/chat`` would
  re-apply the GGUF's chat template and DOUBLE-wrap that prompt. ``/completion``
  takes the raw prompt verbatim; with ``parse_special: true`` the server tokenizes
  himmy's special tokens correctly, and we stop on ``<|assistant_end|>`` so the
  reply ends exactly where the transformers worker's would.

The model is loaded ONCE by ``llama-server`` (Metal); this bridge is a thin HTTP
client. Either point it at an already-running server (``base_url``) or let it spawn
and own one (``server_bin`` + ``model_path``).

CRITICAL — ``n_gpu_layers`` is capped at 11 (NOT 99), measured on M3 Pro:
  Full GPU offload (``-ngl 12``..99) of this nanochat GGUF produces DEGENERATE
  output on Metal — a stream of ``\\x1f`` (U+001F) tokens — while CPU (``-ngl 0``)
  and partial offload (``-ngl <=11``) are coherent. The fork ships no nanochat-
  specific Metal kernels; the squared-ReLU FFN (``ggml_relu`` -> ``ggml_sqr``,
  llama-graph.cpp LLM_FFN_RELU_SQR) overflows when layer 12+ runs on the GPU
  (the same F16 squared-ReLU overflow the model card warns about, here in the
  Metal compute path rather than the weights). The 15-layer model with the first
  11 layers offloaded keeps the overflow-prone tail on CPU and still gives
  ~120-205 tok/s vs the ~6.7 tok/s CPU/fp32 transformers baseline (~20-25x).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

_HOME = os.path.expanduser("~")
_DEFAULT_SERVER = f"{_HOME}/himalaya-runtime/llama.cpp/build/bin/llama-server"
_DEFAULT_MODEL = f"{_HOME}/himalaya-runtime/himalayagpt-0.5b-it-Q8_0.gguf"

#: nanochat assistant-turn terminator: himmy's prompt opens an assistant turn and
#: the model closes it with this token. Stopping here yields exactly the assistant's
#: reply (mirrors the transformers worker, which decodes up to the same boundary).
_STOP = ["<|assistant_end|>", "<|bos|>", "<|user_start|>"]


class HimalayaGptFastBridge:
    """llama.cpp ``/completion``-backed worker with the slow worker's interface."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        server_bin: str = _DEFAULT_SERVER,
        model_path: str = _DEFAULT_MODEL,
        host: str = "127.0.0.1",
        port: int = 8081,
        n_gpu_layers: int = 11,
        ctx_size: int = 4096,
        default_kwargs: dict | None = None,
    ) -> None:
        # If base_url is given we attach to an existing server and never spawn one.
        self._external = base_url is not None
        self._base_url = (base_url or f"http://{host}:{port}").rstrip("/")
        self._server_bin = server_bin
        self._model_path = model_path
        self._host = host
        self._port = port
        self._n_gpu_layers = n_gpu_layers
        self._ctx_size = ctx_size
        self._proc: subprocess.Popen | None = None
        # nanochat squared-ReLU overflows F16 -> NaN; the Q8_0 weights are fine, but
        # generation is greedy (temp 0) to match the transformers reference exactly.
        self._default_kwargs = default_kwargs or {
            "max_new_tokens": 256,
            "temperature": 0.0,
            "repetition_penalty": 1.15,
        }

    # -- lifecycle -----------------------------------------------------------
    def start(self, *, timeout_s: float = 120.0) -> None:
        """Spawn llama-server (unless attached to an external one) and block on /health."""
        if self._external:
            self._wait_healthy(timeout_s)
            return
        if self._proc is not None:
            return
        if not Path(self._server_bin).exists():
            raise FileNotFoundError(f"llama-server not found: {self._server_bin}")
        if not Path(self._model_path).exists():
            raise FileNotFoundError(f"GGUF not found: {self._model_path}")
        argv = [
            self._server_bin,
            "-m", self._model_path,
            "--host", self._host,
            "--port", str(self._port),
            "-ngl", str(self._n_gpu_layers),
            "-c", str(self._ctx_size),
            "--no-warmup",
        ]
        self._proc = subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._wait_healthy(timeout_s)

    def _wait_healthy(self, timeout_s: float) -> None:
        deadline = time.time() + timeout_s
        last_err: Exception | None = None
        while time.time() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early (code {self._proc.returncode})"
                )
            try:
                with urllib.request.urlopen(self._base_url + "/health", timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception as exc:  # noqa: BLE001 - server not up yet, keep polling
                last_err = exc
            time.sleep(0.5)
        raise RuntimeError(f"llama-server not healthy after {timeout_s}s: {last_err}")

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            self._proc.kill()
        finally:
            self._proc = None

    # -- generation ----------------------------------------------------------
    def generate(self, prompt: str, **overrides) -> str:
        """POST the RAW prompt to /completion (parse_special) and return the text.

        The kwargs match the transformers worker's vocabulary
        (``max_new_tokens`` / ``temperature`` / ``repetition_penalty``) so the
        manager and the A/B runner inject identical values; they are mapped onto
        llama.cpp's ``n_predict`` / ``temperature`` / ``repeat_penalty``.
        """
        kw = {**self._default_kwargs, **overrides}
        payload = {
            "prompt": prompt,
            "n_predict": int(kw.get("max_new_tokens", 256)),
            "temperature": float(kw.get("temperature", 0.0)),
            "repeat_penalty": float(kw.get("repetition_penalty", 1.0)),
            # honor himmy's <|bos|>/<|user_start|>/... special tokens in the raw prompt
            "parse_special": True,
            # do NOT echo the prompt back in the reply
            "stop": _STOP,
            "cache_prompt": True,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._base_url + "/completion",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:  # noqa: BLE001
            raise RuntimeError(f"llama-server /completion failed: {exc}") from exc
        return str(body.get("content", ""))

    def generate_fn(self, **overrides) -> Callable[[str], str]:
        """Return a ``(prompt) -> str`` callable for HimalayaGptClientManager."""

        def _fn(prompt: str) -> str:
            return self.generate(prompt, **overrides)

        return _fn


_SHARED: HimalayaGptFastBridge | None = None


def shared_fast_bridge(**kwargs) -> HimalayaGptFastBridge:
    """Process-wide singleton fast bridge (one server per python process)."""
    global _SHARED
    if _SHARED is None:
        _SHARED = HimalayaGptFastBridge(**kwargs)
        _SHARED.start()
    return _SHARED
