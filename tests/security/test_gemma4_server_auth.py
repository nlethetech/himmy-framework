"""Regression test for the gemma4 OpenAI-compat server auth hole (Finding #8).

The shim at ``scripts/gemma4_server/serve.py`` used to expose an unauthenticated,
GPU-backed ``/v1/chat/completions`` endpoint and a ``--host`` flag with no warning,
so ``--host 0.0.0.0`` handed any LAN host a generator they could drive or DoS.

These tests verify the fix WITHOUT touching the real model or torch/MPS:
  - a request with no / wrong Bearer is 401 when GEMMA4_API_KEY is set,
  - the correct Bearer is allowed,
  - a non-loopback bind without a token is refused at startup.

We stub the heavy deps (``torch``, ``transformers``, ``gemma_args``) so the module
imports anywhere, and monkeypatch ``_generate`` so no model ever runs.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SERVE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "gemma4_server"


@pytest.fixture
def serve_module(monkeypatch):
    """Import scripts/gemma4_server/serve.py with heavy deps stubbed out."""
    # --- stub torch (only torch.backends.mps.is_available + torch.no_grad used at import) ---
    torch_stub = types.ModuleType("torch")
    backends = types.ModuleType("torch.backends")
    mps = types.ModuleType("torch.backends.mps")
    mps.is_available = lambda: False  # type: ignore[attr-defined]
    backends.mps = mps  # type: ignore[attr-defined]
    torch_stub.backends = backends  # type: ignore[attr-defined]
    torch_stub.bfloat16 = "bfloat16"  # type: ignore[attr-defined]

    class _NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    torch_stub.no_grad = lambda: _NoGrad()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.backends", backends)
    monkeypatch.setitem(sys.modules, "torch.backends.mps", mps)

    # --- stub transformers (referenced only inside _load(), which we never call) ---
    monkeypatch.setitem(sys.modules, "transformers", types.ModuleType("transformers"))

    # --- stub gemma_args.extract_tool_calls ---
    gemma_args_stub = types.ModuleType("gemma_args")
    gemma_args_stub.extract_tool_calls = lambda raw: (raw, [])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "gemma_args", gemma_args_stub)

    # make the script dir importable, and ensure a fresh module each test
    monkeypatch.syspath_prepend(str(_SERVE_DIR))
    sys.modules.pop("serve", None)
    serve = importlib.import_module("serve")

    # avoid any model use: stub generation + mark "loaded"
    monkeypatch.setattr(
        serve, "_generate", lambda req: ("hello from stub", 1, 2), raising=True
    )
    monkeypatch.setattr(serve, "_model", object(), raising=False)
    yield serve
    sys.modules.pop("serve", None)


def _payload() -> dict:
    return {"model": "x", "messages": [{"role": "user", "content": "hi"}]}


def test_missing_bearer_is_401(serve_module, monkeypatch):
    monkeypatch.setenv("GEMMA4_API_KEY", "s3cret-token")
    client = TestClient(serve_module._app)
    r = client.post("/v1/chat/completions", json=_payload())
    assert r.status_code == 401


def test_wrong_bearer_is_401(serve_module, monkeypatch):
    monkeypatch.setenv("GEMMA4_API_KEY", "s3cret-token")
    client = TestClient(serve_module._app)
    r = client.post(
        "/v1/chat/completions",
        json=_payload(),
        headers={"Authorization": "Bearer not-the-token"},
    )
    assert r.status_code == 401


def test_correct_bearer_is_allowed(serve_module, monkeypatch):
    monkeypatch.setenv("GEMMA4_API_KEY", "s3cret-token")
    client = TestClient(serve_module._app)
    r = client.post(
        "/v1/chat/completions",
        json=_payload(),
        headers={"Authorization": "Bearer s3cret-token"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "hello from stub"


def test_non_loopback_bind_without_token_is_refused(serve_module, monkeypatch):
    monkeypatch.delenv("GEMMA4_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        serve_module._check_bind("0.0.0.0", has_token=False)  # noqa: S104 - testing the guard


def test_non_loopback_bind_with_token_is_allowed(serve_module):
    # should NOT raise
    serve_module._check_bind("0.0.0.0", has_token=True)  # noqa: S104 - testing the guard


def test_loopback_bind_without_token_is_allowed(serve_module):
    serve_module._check_bind("127.0.0.1", has_token=False)
