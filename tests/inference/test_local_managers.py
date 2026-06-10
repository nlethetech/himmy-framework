"""Tests for local-model client managers + cost-aware routing (all offline)."""

from __future__ import annotations

from typing import Any

from himmy.services.inference import (
    ClaudeCliClientManager,
    HimalayaGptClientManager,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    InferenceStatus,
    OllamaClientManager,
    Route,
    RoutingClientManager,
)
from himmy.services.inference.models import InferenceError, InferenceErrorCode
from tests.conftest import run_async


def _req() -> InferenceRequest:
    return InferenceRequest(
        messages=[
            InferenceMessage(role="system", content="be brief"),
            InferenceMessage(role="user", content="hello"),
        ],
        generation_params={"temperature": 0.1, "max_tokens": 50},
    )


# ----------------------------------------------------------------- Ollama
def test_ollama_maps_chat_reply() -> None:
    """A fake Ollama /api/chat reply maps to a SUCCESS response with token counts."""
    seen: dict[str, Any] = {}

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        seen["path"] = path
        seen["payload"] = payload
        return {
            "message": {"content": "नमस्ते from ollama"},
            "prompt_eval_count": 12,
            "eval_count": 7,
        }

    mgr = OllamaClientManager(model="llama3.2", transport=transport)
    resp = run_async(mgr.generate(_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "नमस्ते from ollama"
    assert resp.input_tokens == 12 and resp.output_tokens == 7
    assert resp.model_path == "ollama:llama3.2"
    assert seen["path"] == "/api/chat"
    assert seen["payload"]["options"]["temperature"] == 0.1
    assert seen["payload"]["options"]["num_predict"] == 50


def test_ollama_failure_is_normalized() -> None:
    """A transport error becomes a FAILED response (never raises)."""

    def transport(path: str, payload: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("server down")

    resp = run_async(OllamaClientManager(transport=transport).generate(_req()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.error is not None


# ----------------------------------------------------------------- Claude CLI
def test_claude_cli_runs_and_captures_output() -> None:
    """The CLI runner is invoked with --model and the prompt; output is captured."""
    captured: dict[str, Any] = {}

    def runner(argv: list[str], stdin: str) -> str:
        captured["argv"] = argv
        captured["stdin"] = stdin
        return "  haiku says hi\n"

    mgr = ClaudeCliClientManager(model="haiku", runner=runner)
    resp = run_async(mgr.generate(_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "haiku says hi"
    assert resp.model_path == "claude-cli:haiku"
    assert "--model" in captured["argv"] and "haiku" in captured["argv"]
    assert "hello" in captured["stdin"]


def test_claude_cli_parses_json_usage_and_cost() -> None:
    """With JSON output the CLI's token usage + total_cost_usd are surfaced."""
    import json

    captured: dict[str, Any] = {}

    def runner(argv: list[str], stdin: str) -> str:
        captured["argv"] = argv
        return json.dumps(
            {
                "type": "result",
                "result": "pong",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 12,
                    "cache_read_input_tokens": 20,
                },
                "total_cost_usd": 0.0034,
            }
        )

    resp = run_async(
        ClaudeCliClientManager(model="haiku", runner=runner).generate(_req())
    )
    assert resp.output_text == "pong"  # the JSON envelope is unwrapped
    assert resp.input_tokens == 120  # 100 + 20 cache-read
    assert resp.output_tokens == 12
    assert resp.cost == 0.0034
    # the manager asks the CLI for JSON output
    assert "--output-format" in captured["argv"]


def test_claude_cli_failure_is_normalized() -> None:
    """A runner error becomes a FAILED response."""

    def runner(argv: list[str], stdin: str) -> str:
        raise RuntimeError("not logged in")

    resp = run_async(ClaudeCliClientManager(runner=runner).generate(_req()))
    assert resp.status == InferenceStatus.FAILED


# ----------------------------------------------------------------- HimalayaGPT
def test_himalayagpt_uses_injected_generate_fn() -> None:
    """HimalayaGPT routes generation through the injected function (no model load)."""
    mgr = HimalayaGptClientManager(generate_fn=lambda prompt: f"echo:{len(prompt)}")
    resp = run_async(mgr.generate(_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text.startswith("echo:")
    assert resp.provider_name == "himalayagpt"


# ----------------------------------------------------- cost-aware routing
class _Manager:
    def __init__(self, provider: str, ok: bool) -> None:
        self.provider = provider
        self.ok = ok
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return model_key

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        if self.ok:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text=self.provider,
                provider_name=self.provider,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=InferenceErrorCode.PROVIDER_UNAVAILABLE, message="down"
            ),
            provider_name=self.provider,
        )


def test_cost_ordered_prefers_cheapest_route() -> None:
    """cost_ordered tries the free local route before the paid cloud route."""
    local = _Manager("local", ok=True)
    cloud = _Manager("cloud", ok=True)
    router = RoutingClientManager.cost_ordered(
        [
            Route(manager=cloud, cost=3.0, label="cloud"),
            Route(manager=local, cost=0.0, label="local"),
        ]
    )
    resp = run_async(router.generate(_req()))
    assert resp.provider_name == "local"  # cheapest tried first
    assert cloud.calls == 0
    assert resp.metadata["route_index"] == 0


def test_cost_ordered_escalates_to_cloud_on_local_failure() -> None:
    """A failed free route fails over to the paid cloud route."""
    local = _Manager("local", ok=False)
    cloud = _Manager("cloud", ok=True)
    router = RoutingClientManager.cost_ordered(
        [Route(manager=cloud, cost=3.0), Route(manager=local, cost=0.0)]
    )
    resp = run_async(InferenceService(router).run(_req()))
    assert resp.provider_name == "cloud"
    assert local.calls >= 1 and cloud.calls >= 1


def test_ollama_timeout_env_knob(monkeypatch) -> None:
    """HIMMY_OLLAMA_TIMEOUT sets the default request budget; bad values fall back.

    The knob exists for slow hosts (CPU-only CI runners) where generation
    legitimately exceeds the 120s default. An explicit ``timeout=`` always wins.
    """
    monkeypatch.delenv("HIMMY_OLLAMA_TIMEOUT", raising=False)
    assert OllamaClientManager()._timeout == 120.0

    monkeypatch.setenv("HIMMY_OLLAMA_TIMEOUT", "300")
    assert OllamaClientManager()._timeout == 300.0
    # An explicit constructor timeout overrides the env.
    assert OllamaClientManager(timeout=42.0)._timeout == 42.0

    # Misconfiguration must not take the offline path down.
    monkeypatch.setenv("HIMMY_OLLAMA_TIMEOUT", "not-a-number")
    assert OllamaClientManager()._timeout == 120.0
    monkeypatch.setenv("HIMMY_OLLAMA_TIMEOUT", "-5")
    assert OllamaClientManager()._timeout == 120.0


def test_default_ollama_model_resolution(monkeypatch) -> None:
    """build_manager_for('ollama') picks an INSTALLED chat model, not llama3.2.

    HIMMY_OLLAMA_MODEL overrides; embedding models are skipped; a down server
    falls back to the historical default so offline behavior is unchanged.
    """
    import httpx

    from himmy.cli.provider import _default_ollama_model

    monkeypatch.setenv("HIMMY_OLLAMA_MODEL", "custom:7b")
    assert _default_ollama_model() == "custom:7b"
    monkeypatch.delenv("HIMMY_OLLAMA_MODEL", raising=False)

    def fake_get(url, timeout):  # noqa: ANN001, ANN202
        request = httpx.Request("GET", url)
        return httpx.Response(
            200,
            json={
                "models": [
                    {"name": "qwen3-embedding:latest"},
                    {"name": "qwen2.5:3b-instruct"},
                ]
            },
            request=request,
        )

    monkeypatch.setattr(httpx, "get", fake_get)
    assert _default_ollama_model() == "qwen2.5:3b-instruct"

    def down_get(url, timeout):  # noqa: ANN001, ANN202
        raise httpx.ConnectError("down")

    monkeypatch.setattr(httpx, "get", down_get)
    assert _default_ollama_model() == "llama3.2"
