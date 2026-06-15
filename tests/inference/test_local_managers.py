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


def test_claude_cli_structured_output_is_nudged_and_parsed() -> None:
    """STRUCTURED_OUTPUT requests put the schema in the prompt and parse the reply.

    The text-only CLI has no provider-native schema constraint (unlike Ollama's
    ``format`` field), so the manager must nudge via the prompt and extract the JSON
    object from the reply — this is what the benchmark judge tier relies on.
    """
    from himmy.services.inference.models import ResponseFormat

    captured: dict[str, Any] = {}
    schema = {
        "type": "object",
        "properties": {"score": {"type": "number"}},
        "required": ["score"],
    }

    def runner(argv: list[str], stdin: str) -> str:
        captured["stdin"] = stdin
        return 'Here is the verdict:\n{"score": 0.9, "rationale": "good"}'

    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="grade this")],
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema=schema,
    )
    resp = run_async(ClaudeCliClientManager(model="haiku", runner=runner).generate(req))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_structured == {"score": 0.9, "rationale": "good"}
    assert '"score"' in captured["stdin"]  # the schema was put in the prompt


def test_claude_cli_structured_output_unparseable_reply_is_none() -> None:
    """A prose reply with no JSON object yields output_structured=None (not a crash)."""
    from himmy.services.inference.models import ResponseFormat

    def runner(argv: list[str], stdin: str) -> str:
        return "I cannot produce JSON right now."

    req = InferenceRequest(
        messages=[InferenceMessage(role="user", content="grade this")],
        response_format=ResponseFormat.STRUCTURED_OUTPUT,
        output_json_schema={"type": "object"},
    )
    resp = run_async(ClaudeCliClientManager(model="haiku", runner=runner).generate(req))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_structured is None


def test_claude_cli_failure_is_normalized() -> None:
    """A runner error becomes a FAILED response."""

    def runner(argv: list[str], stdin: str) -> str:
        raise RuntimeError("not logged in")

    resp = run_async(ClaudeCliClientManager(runner=runner).generate(_req()))
    assert resp.status == InferenceStatus.FAILED


def test_claude_cli_kills_subprocess_on_timeout(monkeypatch: Any) -> None:
    """On a per-call timeout the spawned ``claude`` process is reaped (no orphan)."""
    import asyncio

    class _FakeProc:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.terminated = False
            self.killed = False
            self.stdin = None

        async def communicate(self, _data: bytes) -> tuple[bytes, bytes]:
            # Hang forever so ``asyncio.wait_for`` raises TimeoutError.
            await asyncio.sleep(3600)
            raise AssertionError("communicate should have been cancelled")

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode if self.returncode is not None else 0

    proc = _FakeProc()

    async def _fake_create_subprocess_exec(*_a: Any, **_k: Any) -> _FakeProc:
        return proc

    monkeypatch.setattr(
        asyncio, "create_subprocess_exec", _fake_create_subprocess_exec
    )

    mgr = ClaudeCliClientManager(model="haiku")

    async def _drive() -> None:
        try:
            await mgr._run(["claude"], "hello", timeout=0.05)
        except TimeoutError:
            return
        raise AssertionError("expected a timeout")

    run_async(_drive())
    # The orphaned subprocess was reaped (terminate ran; process is no longer alive).
    assert proc.terminated is True
    assert proc.returncode is not None


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


def test_slow_local_floor_beats_the_30s_request_default() -> None:
    """The service's proportional ceiling honors a manager's declared
    min_timeout_seconds — the framework's 30s request default must not kill
    legitimate slow-CPU generations (the cause of the 2026-06-10 nightly reds:
    every Ollama call died at ~32s on the shared runner)."""
    import asyncio

    class _SlowManager:
        provider_name = "slow"
        min_timeout_seconds = 120.0

        def resolve(self, model_key: str) -> str:
            return "slow:model"

        async def generate(self, request):  # noqa: ANN001, ANN202
            # Longer than the 30s-derived ceiling (~31.5s scaled down by the
            # test's tiny grace factors), shorter than the declared floor.
            await asyncio.sleep(0.05)
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text="made it",
            )

    # Shrink the proportionality: un-floored ceiling = 0.001*0.001 = 1µs (the
    # call dies); floored ceiling = 120*0.001 = 0.12s (the 0.05s call survives).
    # The sleep only outlives the ceiling if the 120s floor was honored.
    service = InferenceService(
        _SlowManager(),
        timeout_grace_seconds=0.0,
        timeout_grace_factor=0.001,
    )
    request = _req()
    request.timeout_seconds = 0.001
    resp = run_async(service.run(request))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.output_text == "made it"


def test_ollama_effective_timeout_floors_the_request_value(monkeypatch) -> None:
    """OllamaClientManager passes max(request timeout, manager budget) to the
    HTTP layer, mirroring the CLI manager's floor."""
    from himmy.services.inference.local import OllamaClientManager

    seen: dict[str, float] = {}

    async def fake_post(self, path, payload, timeout):  # noqa: ANN001, ANN202
        seen["timeout"] = timeout
        return {"message": {"role": "assistant", "content": "ok"}, "done": True}

    monkeypatch.setattr(OllamaClientManager, "_post", fake_post)
    mgr = OllamaClientManager(timeout=200.0)
    request = _req()
    request.timeout_seconds = 30.0
    run_async(mgr.generate(request))
    assert seen["timeout"] == 200.0
