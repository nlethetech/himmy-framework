"""Inference kernel: local / self-hosted client managers (Ollama, Claude CLI, HimalayaGPT).

First-class managers for the models Nepal teams actually run locally — so an agent
can run on a laptop with no cloud cost. Each implements the ``ClientManager``
contract (``resolve`` + ``generate`` that never raises for provider failures) and
takes an injectable transport/runner/generate-fn, so the whole layer is testable
offline. Compose them with :class:`RoutingClientManager` for cost-aware routing
(local/free first, cloud only on failure).

* :class:`OllamaClientManager` — Ollama's HTTP chat API (``/api/chat``).
* :class:`ClaudeCliClientManager` — the local ``claude`` CLI (Claude Max), via
  subprocess (NOT an HTTP API).
* :class:`HimalayaGptClientManager` — a self-hosted HF Transformers model.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)


def _failed(
    request: InferenceRequest,
    *,
    message: str,
    provider: str,
    model_path: str,
    started: float,
    code: InferenceErrorCode = InferenceErrorCode.PROVIDER_UNAVAILABLE,
) -> InferenceResponse:
    """Normalize any local-provider failure into a FAILED response (never raise)."""
    return InferenceResponse(
        request_id=request.request_id,
        status=InferenceStatus.FAILED,
        error=InferenceError(code=code, message=message, retryable=True),
        provider_name=provider,
        model_path=model_path,
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )


def _chat_messages(request: InferenceRequest) -> list[dict[str, str]]:
    """Project the request's messages into ``[{role, content}]`` (chat APIs)."""
    return [{"role": str(m.role), "content": m.content} for m in request.messages]


def _compose_prompt(request: InferenceRequest) -> str:
    """Flatten the conversation into a single labeled prompt (CLI / LM inputs)."""
    labels = {
        "system": "[System]",
        "user": "[User]",
        "assistant": "[Assistant]",
        "tool": "[Tool result]",
    }
    parts = [
        f"{labels.get(str(m.role).lower(), '[User]')}\n{m.content}"
        for m in request.messages
        if m.content
    ]
    return "\n\n".join(parts)


class OllamaClientManager:
    """A :class:`ClientManager` backed by a local Ollama server's chat API."""

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        model_registry: dict[str, str] | None = None,
        transport: Callable[[str, dict[str, Any]], Any] | None = None,
        timeout: float = 120.0,
        provider_name: str = "ollama",
    ) -> None:
        """Configure the default model, server URL, and (test) transport."""
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._registry = dict(model_registry or {})
        self._transport = transport
        self._timeout = timeout
        self.provider_name = provider_name

    def resolve(self, model_key: str) -> str:
        """Map a model key to an Ollama model tag (registry, else the key/default)."""
        if not model_key or model_key == "default":
            return self._registry.get("default", self._model)
        return self._registry.get(model_key, model_key)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Call Ollama ``/api/chat`` and map the reply onto an InferenceResponse."""
        model = self.resolve(request.model_key)
        started = time.perf_counter()
        params = request.generation_params or {}
        options: dict[str, Any] = {}
        if params.get("temperature") is not None:
            options["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            options["num_predict"] = params["max_tokens"]
        payload: dict[str, Any] = {
            "model": model,
            "messages": _chat_messages(request),
            "stream": False,
        }
        if options:
            payload["options"] = options
        try:
            data = await self._post("/api/chat", payload, request.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - normalize to FAILED
            return _failed(
                request,
                message=f"ollama request failed: {exc}",
                provider=self.provider_name,
                model_path=f"ollama:{model}",
                started=started,
            )
        text = ((data.get("message") or {}).get("content")) or ""
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=text,
            model_path=f"ollama:{model}",
            provider_name=self.provider_name,
            input_tokens=int(data.get("prompt_eval_count", 0) or 0),
            output_tokens=int(data.get("eval_count", 0) or 0),
            cost=0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _post(
        self, path: str, payload: dict[str, Any], timeout: float
    ) -> dict[str, Any]:
        if self._transport is not None:
            result = self._transport(path, payload)
            if isinstance(result, Awaitable):
                return await result  # type: ignore[no-any-return]
            return result
        import httpx

        async with httpx.AsyncClient(timeout=timeout or self._timeout) as client:
            response = await client.post(self._base_url + path, json=payload)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]


class ClaudeCliClientManager:
    """A :class:`ClientManager` that drives the local ``claude`` CLI (Claude Max)."""

    def __init__(
        self,
        *,
        executable: str = "claude",
        model: str = "haiku",
        model_registry: dict[str, str] | None = None,
        extra_args: list[str] | None = None,
        runner: Callable[[list[str], str], Any] | None = None,
        timeout: float = 120.0,
        provider_name: str = "claude-cli",
    ) -> None:
        """Configure the CLI executable, default model, and (test) runner."""
        self._executable = executable
        self._model = model
        self._registry = dict(model_registry or {})
        self._extra_args = list(extra_args or [])
        self._runner = runner
        self._timeout = timeout
        self.provider_name = provider_name

    def resolve(self, model_key: str) -> str:
        """Map a model key to a CLI ``--model`` alias (registry, else key/default)."""
        if not model_key or model_key == "default":
            return self._registry.get("default", self._model)
        return self._registry.get(model_key, model_key)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run ``claude -p --model <m>`` with the prompt on stdin; capture stdout."""
        model = self.resolve(request.model_key)
        started = time.perf_counter()
        argv = [self._executable, "-p", "--model", model, *self._extra_args]
        try:
            out = await self._run(
                argv, _compose_prompt(request), request.timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001 - normalize to FAILED
            return _failed(
                request,
                message=f"claude CLI failed: {exc}",
                provider=self.provider_name,
                model_path=f"claude-cli:{model}",
                started=started,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=out.strip(),
            model_path=f"claude-cli:{model}",
            provider_name=self.provider_name,
            cost=0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _run(self, argv: list[str], stdin: str, timeout: float) -> str:
        if self._runner is not None:
            result = self._runner(argv, stdin)
            if isinstance(result, Awaitable):
                return await result  # type: ignore[no-any-return]
            return str(result)
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(
            proc.communicate(stdin.encode("utf-8")), timeout=timeout or self._timeout
        )
        if proc.returncode != 0:
            raise RuntimeError(
                err.decode("utf-8", "replace").strip() or "claude CLI nonzero exit"
            )
        return out.decode("utf-8", "replace")


class HimalayaGptClientManager:
    """A :class:`ClientManager` for a self-hosted HF Transformers model (HimalayaGPT)."""

    def __init__(
        self,
        *,
        model_name: str = "HimalayaAI/HimalayaGPT-0.5B",
        generate_fn: Callable[[str], str] | None = None,
        max_new_tokens: int = 256,
        provider_name: str = "himalayagpt",
    ) -> None:
        """Configure the model id and (test) generate function."""
        self._model_name = model_name
        self._generate_fn = generate_fn
        self._max_new_tokens = max_new_tokens
        self.provider_name = provider_name
        self._pipe: Any = None

    def resolve(self, model_key: str) -> str:
        """Resolve to the configured model id (registry-less single model)."""
        if not model_key or model_key == "default":
            return self._model_name
        return model_key

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text on a worker thread (Transformers inference is blocking)."""
        started = time.perf_counter()
        try:
            text = await asyncio.to_thread(self._infer, _compose_prompt(request))
        except Exception as exc:  # noqa: BLE001 - normalize to FAILED
            return _failed(
                request,
                message=f"HimalayaGPT generation failed: {exc}",
                provider=self.provider_name,
                model_path=self._model_name,
                started=started,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=text,
            model_path=self._model_name,
            provider_name=self.provider_name,
            cost=0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _infer(self, prompt: str) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(prompt)
        if self._pipe is None:  # pragma: no cover - heavy model load, not in CI
            from transformers import pipeline

            self._pipe = pipeline(
                "text-generation", model=self._model_name, trust_remote_code=True
            )
        out = self._pipe(
            prompt, max_new_tokens=self._max_new_tokens, return_full_text=False
        )
        return str(out[0]["generated_text"])


__all__ = [
    "OllamaClientManager",
    "ClaudeCliClientManager",
    "HimalayaGptClientManager",
]
