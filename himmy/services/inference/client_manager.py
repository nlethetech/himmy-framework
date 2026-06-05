"""Inference kernel: client managers that resolve model keys and generate responses.

Generation lives in the client manager so the offline :class:`StubClientManager`
can fully simulate every response format (text, JSON, tools, structured output,
workflow) without a network or provider keys. It is the framework default.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol, runtime_checkable

from himmy.core.errors import HimmyError
from himmy.services.inference.models import (
    GatewayRuntimeConfig,
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ResponseFormat,
    ToolCallRecord,
    ToolReturnRecord,
    synthesize_from_schema,
)


@runtime_checkable
class ClientManager(Protocol):
    """Resolves a ``model_key`` to a model path and generates a response.

    Generation is part of the protocol (not just resolution) so the stub can
    fully simulate provider behavior offline.

    Failure contract (INF-12)
    -------------------------
    Implementations SHOULD self-normalize: ``generate`` returns an
    :class:`InferenceResponse` with ``status=FAILED`` and a typed
    :class:`InferenceError` rather than raising for provider/transport failures
    (``StubClientManager``/``PydanticAIClientManager`` both do this). The ONE
    sanctioned exception is :class:`NotImplementedError` for a genuinely
    unsupported path (e.g. ``ResponseFormat.TOOL``).

    Regardless of an implementation's discipline, :class:`InferenceService` wraps
    every ``generate`` call in a normalization boundary (``_run_once``), so a
    manager that *does* raise can never bypass retries, latency stamping, or the
    ``INFERENCE_FAILED`` event — ``InferenceService.run`` never raises for
    provider/manager errors. Optionally implement ``generate_stream`` (an async
    generator yielding ``str`` deltas then a final :class:`InferenceResponse`) to
    back :meth:`InferenceService.run_stream`.
    """

    def resolve(self, model_key: str) -> str:
        """Map a logical ``model_key`` to a concrete model-path string."""
        ...

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Produce an :class:`InferenceResponse` for ``request`` (must not raise)."""
        ...


def _last_user_message(request: InferenceRequest) -> str:
    """Return the content of the most recent ``user`` message, or ""."""
    for message in reversed(request.messages):
        if message.role == "user":
            return message.content
    return ""


def _system_message(request: InferenceRequest) -> str:
    """Return the first ``system`` message content, or ""."""
    for message in request.messages:
        if message.role == "system":
            return message.content
    return ""


def _has_tool_result(request: InferenceRequest) -> bool:
    """True when a tool already ran this thread (a prior ``tool``-role message exists).

    Lets the stub behave like a real model: call a tool on the first turn, then —
    having seen the result — answer on the next turn instead of looping forever.
    """
    return any(message.role == "tool" for message in request.messages)


def _estimate_tokens(text: str) -> int:
    """Deterministic, length-based token estimate (~4 chars per token, min 1)."""
    if not text:
        return 1
    return max(1, len(text) // 4)


class StubClientManager:
    """Default offline client manager — deterministic, no network, no keys.

    ``generate`` simulates every supported :class:`ResponseFormat`. For tool and
    workflow paths it executes the request's ``bound_tools`` (synthesizing args
    from each tool's schema) and records the resulting call/return pairs.
    """

    def __init__(
        self, *, default_model_path: str = "stub:echo", latency_ms: float = 1.0
    ) -> None:
        """Configure the deterministic model path and a fixed simulated latency."""
        self._default_model_path = default_model_path
        self._latency_ms = latency_ms
        self.provider_name = "stub"

    def resolve(self, model_key: str) -> str:
        """Resolve a model key to a deterministic ``stub:<key>`` path."""
        if not model_key or model_key == "default":
            return self._default_model_path
        return f"stub:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Deterministically simulate a model response for ``request``."""
        try:
            return await self._generate(request)
        except NotImplementedError:
            # TOOL format (and any other unsupported path) propagates per the doc.
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to a typed failure
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.FAILED,
                model_path=self.resolve(request.model_key),
                provider_name=self.provider_name,
                latency_ms=self._latency_ms,
                error=InferenceError(
                    code=InferenceErrorCode.UNKNOWN,
                    message=str(exc),
                    retryable=False,
                ),
            )

    async def _generate(self, request: InferenceRequest) -> InferenceResponse:
        """Core simulation dispatch by response format."""
        model_path = self.resolve(request.model_key)
        user_text = _last_user_message(request)
        system_text = _system_message(request)
        response_format = request.response_format or ResponseFormat.TEXT

        response = InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            model_path=model_path,
            provider_name=self.provider_name,
            latency_ms=self._latency_ms,
        )

        if response_format == ResponseFormat.TOOL:
            raise NotImplementedError(
                "ResponseFormat.TOOL (force one named tool) is not yet supported"
            )

        if response_format == ResponseFormat.WORKFLOW:
            await self._run_workflow(request, response)
        elif response_format == ResponseFormat.STRUCTURED_OUTPUT:
            self._run_structured(request, response, user_text)
        elif response_format in (ResponseFormat.AUTO_TOOLS, ResponseFormat.TEXT):
            if (
                request.bound_tools
                and response_format == ResponseFormat.AUTO_TOOLS
                and not _has_tool_result(request)
            ):
                await self._run_auto_tools(request, response, user_text)
            else:
                response.output_text = self._echo(system_text, user_text)
        elif response_format == ResponseFormat.JSON_OBJECT:
            obj = {"answer": self._echo(system_text, user_text)}
            response.output_structured = obj
            response.output_text = json.dumps(obj)
        else:
            response.output_text = self._echo(system_text, user_text)

        # Implicit auto-tools: no explicit format but tools were bound (and none have
        # run yet — once a tool result is on the thread, the stub answers instead).
        if (
            request.response_format is None
            and request.bound_tools
            and not response.tool_calls
            and response_format == ResponseFormat.TEXT
            and not _has_tool_result(request)
        ):
            await self._run_auto_tools(request, response, user_text)

        # Token accounting (INF-3): include the full conversation on the input
        # side and ALL produced surfaces (text, structured payload, tool args and
        # returns) on the output side — not just the final echo string.
        response.input_tokens = sum(
            _estimate_tokens(m.content) for m in request.messages
        ) or _estimate_tokens(user_text)
        output_chars = json.dumps(
            {
                "text": response.output_text or "",
                "structured": response.output_structured,
                "tool_calls": [
                    {"name": c.tool_name, "args": c.args} for c in response.tool_calls
                ],
                "tool_returns": [
                    {"name": r.tool_name, "content": r.content}
                    for r in response.tool_returns
                ],
            },
            default=str,
        )
        response.output_tokens = max(1, _estimate_tokens(output_chars))
        # The stub is free; a real price table lives on GatewayRuntimeConfig.
        response.cost = 0.0
        return response

    def _echo(self, system_text: str, user_text: str) -> str:
        """Build a short, deterministic echo answer."""
        prefix = "[stub]"
        if system_text:
            prefix = f"[stub:{system_text.strip()[:48]}]"
        body = user_text.strip() if user_text else "(no user message)"
        return f"{prefix} Acknowledged: {body[:280]}"

    def _run_structured(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        user_text: str,
    ) -> None:
        """Populate ``output_structured`` from the request's output schema."""
        if not request.output_json_schema:
            raise HimmyError(
                "STRUCTURED_OUTPUT requires output_json_schema on the request"
            )
        structured = synthesize_from_schema(
            request.output_json_schema, seed_text=user_text
        )
        response.output_structured = structured
        response.output_text = json.dumps(structured)

    async def _run_auto_tools(
        self,
        request: InferenceRequest,
        response: InferenceResponse,
        user_text: str,
    ) -> None:
        """Call each bound tool once and reference the outputs in the final text."""
        outputs: list[str] = []
        for tool in request.bound_tools:
            call, ret = await self._invoke_tool(tool, user_text)
            response.tool_calls.append(call)
            response.tool_returns.append(ret)
            outputs.append(f"{tool.name}->{ret.content!r}")
        joined = "; ".join(outputs) if outputs else "(no tools)"
        response.output_text = f"[stub] Used tools: {joined}"

    async def _run_workflow(
        self, request: InferenceRequest, response: InferenceResponse
    ) -> None:
        """Execute exactly the current step's tool; the caller advances the state."""
        state = request.workflow
        if state is None:
            raise HimmyError("WORKFLOW requires request.workflow to be set")
        tool_name = state.current_tool_name
        if tool_name is None:
            # Workflow already complete; nothing to call.
            response.output_text = "[stub] workflow complete"
            response.workflow = state
            return

        tool = next((t for t in request.bound_tools if t.name == tool_name), None)
        if tool is None:
            raise HimmyError(
                f"workflow step tool '{tool_name}' not found in bound_tools"
            )
        call, ret = await self._invoke_tool(tool, _last_user_message(request))
        response.tool_calls.append(call)
        response.tool_returns.append(ret)
        response.output_text = f"[stub] step '{tool_name}' -> {ret.content!r}"
        # Pointer is returned UNCHANGED; the caller drives progression.
        response.workflow = state

    async def generate_stream(self, request: InferenceRequest):
        """Stream the echo as deterministic chunked text deltas, then the response.

        Yields ``str`` deltas (the offline analogue of provider token streaming)
        followed by the fully-materialized :class:`InferenceResponse`, matching the
        contract :meth:`InferenceService.run_stream` consumes from a manager.
        """
        response = await self.generate(request)
        text = response.output_text or ""
        for start in range(0, len(text), 24):
            yield text[start : start + 24]
        yield response

    async def _invoke_tool(
        self, tool: Any, user_text: str
    ) -> tuple[ToolCallRecord, ToolReturnRecord]:
        """Synthesize args, run the handler if present, and record the exchange."""
        args = synthesize_from_schema(tool.args_json_schema or {}, seed_text=user_text)
        if not isinstance(args, dict):
            args = {}
        call = ToolCallRecord(
            tool_call_id=_stable_tool_call_id(tool.name),
            tool_name=tool.name,
            args=args,
        )
        if tool.handler is not None:
            ret = await tool.handler(args)
            # Preserve the synthesized call id for clean pairing.
            ret = ret.model_copy(update={"tool_call_id": call.tool_call_id})
        else:
            ret = ToolReturnRecord(
                tool_call_id=call.tool_call_id,
                tool_name=tool.name,
                content={"stub": True, "args": args},
                outcome="success",
            )
        return call, ret


def _stable_tool_call_id(tool_name: str) -> str:
    """A deterministic, readable tool-call id for the stub path."""
    import uuid

    return f"tc_{uuid.uuid4().hex[:12]}_{tool_name}"


class GatewayClientManager:
    """Production routing via the Pydantic AI Gateway / OpenAI-compatible provider.

    Resolves model keys against a :class:`GatewayRuntimeConfig` registry, then
    (INF-2/INF-11) builds a real pydantic-ai model from the registry entry's
    ``api_format`` + ``model_name`` and the gateway ``base_url``/``region`` + key,
    delegating generation to an internal :class:`PydanticAIClientManager` so the
    full request envelope (system prompt, history, tools, model settings, usage,
    cost) is honored. Without a gateway key and the ``[providers]`` extra it either
    delegates to an offline :class:`StubClientManager` (when
    ``HIMMY_GATEWAY_STUB_FALLBACK`` is set) or raises a clear, actionable error.

    Cost: when ``GatewayRuntimeConfig.model_prices`` has an entry for the model,
    the response cost is filled from the provider token usage.
    """

    def __init__(self, runtime_config: GatewayRuntimeConfig) -> None:
        """Store the routing config and prepare lazy delegate managers."""
        self._config = runtime_config
        self.provider_name = "gateway"
        self._stub: StubClientManager | None = None
        self._delegate: Any | None = None

    def resolve(self, model_key: str) -> str:
        """Resolve a model key to ``"{region}:{model_name}"`` via the registry."""
        entry = self._config.model_registry.get(model_key)
        if entry is None:
            # Fall back to treating the key itself as the model name.
            return f"{self._config.region}:{model_key}"
        return f"{self._config.region}:{entry.model_name}"

    def _provider_model_string(self, model_key: str) -> str:
        """Build a pydantic-ai model string from the registry entry.

        e.g. an entry ``api_format='openai', model_name='gpt-4o-mini'`` yields
        ``'openai:gpt-4o-mini'`` — the form ``infer_model`` understands. Unknown
        keys pass through untouched so a fully-qualified model string still works.
        """
        entry = self._config.model_registry.get(model_key)
        if entry is None:
            return model_key
        return f"{entry.api_format}:{entry.model_name}"

    def _build_delegate(self) -> Any:
        """Construct the pydantic-ai-backed delegate, configured for the gateway."""
        from himmy.services.inference.pydantic_ai_manager import (
            PydanticAIClientManager,
        )

        registry = {
            key: f"{entry.api_format}:{entry.model_name}"
            for key, entry in self._config.model_registry.items()
        }
        default_model = registry.get("default", "openai:gpt-4o-mini")
        return PydanticAIClientManager(
            registry,
            default_model=default_model,
            base_url=self._config.base_url,
            runtime_config=self._config,
            provider_name="gateway",
        )

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Route through the gateway; degrade clearly without the providers extra."""
        has_key = bool(os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY"))
        try:  # pydantic-ai is an optional extra; never import it at module top.
            import pydantic_ai  # type: ignore  # noqa: F401

            pydantic_ai_available = True
        except ImportError:
            pydantic_ai_available = False

        if has_key and pydantic_ai_available:
            # Real production path: build a gateway-configured pydantic-ai delegate
            # once and route the full request envelope through it.
            if self._delegate is None:
                self._delegate = self._build_delegate()
            return await self._delegate.generate(request)

        if os.environ.get("HIMMY_GATEWAY_STUB_FALLBACK"):
            if self._stub is None:
                self._stub = StubClientManager()
            return await self._stub.generate(request)

        raise HimmyError(
            "GatewayClientManager requires the [providers] extra and "
            "PYDANTIC_AI_GATEWAY_API_KEY (set HIMMY_GATEWAY_STUB_FALLBACK=1 "
            "to simulate offline)."
        )
