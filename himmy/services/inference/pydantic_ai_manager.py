"""Inference kernel: optional pydantic-ai-backed client manager (local/gateway path).

pydantic-ai is an OPTIONAL extra. This module imports cleanly without it; the
dependency is required only when :meth:`PydanticAIClientManager.generate` actually
runs. The adapter honors the full :class:`InferenceRequest` envelope:

* the system message becomes the Agent's ``instructions`` (INF-1);
* prior messages are reconstructed into ``message_history`` (INF-1);
* bound tools / toolsets are attached so the model can call them and the resulting
  ``tool_calls``/``tool_returns`` are captured (INF-1);
* ``temperature``/``max_tokens``/``top_p`` are forwarded as ``model_settings`` and
  the per-call timeout is honored (INF-1);
* ``result.usage()`` fills ``input_tokens``/``output_tokens`` and a price table
  fills ``cost`` (INF-3);
* WORKFLOW uses ``agent.iter()`` and breaks right after the first ``CallToolsNode``
  so the model is never re-prompted with the tool result (INF-6);
* :meth:`generate_stream` backs ``InferenceService.run_stream`` via
  ``agent.run_stream`` (INF-7).

Every real call is behind ``_require_pydantic_ai()``; the FILE must import even
when pydantic_ai is absent, and these paths are providers-gated in tests.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError
from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ResponseFormat,
    ToolCallRecord,
    ToolReturnRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.inference.models import GatewayRuntimeConfig


def _require_pydantic_ai() -> Any:
    """Import and return the ``pydantic_ai`` module or raise a clear error."""
    try:
        import pydantic_ai  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only with the extra
        raise HimmyError(
            "pydantic-ai not installed; pip install 'himmy[providers]'"
        ) from exc
    return pydantic_ai


class PydanticAIClientManager:
    """pydantic-ai client manager (local-dev and gateway-delegate path).

    Provider keys/base_url are read from the environment / construction; this
    manager builds the Agent, runs it, and maps the result back onto an
    :class:`InferenceResponse`. The file imports without pydantic-ai present.
    """

    def __init__(
        self,
        model_registry: dict[str, str] | None = None,
        *,
        default_model: str = "openai:gpt-4o-mini",
        base_url: str | None = None,
        runtime_config: GatewayRuntimeConfig | None = None,
        provider_name: str = "pydantic_ai",
    ) -> None:
        """Store a ``model_key -> pydantic-ai model string`` registry + options."""
        self._model_registry = dict(model_registry or {})
        self._default_model = default_model
        self._base_url = base_url
        self._runtime_config = runtime_config
        self.provider_name = provider_name

    def resolve(self, model_key: str) -> str:
        """Map a model key to a pydantic-ai-resolvable model string."""
        if not model_key or model_key == "default":
            return self._model_registry.get("default", self._default_model)
        return self._model_registry.get(model_key, model_key)

    # ----------------------------------------------------------- model assembly
    def _infer_model(self, model_string: str) -> Any:
        """Resolve a pydantic-ai model, configuring base_url for the gateway path."""
        pydantic_ai = _require_pydantic_ai()
        if self._base_url:
            # Gateway / OpenAI-compatible: route through the configured base_url.
            try:
                from pydantic_ai.models.openai import OpenAIModel  # type: ignore
                from pydantic_ai.providers.openai import OpenAIProvider  # type: ignore

                # model_string looks like 'openai:gpt-4o-mini'; take the suffix.
                model_name = model_string.split(":", 1)[-1]
                provider = OpenAIProvider(base_url=self._base_url)
                return OpenAIModel(model_name, provider=provider)
            except Exception:  # noqa: BLE001 - fall back to plain inference
                pass
        return pydantic_ai.models.infer_model(model_string)  # type: ignore[attr-defined]

    def _build_agent(self, request: InferenceRequest, model: Any) -> Any:
        """Build an Agent with system instructions, tools, and output type."""
        pydantic_ai = _require_pydantic_ai()
        agent_kwargs: dict[str, Any] = {}
        system_text = self._system_prompt(request)
        if system_text:
            agent_kwargs["instructions"] = system_text
        output_type = self._output_type_for(request)
        if output_type is not None:
            agent_kwargs["output_type"] = output_type
        tools = self._build_tools(request)
        if tools:
            agent_kwargs["tools"] = tools
        return pydantic_ai.Agent(model, **agent_kwargs)  # type: ignore[attr-defined]

    def _build_tools(self, request: InferenceRequest) -> list[Any]:
        """Convert bound tools into pydantic-ai Tools routed to their handlers.

        Only the tools allowed by ``tool_names_override`` (when set) are exposed,
        so WORKFLOW shows exactly the current step's tool.
        """
        pydantic_ai = _require_pydantic_ai()
        allowed = (
            set(request.tool_names_override)
            if request.tool_names_override is not None
            else None
        )
        tools: list[Any] = []
        for bound in request.bound_tools:
            if allowed is not None and bound.name not in allowed:
                continue
            tools.append(self._make_tool(pydantic_ai, bound))
        # Caller-supplied toolsets pass straight through.
        tools.extend(request.toolsets or [])
        return tools

    @staticmethod
    def _make_tool(pydantic_ai: Any, bound: Any) -> Any:
        """Build one pydantic-ai Tool from a BoundTool, routing to its handler.

        The tool's ``args_json_schema`` is handed to pydantic-ai explicitly via
        ``Tool.from_schema`` so the provider advertises the REAL parameter names/types to
        the model. Without it, a ``**kwargs`` runner exposes no parameters and the model
        guesses arg names (e.g. calling ``sql_query`` with ``query`` instead of ``sql``).
        """

        async def _runner(**kwargs: Any) -> Any:
            if bound.handler is None:
                return {"stub": True, "args": kwargs}
            ret: ToolReturnRecord = await bound.handler(kwargs)
            return ret.content

        _runner.__name__ = bound.name
        schema = bound.args_json_schema or {"type": "object", "properties": {}}
        return pydantic_ai.Tool.from_schema(  # type: ignore[attr-defined]
            _runner,
            name=bound.name,
            description=bound.description or "",
            json_schema=schema,
        )

    # ---------------------------------------------------------------- entry pts
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run the request through pydantic-ai (requires the ``[providers]`` extra)."""
        if request.response_format == ResponseFormat.WORKFLOW:
            return await self._generate_workflow(request)

        model_string = self.resolve(request.model_key)
        started = time.perf_counter()
        try:
            model = self._infer_model(model_string)
            agent = self._build_agent(request, model)
            result = await agent.run(
                self._user_prompt(request),
                message_history=self._to_message_history(request),
                model_settings=self._model_settings(request) or None,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            return self._map_result(request, result, model_string, latency_ms)
        except HimmyError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider failures
            return self._failed(request, model_string, started, exc)

    async def _generate_workflow(self, request: InferenceRequest) -> InferenceResponse:
        """WORKFLOW via ``agent.iter()``: break after the first CallToolsNode.

        Filters the toolset to ``tool_names_override`` (the current step) and stops
        right after the model's first tool-call node executes, so the model is
        never re-prompted with the tool result. The caller advances the state.
        """
        pydantic_ai = _require_pydantic_ai()
        model_string = self.resolve(request.model_key)
        started = time.perf_counter()
        try:
            model = self._infer_model(model_string)
            agent = self._build_agent(request, model)
            async with agent.iter(  # type: ignore[attr-defined]
                self._user_prompt(request),
                message_history=self._to_message_history(request),
                model_settings=self._model_settings(request) or None,
            ) as run:
                node = run.next_node
                while not pydantic_ai.Agent.is_end_node(node):  # type: ignore[attr-defined]
                    node = await run.next(node)
                    # Break right AFTER the first CallToolsNode executes.
                    if (
                        isinstance(node, pydantic_ai.messages.CallToolsNode)
                        or type(node).__name__ == "CallToolsNode"
                    ):
                        node = await run.next(node)
                        break
            latency_ms = (time.perf_counter() - started) * 1000.0
            response = self._map_result(request, run, model_string, latency_ms)
            response.workflow = request.workflow  # caller advances
            return response
        except HimmyError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize provider failures
            return self._failed(request, model_string, started, exc)

    async def generate_stream(self, request: InferenceRequest):
        """Stream text deltas via ``agent.run_stream`` then the final response (INF-7)."""
        model_string = self.resolve(request.model_key)
        started = time.perf_counter()
        model = self._infer_model(model_string)
        agent = self._build_agent(request, model)
        last = ""
        async with agent.run_stream(  # type: ignore[attr-defined]
            self._user_prompt(request),
            message_history=self._to_message_history(request),
            model_settings=self._model_settings(request) or None,
        ) as result:
            async for chunk in result.stream_text(delta=False):
                delta = chunk[len(last) :]
                last = chunk
                if delta:
                    yield delta
            latency_ms = (time.perf_counter() - started) * 1000.0
            yield self._map_result(request, result, model_string, latency_ms)

    # ------------------------------------------------------------- request maps
    def _system_prompt(self, request: InferenceRequest) -> str:
        """Join all system messages into the Agent's instructions string."""
        parts = [m.content for m in request.messages if m.role == "system"]
        return "\n\n".join(p for p in parts if p)

    def _user_prompt(self, request: InferenceRequest) -> str:
        """The most recent user message is the current turn's prompt."""
        for message in reversed(request.messages):
            if message.role == "user":
                return message.content
        # No user message: fall back to concatenated non-system content.
        parts = [m.content for m in request.messages if m.role != "system"]
        return "\n".join(p for p in parts if p)

    def _to_message_history(self, request: InferenceRequest) -> Any:
        """Reconstruct pydantic-ai ``message_history`` from prior messages.

        System messages become instructions (handled on the agent) and the final
        user message is the current prompt, so both are excluded here. The
        remaining user/assistant turns are rebuilt into ModelRequest/ModelResponse
        objects.
        """
        pydantic_ai = _require_pydantic_ai()
        messages = pydantic_ai.messages  # type: ignore[attr-defined]

        # Identify the final user message (the current prompt) to exclude it.
        last_user_idx = -1
        for idx in range(len(request.messages) - 1, -1, -1):
            if request.messages[idx].role == "user":
                last_user_idx = idx
                break

        history: list[Any] = []
        for idx, m in enumerate(request.messages):
            if m.role == "system":
                continue
            if idx == last_user_idx:
                continue
            if not m.content:
                continue
            if m.role == "user":
                history.append(
                    messages.ModelRequest(
                        parts=[messages.UserPromptPart(content=m.content)]
                    )
                )
            elif m.role == "assistant":
                history.append(
                    messages.ModelResponse(parts=[messages.TextPart(content=m.content)])
                )
            elif m.role == "tool":
                history.append(
                    messages.ModelRequest(
                        parts=[
                            messages.ToolReturnPart(
                                tool_name=m.name or "tool",
                                content=m.content,
                                tool_call_id=m.tool_call_id or "",
                            )
                        ]
                    )
                )
        return history or None

    def _model_settings(self, request: InferenceRequest) -> dict[str, Any]:
        """Forward generation params + per-call timeout as pydantic-ai settings."""
        gp = request.generation_params or {}
        settings: dict[str, Any] = {}
        if gp.get("temperature") is not None:
            settings["temperature"] = gp["temperature"]
        if gp.get("max_tokens") is not None:
            settings["max_tokens"] = gp["max_tokens"]
        if gp.get("top_p") is not None:
            settings["top_p"] = gp["top_p"]
        if request.timeout_seconds:
            settings["timeout"] = request.timeout_seconds
        return settings

    def _output_type_for(self, request: InferenceRequest) -> Any:
        """Return a pydantic-ai output type for structured requests, else ``None``."""
        if (
            request.response_format == ResponseFormat.STRUCTURED_OUTPUT
            and request.output_json_schema
        ):
            try:
                pydantic_ai = _require_pydantic_ai()
                return pydantic_ai.StructuredDict(request.output_json_schema)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - fall back to free text
                return None
        return None

    # ------------------------------------------------------------- result maps
    def _map_result(
        self,
        request: InferenceRequest,
        result: Any,
        model_string: str,
        latency_ms: float,
    ) -> InferenceResponse:
        """Map a pydantic-ai run result onto an :class:`InferenceResponse`."""
        output = getattr(result, "output", None)
        output_structured: Any = None
        output_text: str | None
        if isinstance(output, (dict, list)):
            output_structured = output
            output_text = None
        else:
            output_text = None if output is None else str(output)

        tool_calls, tool_returns = self._extract_tool_exchange(result)
        input_tokens, output_tokens = self._extract_usage(result)
        cost = self._compute_cost(request, model_string, input_tokens, output_tokens)

        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=output_text,
            output_structured=output_structured,
            tool_calls=tool_calls,
            tool_returns=tool_returns,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            model_path=model_string,
            provider_name=self.provider_name,
            latency_ms=latency_ms,
            # pydantic-ai runs the whole agent (model → tool → model → answer) inside one
            # call, so this response already holds the final answer: the outer loop must
            # not do a continuation turn (which would re-send an OpenAI-invalid history).
            metadata={"round_trip_complete": True},
        )

    @staticmethod
    def _extract_tool_exchange(
        result: Any,
    ) -> tuple[list[ToolCallRecord], list[ToolReturnRecord]]:
        """Pull tool_calls/tool_returns from ``result.all_messages()`` parts."""
        calls: list[ToolCallRecord] = []
        returns: list[ToolReturnRecord] = []
        all_messages = getattr(result, "all_messages", None)
        if all_messages is None:
            return calls, returns
        try:
            messages = all_messages()
        except Exception:  # noqa: BLE001
            return calls, returns
        for msg in messages:
            for part in getattr(msg, "parts", []) or []:
                kind = type(part).__name__
                if kind == "ToolCallPart":
                    args = getattr(part, "args", {})
                    if isinstance(args, str):
                        import json

                        try:
                            args = json.loads(args)
                        except Exception:  # noqa: BLE001
                            args = {"raw": args}
                    calls.append(
                        ToolCallRecord(
                            tool_call_id=getattr(part, "tool_call_id", "") or "",
                            tool_name=getattr(part, "tool_name", "") or "",
                            args=args if isinstance(args, dict) else {},
                        )
                    )
                elif kind == "ToolReturnPart":
                    returns.append(
                        ToolReturnRecord(
                            tool_call_id=getattr(part, "tool_call_id", "") or "",
                            tool_name=getattr(part, "tool_name", "") or "",
                            content=getattr(part, "content", None),
                        )
                    )
        return calls, returns

    @staticmethod
    def _extract_usage(result: Any) -> tuple[int, int]:
        """Read token usage from ``result.usage()`` (request/response tokens)."""
        usage_fn = getattr(result, "usage", None)
        if usage_fn is None:
            return 0, 0
        try:
            usage = usage_fn()
        except Exception:  # noqa: BLE001
            return 0, 0
        # pydantic-ai exposes request_tokens/response_tokens (newer: input/output).
        input_tokens = (
            getattr(usage, "request_tokens", None)
            or getattr(usage, "input_tokens", None)
            or 0
        )
        output_tokens = (
            getattr(usage, "response_tokens", None)
            or getattr(usage, "output_tokens", None)
            or 0
        )
        return int(input_tokens or 0), int(output_tokens or 0)

    def _compute_cost(
        self,
        request: InferenceRequest,
        model_string: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Compute USD cost from the configured price table (0.0 when unpriced)."""
        if self._runtime_config is None:
            return 0.0
        price = self._runtime_config.price_for(
            model_key=request.model_key, model_name=model_string.split(":", 1)[-1]
        )
        return price.cost(input_tokens=input_tokens, output_tokens=output_tokens)

    def _failed(
        self,
        request: InferenceRequest,
        model_string: str,
        started: float,
        exc: Exception,
    ) -> InferenceResponse:
        """Build a normalized FAILED response for a provider exception."""
        latency_ms = (time.perf_counter() - started) * 1000.0
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            model_path=model_string,
            provider_name=self.provider_name,
            latency_ms=latency_ms,
            error=InferenceError(
                code=InferenceErrorCode.UNKNOWN,
                message=str(exc),
                retryable=False,
            ),
        )
