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
import json
import os
import time
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any, cast

from himmy.core.ids import new_uuid
from himmy.services.inference.models import (
    BoundTool,
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    ResponseFormat,
    ToolCallRecord,
    ToolExecutor,
    ToolReturnRecord,
)
from himmy.services.inference.structured import (
    constrained_decoding_requested,
    ollama_format,
    schema_to_gbnf,
)
from himmy.services.inference.tool_formats import format_for
from himmy.services.inference.tool_protocol import (
    _iter_json_objects,
    parse_text_tool_calls,
)
from himmy.services.tools.access import classify_parallel_safe, describe_for_model
from himmy.services.tools.repair import resolve_tool_name, unknown_tool_message
from himmy.services.tools.schema_normalize import normalize_tool_schema


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


def _ollama_tools(
    bound_tools: list[BoundTool], *, provider: str = "ollama"
) -> list[dict[str, Any]]:
    """Project bound tools into Ollama's ``/api/chat`` function-tool schema.

    Descriptions carry a read/write intent hint so a small model doesn't pick a
    write tool for a look-up (reader/writer disambiguation). Each tool's
    ``args_json_schema`` is run through the shared cross-provider normalizer
    (``$ref`` inlining, nullable-union collapse, …) so a schema that another
    provider tolerates still parses for the local backend.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": describe_for_model(
                    tool.name, tool.description, tool.read_only
                ),
                "parameters": normalize_tool_schema(
                    tool.args_json_schema or {"type": "object"}, provider
                ),
            },
        }
        for tool in bound_tools
    ]


def _parse_ollama_tool_calls(data: dict[str, Any]) -> list[ToolCallRecord]:
    """Parse Ollama's ``message.tool_calls`` into typed ToolCallRecords."""
    raw = ((data.get("message") or {}).get("tool_calls")) or []
    calls: list[ToolCallRecord] = []
    for item in raw:
        fn = item.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        calls.append(
            ToolCallRecord(
                tool_call_id=new_uuid(),
                tool_name=str(fn.get("name", "")),
                args=args if isinstance(args, dict) else {},
            )
        )
    return calls


def _react_tool_manifest(
    bound_tools: list[BoundTool], *, provider: str = "claude-cli"
) -> str:
    """A text manifest + protocol instruction for a text-only model (Claude CLI).

    Parsing tolerates several reply shapes (see
    :func:`~himmy.services.inference.tool_protocol.parse_text_tool_calls`), but a single
    documented format keeps the model on the most reliable path. The advertised
    arg schema is normalized for the provider so a text-only model sees the same
    ``$ref``-inlined, nullable-collapsed shape every other backend gets.
    """
    lines = ["You can call tools. Available tools:"]
    for tool in bound_tools:
        schema = json.dumps(
            normalize_tool_schema(
                tool.args_json_schema or {"type": "object"}, provider
            )
        )
        desc = describe_for_model(tool.name, tool.description, tool.read_only)
        lines.append(f"- {tool.name}: {desc} | args schema: {schema}")
    example = bound_tools[0].name if bound_tools else "tool_name"
    lines.append(
        "To call a tool, write a line of the form:\n"
        "TOOL_CALL <tool_name> <json-args>\n"
        f'e.g.  TOOL_CALL {example} {{"some_arg": "value"}}\n'
        "Use the tool names EXACTLY as listed. If the request needs several "
        "independent tool calls, emit them together, one TOOL_CALL line per call, "
        "in the same reply. If none of the listed tools can address the request, do "
        "not invent a call: just reply normally in prose. When you are done using "
        "tools, reply with your final answer in prose."
    )
    return "\n".join(lines)


# Back-compat shim: the tolerant parser supersedes the old TOOL_CALL-only regex.
def _parse_react_tool_calls(
    text: str, known_names: set[str] | None = None
) -> list[ToolCallRecord]:
    """Parse tool calls from a text reply (delegates to the tolerant parser)."""
    return parse_text_tool_calls(text, known_names)


def _default_tool_parallelism() -> int:
    """Max concurrent read-only tool calls per fan-out: ``HIMMY_TOOL_PARALLELISM`` or 8.

    A misconfigured env var (non-int, ``<= 0``) falls back to the default rather than
    failing — a bad knob must never take the offline tool path down. ``1`` restores
    fully serial execution (the pre-optimization behaviour).
    """
    raw = os.environ.get("HIMMY_TOOL_PARALLELISM", "")
    try:
        value = int(raw)
    except ValueError:
        return 8
    return value if value > 0 else 8


def _is_parallel_safe(tool: BoundTool) -> bool:
    """Whether a resolved tool may join a CONCURRENT read-batch (fail-closed).

    An explicit author ``sequential=True`` ALWAYS forces the tool out of any parallel run
    (a non-reentrant client that must not overlap itself), regardless of read-only intent.
    Otherwise the author's explicit ``read_only`` flag wins: only ``read_only=True`` is
    eligible; ``False`` (a writer) is a barrier.

    When ``read_only`` is unset (``None``) we DO NOT trust the first-token-wins
    :func:`classify_read_only` hint to gate concurrency — it infers a name like
    ``fetch_and_delete``/``report_incident`` as read-only despite a write verb, which would
    let a mutating call be hoisted into a concurrent read-batch ahead of a dependent read.
    Instead we use the strict :func:`classify_parallel_safe`, which is read-only ONLY when
    the name has a read verb and NO write verb anywhere. Ambiguous or mixed names stay
    sequential — parallelism is opt-in for unambiguous readers, never assumed.
    """
    if tool.sequential:
        return False
    if tool.read_only is not None:
        return tool.read_only
    return classify_parallel_safe(tool.name)


async def _run_one_tool_call(
    call: ToolCallRecord,
    by_name: dict[str, BoundTool],
    available: list[str],
    executor: ToolExecutor | None,
) -> ToolReturnRecord:
    """Execute a SINGLE tool call (name-repair + authz/approval via the executor).

    Extracted so the fan-out can dispatch calls either serially or concurrently while
    the per-call semantics (typo-repair, capability authz + approval gate inside the
    executor, unknown-tool correction) stay byte-identical to the serial path.
    """
    tool = by_name.get(call.tool_name)
    if tool is None:
        # Repair a near-miss name: auto-apply a confident typo-fix, otherwise hand
        # the model a concrete correction (did-you-mean + the available tools).
        resolution = resolve_tool_name(call.tool_name, available)
        if resolution.name is not None:
            tool = by_name[resolution.name]
    if tool is not None and executor is not None:
        ret = await executor(tool.name, call.args)
        ret = ret.model_copy(update={"tool_call_id": call.tool_call_id})
        if tool.name != call.tool_name:  # record the auto-correction
            ret = ret.model_copy(
                update={
                    "tool_name": tool.name,
                    "metadata": {
                        **(ret.metadata or {}),
                        "repaired_from": call.tool_name,
                    },
                }
            )
        return ret
    resolution = resolve_tool_name(call.tool_name, available)
    message = unknown_tool_message(resolution, available)
    return ToolReturnRecord(
        tool_call_id=call.tool_call_id,
        tool_name=call.tool_name,
        content=message,
        outcome="failed",
        metadata={
            "error_code": "UNKNOWN_TOOL",
            "error_message": message,
            "suggestions": resolution.suggestions,
        },
    )


async def _execute_tool_calls(
    bound_tools: list[BoundTool],
    tool_calls: list[ToolCallRecord],
    executor: ToolExecutor | None,
) -> list[ToolReturnRecord]:
    """Run each tool call via the executor and return the paired ToolReturnRecords.

    A provider that emits tool calls (Ollama/Claude CLI) must also EXECUTE them and
    populate ``tool_returns`` — the runtime only replays existing call/return pairs
    onto the thread, so without this the model never sees a result and loops. Mirrors
    :class:`StubClientManager`'s execution path; ``bound_tools`` is still used to
    validate/repair the call's tool name before invoking the executor.

    Only MAXIMAL RUNS OF CONSECUTIVE read-only calls are executed CONCURRENTLY under a
    bounded semaphore (``HIMMY_TOOL_PARALLELISM``, default 8). Write/side-effecting/
    ambiguous/unknown calls act as strict ORDERING BARRIERS: iterating in original order,
    a pending read-batch is fully awaited before the barrier call runs, and the barrier
    runs sequentially before iteration continues. This preserves the serial
    happens-before across every write boundary — a read positioned AFTER a write still
    observes that write's effect (e.g. ``append_note`` then ``list_notes``) — while still
    parallelising the common all-reads (or trailing-reads-after-a-write) case. Reads have
    no ordering, idempotency, or approval hazard, so within one uninterrupted read-batch
    concurrency cuts wall-clock without changing any result.

    Results are always returned in the calls' ORIGINAL order. The executor
    (:meth:`ToolService.tool_executor`) converts ordinary tool errors into a ``failed``
    RETURN record rather than raising, so on the normal path the batch is fully
    order-preserving and no call runs twice. A concurrent read that does raise an
    UNEXPECTED exception (cancellation, OOM, an executor-wrapper bug) is isolated —
    siblings already scheduled in its batch run to completion and the earliest-in-order
    failure is re-raised. NOTE: on that raise path a read-batch is all-or-nothing, not
    first-failure-stop: a sibling ordered AFTER the failing read within the SAME batch
    will already have executed, whereas a strictly serial loop would have aborted before
    reaching it. Reads carry no ordering/idempotency/approval hazard, so the only
    observable difference is a benign extra read side effect (e.g. a lazy cache
    write-through); no write is ever in a batch. Setting the knob to ``1`` restores fully
    serial (first-failure-stop) behaviour.
    """
    by_name = {t.name: t for t in bound_tools}
    available = list(by_name)
    parallelism = _default_tool_parallelism()

    returns: list[ToolReturnRecord | None] = [None] * len(tool_calls)

    def _is_parallel(call: ToolCallRecord) -> bool:
        """Read-only + executable => eligible to join the current concurrent batch."""
        if parallelism <= 1 or executor is None:
            return False
        tool = by_name.get(call.tool_name)
        if tool is None:
            resolution = resolve_tool_name(call.tool_name, available)
            if resolution.name is not None:
                tool = by_name[resolution.name]
        return tool is not None and _is_parallel_safe(tool)

    async def _flush(batch: list[int]) -> None:
        """Run a batch of CONSECUTIVE read indices concurrently, preserving order.

        Exceptions are isolated per call (siblings still complete); the earliest-in-order
        failure within the batch is re-raised so the caller sees exactly what a serial
        loop would have surfaced. Because a batch only ever contains reads that are
        adjacent in the original ``tool_calls`` order, no read is ever hoisted across a
        write barrier.
        """
        if not batch:
            return
        if len(batch) == 1:
            # Single read: no benefit to spinning up a gather/semaphore.
            returns[batch[0]] = await _run_one_tool_call(
                tool_calls[batch[0]], by_name, available, executor
            )
            return
        semaphore = asyncio.Semaphore(parallelism)

        async def _guarded(idx: int) -> ToolReturnRecord:
            async with semaphore:
                return await _run_one_tool_call(
                    tool_calls[idx], by_name, available, executor
                )

        results = await asyncio.gather(
            *(_guarded(i) for i in batch), return_exceptions=True
        )
        for idx, result in zip(batch, results, strict=True):
            if isinstance(result, BaseException):
                # Re-raise the FIRST failure in original order; siblings already ran.
                raise result
            returns[idx] = result

    # Walk calls in order, accumulating a batch of consecutive reads. Any non-parallel
    # call is a barrier: flush the pending batch (await completion), run it sequentially,
    # then continue — exactly the serial happens-before, minus the read-batch latency.
    batch: list[int] = []
    for i, call in enumerate(tool_calls):
        if _is_parallel(call):
            batch.append(i)
            continue
        await _flush(batch)
        batch = []
        returns[i] = await _run_one_tool_call(call, by_name, available, executor)
    await _flush(batch)

    return [r for r in returns if r is not None]


def _default_ollama_timeout() -> float:
    """Per-request timeout: ``HIMMY_OLLAMA_TIMEOUT`` env when set, else 120s.

    Invalid values fall back to the default rather than failing construction —
    a misconfigured env var must not take the offline path down.
    """
    raw = os.environ.get("HIMMY_OLLAMA_TIMEOUT", "")
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    return value if value > 0 else 120.0


class OllamaClientManager:
    """A :class:`ClientManager` backed by a local Ollama server's chat API."""

    def __init__(
        self,
        *,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        model_registry: dict[str, str] | None = None,
        transport: Callable[[str, dict[str, Any]], Any] | None = None,
        timeout: float | None = None,
        provider_name: str = "ollama",
        tool_call_format: str | None = None,
    ) -> None:
        """Configure the default model, server URL, and (test) transport.

        ``timeout`` defaults to the ``HIMMY_OLLAMA_TIMEOUT`` env var when unset
        (else 120s) — an operational knob for slow hosts (CPU-only CI runners,
        small boards) where generation legitimately exceeds the default budget.

        ``tool_call_format`` pins the tool-call grammar for this manager; ``None``
        auto-selects by resolved model tag and falls back to the GENERIC format
        (today's behavior). See :mod:`himmy.services.inference.tool_formats`.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._registry = dict(model_registry or {})
        self._transport = transport
        self._timeout = timeout if timeout is not None else _default_ollama_timeout()
        self._tool_call_format = tool_call_format
        #: Slow-local-provider floor (INF): the framework's 30s default request
        #: timeout kills legitimate CPU generations; the service's proportional
        #: ceiling honors this declared minimum (see InferenceService.run).
        self.min_timeout_seconds = self._timeout
        self.provider_name = provider_name

    def resolve(self, model_key: str) -> str:
        """Map a model key to an Ollama model tag (registry, else the key/default)."""
        if not model_key or model_key == "default":
            return self._registry.get("default", self._model)
        return self._registry.get(model_key, model_key)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Call Ollama ``/api/chat`` and map the reply onto an InferenceResponse."""
        model = self.resolve(request.model_key)
        tool_format = format_for(model, self._tool_call_format)
        started = time.perf_counter()
        params = request.generation_params or {}
        options: dict[str, Any] = {}
        if params.get("temperature") is not None:
            options["temperature"] = params["temperature"]
        if params.get("max_tokens") is not None:
            options["num_predict"] = params["max_tokens"]
        if request.seed is not None:
            options["seed"] = request.seed
        messages = _chat_messages(request)
        # Text-path formats (e.g. Hermes/Qwen ChatML-XML) drive tools through the
        # prompt: suppress Ollama's native ``tools=`` field and inject the rendered
        # manifest as a leading system message, so the FORMAT is the only variable.
        use_text_path = bool(request.bound_tools) and tool_format.flags.use_text_tool_path
        if use_text_path:
            manifest = tool_format.render_system_manifest(
                request.bound_tools, self.provider_name
            )
            messages = [{"role": "system", "content": manifest}, *messages]
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        if request.bound_tools and not use_text_path:
            payload["tools"] = _ollama_tools(
                request.bound_tools, provider=self.provider_name
            )
        structured_requested = (
            request.response_format == ResponseFormat.STRUCTURED_OUTPUT
            and request.output_json_schema is not None
        )
        if structured_requested:
            # Ollama's native structured-output: constrain the reply to the schema.
            # Opt-in (constrained_decoding): run the schema through the shared
            # cross-provider normalizer first ($ref inlining, nullable collapse) so a
            # referenced/exotic schema still constrains rather than being passed
            # through un-inlined. Default (un-opted-in) path is byte-identical to
            # before: the raw schema, exactly as today.
            if constrained_decoding_requested(request):
                payload["format"] = ollama_format(
                    request.output_json_schema, provider=self.provider_name
                )
            else:
                payload["format"] = request.output_json_schema
        # Floor the per-call budget like the CLI manager does: a CPU host can
        # legitimately need minutes; request defaults must not kill valid calls.
        effective_timeout = max(request.timeout_seconds or 0.0, self._timeout)
        try:
            data = await self._post("/api/chat", payload, effective_timeout)
        except Exception as exc:  # noqa: BLE001 - normalize to FAILED
            return _failed(
                request,
                message=f"ollama request failed: {exc}",
                provider=self.provider_name,
                model_path=f"ollama:{model}",
                started=started,
            )
        text = ((data.get("message") or {}).get("content")) or ""
        tool_calls = _parse_ollama_tool_calls(data)
        # Text fallback: a small model that wrote its tool call in prose instead of
        # using Ollama's native tool field would otherwise be missed entirely.
        if not tool_calls and request.bound_tools and text:
            known = {t.name for t in request.bound_tools}
            tool_calls = tool_format.parse(text, known)
            if tool_calls:
                text = ""  # the reply was a tool call, not a final answer
        tool_returns = await _execute_tool_calls(
            request.bound_tools, tool_calls, request.tool_executor
        )
        output_structured = None
        if structured_requested and text:
            try:
                output_structured = json.loads(text)
            except json.JSONDecodeError:
                output_structured = None
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=text,
            output_structured=output_structured,
            tool_calls=tool_calls,
            tool_returns=tool_returns,
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
            return cast(dict[str, Any], result)
        import httpx

        async with httpx.AsyncClient(timeout=timeout or self._timeout) as client:
            response = await client.post(self._base_url + path, json=payload)
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]


def _parse_cli_output(out: str) -> tuple[str, int, int, float]:
    """Parse ``claude`` CLI stdout → (text, input_tokens, output_tokens, cost_usd).

    With ``--output-format json`` the CLI emits a result object carrying ``result``,
    ``usage``, and ``total_cost_usd``. A plain-text reply (test runner / older CLI)
    falls back to (text, 0, 0, 0.0) so behaviour is unchanged when usage is absent.
    """
    text = out.strip()
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text, 0, 0, 0.0
    if not isinstance(data, dict) or "result" not in data:
        return text, 0, 0, 0.0
    result = str(data.get("result") or "")
    usage = data.get("usage") or {}
    in_tok = int(usage.get("input_tokens", 0) or 0)
    # Cache reads/creates are billed input the CLI breaks out separately; fold
    # them in so the token count matches what was actually charged.
    in_tok += int(usage.get("cache_read_input_tokens", 0) or 0)
    in_tok += int(usage.get("cache_creation_input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    cost = float(data.get("total_cost_usd", 0.0) or 0.0)
    return result.strip(), in_tok, out_tok, cost


# The ``claude`` CLI is a full agent: by default it will use its OWN built-in tools
# (web search/fetch, bash, file edits, …) to satisfy a prompt. Himmy drives tools
# itself via the ReAct protocol, so we disable the CLI's tools — otherwise it goes off
# doing its own multi-turn research and hangs instead of just emitting our TOOL_CALL.
_CLI_BUILTIN_TOOLS = (
    "WebSearch",
    "WebFetch",
    "Bash",
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Task",
    "NotebookEdit",
    "TodoWrite",
)

# The local CLI is slow (cold starts, queueing): even a trivial prompt can take ~20s.
# Never let a low per-request timeout kill a legitimately slow-but-valid call — floor it.
_CLI_MIN_TIMEOUT = 150.0


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
        tool_call_format: str | None = None,
    ) -> None:
        """Configure the CLI executable, default model, and (test) runner.

        ``tool_call_format`` pins the tool-call grammar for this manager; ``None``
        auto-selects by resolved model tag and falls back to the GENERIC format
        (today's behavior). See :mod:`himmy.services.inference.tool_formats`.
        """
        self._executable = executable
        self._model = model
        self._registry = dict(model_registry or {})
        self._extra_args = list(extra_args or [])
        self._runner = runner
        self._timeout = timeout
        self._tool_call_format = tool_call_format
        #: Slow-local-provider floor: mirrors the effective_timeout floor below.
        self.min_timeout_seconds = _CLI_MIN_TIMEOUT
        self.provider_name = provider_name

    def resolve(self, model_key: str) -> str:
        """Map a model key to a CLI ``--model`` alias (registry, else key/default)."""
        if not model_key or model_key == "default":
            return self._registry.get("default", self._model)
        return self._registry.get(model_key, model_key)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Run ``claude -p --model <m>`` with the prompt on stdin; capture stdout.

        Requests ``--output-format json`` so the CLI reports real token usage +
        ``total_cost_usd`` (powering the cost/token HUD). A plain-text reply (e.g. a
        test runner, or an older CLI) is still accepted via :func:`_parse_cli_output`.
        """
        model = self.resolve(request.model_key)
        tool_format = format_for(model, self._tool_call_format)
        started = time.perf_counter()
        argv = [self._executable, "-p", "--model", model]
        if not any("--output-format" in a for a in self._extra_args):
            argv += ["--output-format", "json"]
        # Drive the CLI as a pure text model: disable its own agentic tools unless the
        # caller already configured tool flags. Keeps it from doing its own research
        # (and hanging) instead of emitting the ReAct TOOL_CALL we ask for.
        if not any(
            "--tools" in a or "isallowedTools" in a or "allowedTools" in a
            for a in self._extra_args
        ):
            argv += ["--disallowedTools", *_CLI_BUILTIN_TOOLS]
        argv += self._extra_args
        prompt = _compose_prompt(request)
        if request.bound_tools:
            # Text-only CLI: a best-effort ReAct protocol the runtime loop drives.
            prompt = (
                f"{prompt}\n\n"
                f"{tool_format.render_system_manifest(request.bound_tools, self.provider_name)}"
            )
        structured_requested = (
            request.response_format == ResponseFormat.STRUCTURED_OUTPUT
            and request.output_json_schema is not None
        )
        if structured_requested:
            # Text-only CLI: no provider-native schema constraint (unlike Ollama's
            # `format` field), so nudge via the prompt and parse the reply below.
            prompt = (
                f"{prompt}\n\nReply with ONLY a single JSON object that matches this "
                "JSON Schema — no prose, no code fences:\n"
                f"{json.dumps(request.output_json_schema)}"
            )
        # Floor the per-call timeout: the local CLI is slow, and a 30s framework
        # default would kill valid ~20-30s responses (then retry, wasting minutes).
        effective_timeout = max(request.timeout_seconds or 0.0, _CLI_MIN_TIMEOUT)
        try:
            out = await self._run(argv, prompt, effective_timeout)
        except Exception as exc:  # noqa: BLE001 - normalize to FAILED
            return _failed(
                request,
                message=f"claude CLI failed: {exc}",
                provider=self.provider_name,
                model_path=f"claude-cli:{model}",
                started=started,
            )
        text, in_tok, out_tok, cli_cost = _parse_cli_output(out)
        tool_calls = (
            tool_format.parse(text, {t.name for t in request.bound_tools})
            if request.bound_tools
            else []
        )
        tool_returns = await _execute_tool_calls(
            request.bound_tools, tool_calls, request.tool_executor
        )
        output_structured = None
        if structured_requested and text and not tool_calls:
            # The CLI replies in prose-capable text: take the first JSON object
            # embedded in it (tolerates fences/preamble a nudge can't fully prevent).
            output_structured = next(
                (o for o in _iter_json_objects(text) if isinstance(o, dict)), None
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="" if tool_calls else text,
            output_structured=output_structured,
            tool_calls=tool_calls,
            tool_returns=tool_returns,
            model_path=f"claude-cli:{model}",
            provider_name=self.provider_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost=cli_cost,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _run(self, argv: list[str], stdin: str, timeout: float) -> str:
        if self._runner is not None:
            result = self._runner(argv, stdin)
            if isinstance(result, Awaitable):
                return await result  # type: ignore[no-any-return]
            return str(result)
        # Strip the parent Claude Code session's markers so a nested `claude -p`
        # spawned from inside one starts a clean session (avoids the nesting error).
        child_env = {
            k: v
            for k, v in os.environ.items()
            if k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE")
        }
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=child_env,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(stdin.encode("utf-8")),
                timeout=timeout or self._timeout,
            )
        finally:
            # On timeout/cancel, ``communicate`` is cancelled but the detached
            # ``claude`` subprocess keeps running. Reap it so we never leak a
            # long-lived process (mirrors the MCP client's terminate->kill ladder).
            if proc.returncode is None:
                with suppress(ProcessLookupError, Exception):
                    proc.terminate()
                with suppress(Exception):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                with suppress(ProcessLookupError, Exception):
                    if proc.returncode is None:
                        proc.kill()
                with suppress(Exception):
                    await proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(
                err.decode("utf-8", "replace").strip() or "claude CLI nonzero exit"
            )
        return out.decode("utf-8", "replace")


class HimalayaGptClientManager:
    """A :class:`ClientManager` for a self-hosted HF Transformers model (HimalayaGPT)."""

    #: When no explicit ``tool_call_format`` is pinned, the HimalayaGPT-0.5b base model
    #: has NO native tool field and does not emit ``<tool_call>`` zero-shot, so it
    #: defaults to the few-shot Hermes grammar (manifest carries 2 synthesized exemplars
    #: + the "output only <tool_call>, no code" nudge — the recipe that makes the 0.5b
    #: tool-capable). An explicit ``tool_call_format`` (incl. "generic") overrides this.
    _DEFAULT_TOOL_FORMAT = "hermes_chatml_xml_fewshot"

    def __init__(
        self,
        *,
        model_name: str = "himalaya-ai/himalayagpt-0.5b-it",
        generate_fn: Callable[[str], str] | None = None,
        max_new_tokens: int = 256,
        provider_name: str = "himalayagpt",
        tool_call_format: str | None = None,
    ) -> None:
        """Configure the model id and (test) generate function.

        ``tool_call_format`` pins the tool-call grammar for this manager. ``None``
        resolves to the few-shot Hermes format (:data:`_DEFAULT_TOOL_FORMAT`) — the
        base model has no native tool field, so it needs the prompted grammar + few-shot
        to call tools at all. See :mod:`himmy.services.inference.tool_formats`.
        """
        self._model_name = model_name
        self._generate_fn = generate_fn
        self._max_new_tokens = max_new_tokens
        self.provider_name = provider_name
        self._tool_call_format = tool_call_format
        self._pipe: Any = None

    def resolve(self, model_key: str) -> str:
        """Resolve to the configured model id (registry-less single model)."""
        if not model_key or model_key == "default":
            return self._model_name
        return model_key

    def _resolve_format(self, model: str) -> Any:
        """Resolve the tool-call format, defaulting this model to the few-shot Hermes
        grammar. An explicit ``tool_call_format`` (any registered name) wins."""
        override = self._tool_call_format or self._DEFAULT_TOOL_FORMAT
        return format_for(model, override)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text on a worker thread (Transformers inference is blocking).

        With bound tools, the manifest is rendered via the resolved format and injected
        as a leading system turn (the model has no native tool field), the reply is
        parsed for ``<tool_call>`` blocks via the format, and any calls are executed and
        their results fed back in the format's result grammar — so the model is driven
        tool-capable end-to-end THROUGH this manager + the format registry.
        """
        model = self.resolve(request.model_key)
        tool_format = self._resolve_format(model)
        started = time.perf_counter()
        prompt = self._compose_tool_prompt(request, tool_format)
        # Guaranteed structured output (opt-in): compile the schema into a GBNF
        # grammar and pass it to the llama.cpp backend so decoding is CONSTRAINED to
        # valid JSON. Fail-open: only when the request opted in AND asked for a
        # structured reply AND the backend's generate-fn accepts a ``grammar`` kwarg;
        # otherwise the prompt is unchanged (today's behaviour).
        grammar = self._maybe_grammar(request)
        try:
            text = await asyncio.to_thread(self._infer, prompt, grammar)
        except Exception as exc:  # noqa: BLE001 - normalize to FAILED
            return _failed(
                request,
                message=f"HimalayaGPT generation failed: {exc}",
                provider=self.provider_name,
                model_path=self._model_name,
                started=started,
            )
        tool_calls: list[ToolCallRecord] = []
        if request.bound_tools and text:
            known = {t.name for t in request.bound_tools}
            try:
                tool_calls = tool_format.parse(text, known)
            except Exception:  # noqa: BLE001 - parse is fail-open, never fatal
                tool_calls = []
            if tool_calls:
                text = ""  # the reply was a tool call, not a final answer
        tool_returns = await _execute_tool_calls(
            request.bound_tools, tool_calls, request.tool_executor
        )
        # When decoding was grammar-constrained to the schema, parse the (now
        # guaranteed-valid) JSON reply into ``output_structured`` like Ollama does.
        output_structured = None
        if grammar is not None and text and not tool_calls:
            try:
                output_structured = json.loads(text)
            except json.JSONDecodeError:
                output_structured = None
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=text,
            output_structured=output_structured,
            tool_calls=tool_calls,
            tool_returns=tool_returns,
            model_path=self._model_name,
            provider_name=self.provider_name,
            cost=0.0,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _maybe_grammar(self, request: InferenceRequest) -> str | None:
        """Compile a GBNF grammar for an opt-in structured-output request, else ``None``.

        Returns ``None`` (the unconstrained path, byte-identical to before) unless:
        the request opted into constrained decoding
        (:func:`constrained_decoding_requested`), it asked for a STRUCTURED_OUTPUT
        reply with a schema, AND the configured ``generate_fn`` actually accepts a
        ``grammar`` keyword (a backend that can apply it — e.g. the llama.cpp fast
        bridge). A tool-call turn is left unconstrained so the model can still emit a
        ``<tool_call>`` block. Fail-open: any error returns ``None``.
        """
        if request.bound_tools:
            return None
        if not constrained_decoding_requested(request):
            return None
        if (
            request.response_format != ResponseFormat.STRUCTURED_OUTPUT
            or request.output_json_schema is None
        ):
            return None
        if not self._generate_fn_accepts_grammar():
            return None
        try:
            return schema_to_gbnf(request.output_json_schema, provider=self.provider_name)
        except Exception:  # noqa: BLE001 - fail-open: never break a call on a grammar bug
            return None

    def _generate_fn_accepts_grammar(self) -> bool:
        """True when the configured ``generate_fn`` accepts a ``grammar`` keyword.

        The real Transformers pipeline path (``generate_fn is None``) has no grammar
        backend, so it returns ``False`` (no constraint). An injected bridge whose
        callable takes ``grammar`` (the llama.cpp fast bridge) returns ``True``.
        """
        fn = self._generate_fn
        if fn is None:
            return False
        try:
            import inspect

            sig = inspect.signature(fn)
        except (TypeError, ValueError):  # pragma: no cover - exotic callables
            return False
        params = sig.parameters
        if "grammar" in params:
            return True
        # A ``**kwargs``-accepting callable can also take ``grammar``.
        return any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )

    def _compose_tool_prompt(self, request: InferenceRequest, tool_format: Any) -> str:
        """Flatten the conversation into the single labeled prompt the model receives.

        No bound tools (or a non-text-path format): unchanged ``_compose_prompt`` so the
        plain-chat path is a strict no-op. With bound tools on a text-path format, the
        rendered tool manifest is prepended as a leading ``[System]`` block and any
        already-present tool returns are re-labeled via the format's result grammar, so
        the model sees the tool advertisement, the few-shot exemplars, and the prior
        results in the grammar it was taught.

        When the format exposes ``fewshot_turns`` (the himalaya few-shot format), the
        exemplars are emitted as REAL prior ``[User]``/``[Assistant]`` turns (between the
        manifest and the real conversation) and the FLATTENED in-manifest ``# Examples``
        block is suppressed (the ``render_system_manifest_base`` renderer) so the
        exemplars are not duplicated — the vendor multi-turn few-shot recipe. This is
        OPT-IN: a format without ``fewshot_turns`` (canonical Hermes/Qwen, GENERIC, every
        other manager) renders single-turn byte-identically to before.
        """
        use_text_path = bool(request.bound_tools) and getattr(
            tool_format.flags, "use_text_tool_path", False
        )
        if not use_text_path:
            return _compose_prompt(request)
        # Multi-turn few-shot (opt-in): exemplars become real prior turns and the system
        # block uses the base (no flattened # Examples) manifest. Fail-open: any error or
        # an absent renderer falls back to the single-turn flattened manifest.
        fewshot_turns: list[tuple[str, str]] = []
        fewshot_renderer = getattr(tool_format, "fewshot_turns", None)
        if fewshot_renderer is not None:
            try:
                fewshot_turns = list(fewshot_renderer(request.bound_tools))
            except Exception:  # noqa: BLE001 - few-shot turns are best-effort, never fatal
                fewshot_turns = []
        if fewshot_turns:
            base_renderer = (
                getattr(tool_format, "render_system_manifest_base", None)
                or tool_format.render_system_manifest
            )
            manifest = base_renderer(request.bound_tools, self.provider_name)
        else:
            manifest = tool_format.render_system_manifest(
                request.bound_tools, self.provider_name
            )
        parts = [f"[System]\n{manifest}"]
        for role, content in fewshot_turns:
            label = {"user": "[User]", "assistant": "[Assistant]"}.get(
                str(role).lower(), "[User]"
            )
            parts.append(f"{label}\n{content}")
        results: list[ToolReturnRecord] = []

        def _flush_results() -> None:
            if results:
                body = tool_format.render_tool_results(results)
                if body:
                    parts.append(f"[Tool result]\n{body}")
                results.clear()

        for message in request.messages:
            role = str(message.role).lower()
            if role == "tool":
                # Buffer consecutive tool returns so the format can batch them, then
                # render via the format's result grammar (<tool_response> for Hermes).
                results.append(
                    ToolReturnRecord(
                        tool_call_id="",
                        tool_name="",
                        content=message.content,
                    )
                )
                continue
            _flush_results()
            if not message.content:
                continue
            label = {"user": "[User]", "assistant": "[Assistant]"}.get(
                role, "[System]"
            )
            parts.append(f"{label}\n{message.content}")
        _flush_results()
        return "\n\n".join(parts)

    def _infer(self, prompt: str, grammar: str | None = None) -> str:
        if self._generate_fn is not None:
            # Pass the GBNF grammar only when one was compiled (opt-in structured
            # output) — ``_maybe_grammar`` already verified the callable accepts it,
            # so a plain ``(prompt) -> str`` fn is still called with a single arg.
            if grammar is not None:
                return self._generate_fn(prompt, grammar=grammar)  # type: ignore[call-arg]
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
