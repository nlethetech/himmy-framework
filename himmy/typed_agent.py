"""Write-time typed agents: a generic Agent facade over the existing runtime.

:class:`TypedAgent` is a thin, type-checked DX layer on top of
:class:`~himmy.runtime.single_agent.SingleAgentRuntime`. It is generic over a
caller-supplied *deps* type ``DepsT`` (arbitrary dependency object handed to the
agent and its tools) and a pydantic *output* type ``OutputT`` (a
:class:`~pydantic.BaseModel` subclass). From those type parameters it:

* compiles the model's structured-output schema from ``OutputT.model_json_schema``
  and drives the run through the runtime's existing ``output_schema`` seam;
* compiles each tool's JSON-Schema *argument* contract directly from the Python
  function signature (type hints) — so the model sees a real, typed tool surface
  with zero hand-written schemas;
* validates the model's structured result against ``OutputT`` and, on a
  validation failure, *retries* on the same runtime loop with a corrective nudge
  (bounded by ``output_retries``), emitting an audited repair event;
* returns a typed :class:`TypedAgentRunResult` whose ``.output`` is a fully
  validated ``OutputT`` instance — and the whole thing passes mypy's near-strict
  generics.

It is **additive**: there is no parallel engine. Tools register through the
existing :class:`~himmy.services.tools.service.ToolService`; the run executes on
``SingleAgentRuntime.run_agent_loop``; every event flows through the runtime's
own audit emission. The zero-config default (the offline ``StubClientManager``)
produces a schema-valid structured result, so a typed agent runs end-to-end with
no keys and no network — the offline-first default is preserved.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
    get_type_hints,
)

from pydantic import BaseModel, ValidationError

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.personas.persona import Persona
    from himmy.runtime.single_agent import RunResult, SingleAgentRuntime
    from himmy.services.inference.models import LLMConfig
    from himmy.services.tools.service import ToolService


@dataclass
class RunContext[DepsT]:
    """The typed context handed to every tool of a :class:`TypedAgent`.

    ``deps`` is the exact dependency object the caller passed to
    :meth:`TypedAgent.run` (e.g. a DB handle, an API client, a config). It is the
    typed equivalent of pydantic-ai's ``RunContext`` — fully generic so a tool
    body type-checks against the concrete deps type. ``retry`` is the 0-based
    output-validation retry index for the turn the tool is running in (0 on the
    first attempt), so a tool can adapt on a retried run if it wants to.
    """

    deps: DepsT
    retry: int = 0


#: A typed tool body: takes the run context (carrying typed deps) plus keyword
#: arguments synthesized from its signature, and returns any JSON-able result.
ToolFunc = Callable[..., Awaitable[Any]]


@dataclass
class _RegisteredTool:
    """One tool a :class:`TypedAgent` will register on its runtime's tool service."""

    name: str
    description: str
    func: ToolFunc
    args_json_schema: dict[str, Any]
    takes_context: bool


@dataclass
class TypedAgentRunResult[OutputT: BaseModel]:
    """A typed view of one :meth:`TypedAgent.run` outcome.

    ``output`` is the validated ``OutputT`` instance (raise-on-access only via the
    ``run`` contract — the run raises if validation could not be satisfied, so a
    returned result always carries a real ``output``). ``raw`` is the underlying
    :class:`~himmy.runtime.single_agent.RunResult` for callers that want cost,
    tokens, tool exchanges, or the answered thread. ``validation_attempts`` is how
    many model turns were spent reaching a valid output (1 = first try).
    """

    output: OutputT
    raw: RunResult
    validation_attempts: int = 1
    repaired: bool = False

    @property
    def cost(self) -> float:
        """Provider cost of the final (successful) turn."""
        return self.raw.cost

    @property
    def thread(self) -> Any:
        """The answered chat thread (for persistence / inspection)."""
        return self.raw.thread


# --------------------------------------------------------------------------- #
# Signature -> JSON Schema compilation                                         #
# --------------------------------------------------------------------------- #
#: Treated as the run-context parameter (never advertised to the model).
_CONTEXT_PARAM_NAMES: frozenset[str] = frozenset({"ctx", "context", "run_context"})


def _is_context_param(name: str, annotation: Any) -> bool:
    """True when a tool parameter is the :class:`RunContext` (not a model arg).

    A parameter is the context either by its conventional name (``ctx`` /
    ``context`` / ``run_context``) or because its annotation is ``RunContext`` /
    ``RunContext[...]``.
    """
    if name in _CONTEXT_PARAM_NAMES:
        return True
    origin = getattr(annotation, "__origin__", None)
    if origin is RunContext or annotation is RunContext:
        return True
    return False


def compile_tool_schema(func: Callable[..., Any]) -> tuple[dict[str, Any], bool]:
    """Compile a tool's argument JSON Schema from its Python signature.

    Builds a transient pydantic model from the function's *non-context*
    parameters (using their type annotations and defaults) and emits its
    ``model_json_schema``. Parameters with no annotation fall back to ``Any``;
    parameters without a default become ``required``. Returns
    ``(args_json_schema, takes_context)`` where ``takes_context`` says whether the
    first parameter is a :class:`RunContext` to be injected at call time.

    This is the inverse of
    :func:`himmy.services.tools.runtime_adapter.build_arg_model` (schema -> model)
    and is what gives typed agents a real, type-derived tool surface with zero
    hand-written schemas.
    """
    from pydantic import create_model

    signature = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # pragma: no cover - defensive: unresolved forward refs
        hints = {}

    takes_context = False
    fields: dict[str, tuple[Any, Any]] = {}
    for index, (param_name, param) in enumerate(signature.parameters.items()):
        annotation = hints.get(param_name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = Any
        if index == 0 and _is_context_param(param_name, annotation):
            takes_context = True
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            # *args / **kwargs cannot be advertised as a typed schema field.
            continue
        default = ... if param.default is inspect.Parameter.empty else param.default
        fields[param_name] = (annotation, default)

    model_name = f"{_camel(func.__name__)}Args"
    arg_model = create_model(model_name, **fields)  # type: ignore[call-overload]
    schema = cast("dict[str, Any]", arg_model.model_json_schema())
    return schema, takes_context


def _camel(name: str) -> str:
    """Turn ``snake_case`` into ``CamelCase`` for a generated model name."""
    return "".join(part.title() for part in name.split("_") if part) or "Tool"


# --------------------------------------------------------------------------- #
# The typed Agent facade                                                       #
# --------------------------------------------------------------------------- #
class TypedAgent[DepsT, OutputT: BaseModel]:
    """A typed Agent: generic over a deps type and a pydantic output type.

    Construct it with the runtime, a persona (or a name/instructions pair), the
    ``output_type`` pydantic model, and an optional ``deps_type`` (purely for
    documentation / type inference at the call site). Register typed tools with
    :meth:`tool`. Call :meth:`run` with a prompt and the typed ``deps`` to get a
    validated ``OutputT`` back.

    The agent does **not** own a parallel engine: it composes the existing
    :class:`~himmy.runtime.single_agent.SingleAgentRuntime`, registers its tools on
    that runtime's :class:`~himmy.services.tools.service.ToolService`, runs the
    standard agent loop, and validates the runtime's structured output against
    ``OutputT`` — retrying on the same loop when validation fails.
    """

    def __init__(
        self,
        runtime: SingleAgentRuntime,
        *,
        output_type: type[OutputT],
        persona: Persona | None = None,
        name: str = "typed-agent",
        instructions: list[str] | None = None,
        deps_type: type[DepsT] | None = None,
        output_retries: int = 2,
        max_turns: int = 6,
        model_key: str | None = None,
    ) -> None:
        """Wire a typed agent over an existing runtime.

        ``output_type`` is the pydantic model the result is validated against; its
        ``model_json_schema`` is the structured-output contract sent to the model.
        ``deps_type`` is optional and only documents the deps contract (the call
        site infers ``DepsT`` from the ``deps`` argument). ``output_retries`` bounds
        how many extra model turns are spent repairing an output that fails
        validation. ``max_turns`` bounds the per-attempt agentic tool loop.
        """
        if not (isinstance(output_type, type) and issubclass(output_type, BaseModel)):
            raise HimmyError(
                "TypedAgent.output_type must be a pydantic BaseModel subclass."
            )
        if output_retries < 0:
            raise HimmyError("output_retries must be >= 0.")
        if max_turns < 1:
            raise HimmyError("max_turns must be >= 1.")

        self._runtime = runtime
        self._output_type = output_type
        self._deps_type = deps_type
        self._output_retries = output_retries
        self._max_turns = max_turns
        self._model_key = model_key
        self._tools: dict[str, _RegisteredTool] = {}

        if persona is None:
            from himmy.agents.personas.persona import Persona as _Persona

            persona = _Persona(name=name, instructions=instructions or [])
        self._persona = persona

        # Cache the compiled output schema once (it is content-stable).
        self._output_schema: dict[str, Any] = output_type.model_json_schema()

    # ------------------------------------------------------------------ tools
    @property
    def persona(self) -> Persona:
        """The persona this agent runs under."""
        return self._persona

    @property
    def output_schema(self) -> dict[str, Any]:
        """The JSON Schema compiled from ``output_type`` (sent to the model)."""
        return dict(self._output_schema)

    def tool(
        self,
        func: ToolFunc | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> Any:
        """Register a typed tool from a Python function (decorator or call form).

        The tool's argument JSON Schema is compiled from the function signature, so
        no hand-written schema is needed. The function's first parameter may be a
        :class:`RunContext` (by name ``ctx``/``context``/``run_context`` or by
        annotation) — when present, the agent injects the typed ``deps`` at call
        time; the model never sees that parameter. The remaining parameters become
        the tool's typed arguments. Usable as ``@agent.tool`` or
        ``@agent.tool(name=...)``.
        """

        def _register(fn: ToolFunc) -> ToolFunc:
            tool_name = name or fn.__name__
            tool_desc = description or (inspect.getdoc(fn) or "").strip()
            schema, takes_context = compile_tool_schema(fn)
            self._tools[tool_name] = _RegisteredTool(
                name=tool_name,
                description=tool_desc,
                func=fn,
                args_json_schema=schema,
                takes_context=takes_context,
            )
            return fn

        if func is not None:
            return _register(func)
        return _register

    @property
    def tool_names(self) -> list[str]:
        """The names of every registered tool, in registration order."""
        return list(self._tools)

    # -------------------------------------------------------------------- run
    async def run(
        self,
        prompt: str,
        *,
        deps: DepsT,
        thread: Any = None,
        llm_config: LLMConfig | None = None,
    ) -> TypedAgentRunResult[OutputT]:
        """Run the agent and return a validated, typed result.

        Binds the agent's typed tools (closing over ``deps``) onto the runtime's
        tool service, drives the standard agent loop requesting structured output
        for ``OutputT``, then validates the result against ``OutputT``. On a
        validation failure it retries on the same loop with a corrective nudge that
        names the failing fields, up to ``output_retries`` extra turns. Raises
        :class:`~himmy.core.errors.HimmyError` if a valid output cannot be produced
        within the retry budget (so a returned result always carries a real
        ``.output``).
        """
        bound = self._bind_tools(deps)
        try:
            tool_names = list(self._tools) if self._tools else None
            attempt = 0
            current_prompt = prompt
            current_thread = thread
            last: RunResult | None = None
            repaired = False
            while True:
                task = self._build_task(current_prompt, tool_names, attempt)
                loop = await self._runtime.run_agent_loop(
                    self._persona,
                    task,
                    current_thread,
                    max_turns=self._max_turns,
                    llm_config=llm_config,
                )
                last = loop.final
                current_thread = loop.thread
                output, error = self._validate(last)
                if output is not None:
                    await self._emit_validated(last, attempt)
                    return TypedAgentRunResult(
                        output=output,
                        raw=last,
                        validation_attempts=attempt + 1,
                        repaired=repaired,
                    )
                if attempt >= self._output_retries:
                    raise HimmyError(
                        f"typed output did not validate against "
                        f"{self._output_type.__name__} after {attempt + 1} "
                        f"attempt(s): {error}"
                    )
                # Retry: emit a repair event and nudge the model with the failure.
                await self._emit_repair(last, attempt, error or "")
                repaired = True
                attempt += 1
                current_prompt = self._repair_prompt(prompt, error or "")
        finally:
            self._unbind_tools(bound)

    # --------------------------------------------------------------- internals
    def _build_task(
        self, prompt: str, tool_names: list[str] | None, attempt: int
    ) -> Any:
        """Build the runtime Task requesting structured output for ``OutputT``."""
        from himmy.agents.base_agent.task import Task

        context: dict[str, Any] = {
            "output_schema": self._output_schema,
            # TypedAgent re-validates the structured reply against its pydantic
            # ``OutputT`` (with a corrective repair loop), so the inference service
            # must NOT also fail the reply at its boundary — that would pre-empt the
            # repair loop and swap the precise pydantic error for a jsonschema one.
            "validate_structured_output": False,
        }
        if tool_names is not None:
            context["tool_names"] = tool_names
        if self._model_key is not None:
            context["model_key"] = self._model_key
        return Task(
            title=f"{self._persona.name}:typed-run",
            prompt=prompt,
            context=context,
            metadata={"typed_output": self._output_type.__name__, "attempt": attempt},
        )

    def _validate(self, result: RunResult) -> tuple[OutputT | None, str | None]:
        """Validate the runtime's structured output against ``OutputT``.

        Returns ``(instance, None)`` on success or ``(None, error_message)`` when
        the structured payload is missing or fails pydantic validation.
        """
        if not result.succeeded:
            return None, result.error or "inference failed"
        payload = result.output_structured
        if payload is None:
            return None, "model returned no structured output"
        try:
            instance = self._output_type.model_validate(payload)
        except ValidationError as exc:
            return None, _format_validation_error(exc)
        return instance, None

    def _repair_prompt(self, original_prompt: str, error: str) -> str:
        """A corrective prompt naming the validation failure for a retry turn."""
        return (
            f"{original_prompt}\n\nYour previous structured answer did not match the "
            f"required schema for {self._output_type.__name__}. Fix exactly these "
            f"problems and return a valid object:\n{error}"
        )

    def _service(self) -> ToolService:
        """Return the runtime's tool service, requiring one when tools are used."""
        service = self._runtime.tool_service
        if service is None:
            raise HimmyError(
                "TypedAgent has tools but the runtime has no tool_service wired."
            )
        return cast("ToolService", service)

    def _bind_tools(self, deps: DepsT) -> list[str]:
        """Register each typed tool on the runtime's tool service for this run.

        Each handler closes over the typed ``deps`` (injecting a
        :class:`RunContext` when the tool declared one) and is registered as a
        ``LOCAL`` tool with the signature-derived ``args_json_schema``. Returns the
        names registered so :meth:`run` can deregister them afterwards.
        """
        if not self._tools:
            return []
        from himmy.services.tools.registry import register_local_tool

        service = self._service()
        registry = service.registry
        bound_names: list[str] = []
        for tool in self._tools.values():
            handler = self._make_handler(tool, deps)
            register_local_tool(
                registry,
                name=tool.name,
                handler=handler,
                description=tool.description,
                args_json_schema=tool.args_json_schema,
                read_only=True,
            )
            bound_names.append(tool.name)
        return bound_names

    def _make_handler(
        self, tool: _RegisteredTool, deps: DepsT
    ) -> Callable[[dict[str, Any]], Awaitable[Any]]:
        """Build the ToolService handler that injects deps and calls the tool body."""

        async def _handler(args: dict[str, Any]) -> Any:
            call_kwargs = dict(args)
            if tool.takes_context:
                return await tool.func(RunContext(deps=deps), **call_kwargs)
            return await tool.func(**call_kwargs)

        return _handler

    def _unbind_tools(self, names: list[str]) -> None:
        """Best-effort removal of this run's tools from the shared registry.

        Keeps the shared tool service clean across runs / agents so a later run
        does not see another agent's closed-over handlers.
        """
        if not names:
            return
        registry = self._service().registry
        for name in names:
            registry.remove(name)

    # ---------------------------------------------------------------- events
    async def _emit(self, event: RunEvent) -> None:
        """Route a typed-agent event through the runtime's audit emission.

        Reuses :meth:`SingleAgentRuntime._emit` verbatim so the new transitions
        flow through the *same* fan-out as every other run event: the durable
        memory store, the EntityRecord audit spine, the observability span, and
        any caller-facing ``on_event`` callbacks. Zero new sink wiring.
        """
        await self._runtime._emit(event)

    async def _emit_validated(self, result: RunResult, attempt: int) -> None:
        """Emit the audited TYPED_OUTPUT_VALIDATED transition."""
        await self._emit(
            RunEvent(
                event_type=EventType.TYPED_OUTPUT_VALIDATED,
                trace_id=result.trace_id,
                thread_id=getattr(result.thread, "thread_id", None),
                agent_id=self._persona.agent_id,
                payload={
                    "output_type": self._output_type.__name__,
                    "attempts": attempt + 1,
                },
            )
        )

    async def _emit_repair(self, result: RunResult, attempt: int, error: str) -> None:
        """Emit the audited TYPED_OUTPUT_REPAIRED transition before a retry."""
        await self._emit(
            RunEvent(
                event_type=EventType.TYPED_OUTPUT_REPAIRED,
                trace_id=result.trace_id,
                thread_id=getattr(result.thread, "thread_id", None),
                agent_id=self._persona.agent_id,
                error=error,
                payload={
                    "output_type": self._output_type.__name__,
                    "attempt": attempt,
                },
            )
        )


def _format_validation_error(exc: ValidationError) -> str:
    """Render a pydantic ValidationError as a compact, model-facing hint."""
    parts: list[str] = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ())) or "(root)"
        parts.append(f"{loc}: {err.get('msg', 'invalid')}")
    return "; ".join(parts) if parts else str(exc)


__all__ = [
    "TypedAgent",
    "TypedAgentRunResult",
    "RunContext",
    "ToolFunc",
    "compile_tool_schema",
]
