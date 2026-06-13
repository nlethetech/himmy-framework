"""Tools kernel: the tool catalog and local/HTTP registration helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from himmy.services.tools.models import (
    HttpToolConfig,
    ToolBackendKind,
    ToolDefinition,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.entities.protocol import EntityRegistryProtocol


class ToolRegistry:
    """In-memory catalog of tool definitions plus their local handlers.

    Definitions are JSON-serializable; the (possibly sync or async) Python
    callables for ``LOCAL`` tools are kept in a separate ``name -> handler`` map
    so ``ToolDefinition`` stays a clean data shape. When an ``EntityRegistry`` is
    supplied, each registration is also projected to a ``tool_definition`` record.
    """

    def __init__(
        self, *, entity_registry: EntityRegistryProtocol | None = None
    ) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._entity_registry = entity_registry

    @property
    def entity_registry(self) -> EntityRegistryProtocol | None:
        """The audit spine tool registrations project onto (``None`` when unwired).

        Exposed so co-registered packs (e.g. memory) can share the same registry
        and route their own writes onto one spine.
        """
        return self._entity_registry

    def register(
        self,
        definition: ToolDefinition,
        handler: Callable[..., Any] | None = None,
    ) -> ToolDefinition:
        """Register (or replace) a tool definition and its optional handler."""
        self._definitions[definition.name] = definition
        if handler is not None:
            self._handlers[definition.name] = handler
        if self._entity_registry is not None:
            record = definition.to_record()
            self._entity_registry.register(record)
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        """Return the definition for ``name`` or ``None`` when unknown."""
        return self._definitions.get(name)

    def remove(self, name: str) -> bool:
        """Drop a tool definition and its handler; return True if one existed.

        Used by short-lived registrations (e.g. a typed agent binding per-run,
        deps-closing tools) to keep a shared registry clean across runs. The entity
        projection is append-only by design, so a removal does not retract any
        already-registered ``tool_definition`` record from the audit spine.
        """
        existed = self._definitions.pop(name, None) is not None
        self._handlers.pop(name, None)
        return existed

    def list(self) -> list[ToolDefinition]:
        """Return every registered tool definition."""
        return list(self._definitions.values())

    def handler_for(self, name: str) -> Callable[..., Any] | None:
        """Return the local Python handler registered for ``name`` (if any)."""
        return self._handlers.get(name)


def register_local_tool(
    registry: ToolRegistry,
    *,
    name: str,
    handler: Callable[..., Any],
    description: str = "",
    args_json_schema: dict[str, Any] | None = None,
    output_json_schema: dict[str, Any] | None = None,
    requires_approval: bool = False,
    sequential: bool = False,
    timeout_seconds: float | None = None,
    retry_hints: dict[str, Any] | None = None,
    sensitive_arg_names: list[str] | None = None,
    read_only: bool | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolDefinition:
    """Register a ``LOCAL`` tool backed by a Python callable.

    The handler may be synchronous or a coroutine function; the service handles
    both. ``timeout_seconds``, ``retry_hints``, and ``sensitive_arg_names`` are
    all reachable so the service can enforce/redact them. ``read_only`` declares
    whether the tool only reads (vs mutates) — surfaced to small models so a
    look-up doesn't land on a write tool; omit it to infer from the name. Returns
    the created :class:`ToolDefinition`.
    """
    definition = ToolDefinition(
        name=name,
        kind=ToolBackendKind.LOCAL,
        description=description,
        args_json_schema=args_json_schema or {},
        output_json_schema=output_json_schema,
        requires_approval=requires_approval,
        read_only=read_only,
        sequential=sequential,
        timeout_seconds=timeout_seconds,
        retry_hints=retry_hints or {},
        sensitive_arg_names=sensitive_arg_names or [],
        metadata=metadata or {},
    )
    return registry.register(definition, handler=handler)


def register_http_tool(
    registry: ToolRegistry,
    *,
    name: str,
    http_config: HttpToolConfig,
    description: str = "",
    args_json_schema: dict[str, Any] | None = None,
    output_json_schema: dict[str, Any] | None = None,
    requires_approval: bool = False,
    sequential: bool = False,
    timeout_seconds: float | None = None,
    retry_hints: dict[str, Any] | None = None,
    sensitive_arg_names: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ToolDefinition:
    """Register an ``HTTP`` connector described declaratively by ``http_config``.

    No per-tool client code is needed: the service resolves the base URL, path
    placeholders, query/body/header args, and env-backed auth at call time.
    Definition-level ``timeout_seconds`` (overrides ``http_config.timeout_seconds``),
    ``retry_hints``, ``sequential``, and ``sensitive_arg_names`` are all reachable.
    """
    definition = ToolDefinition(
        name=name,
        kind=ToolBackendKind.HTTP,
        description=description,
        args_json_schema=args_json_schema or {},
        output_json_schema=output_json_schema,
        requires_approval=requires_approval,
        sequential=sequential,
        timeout_seconds=timeout_seconds,
        retry_hints=retry_hints or {},
        sensitive_arg_names=sensitive_arg_names or [],
        http_config=http_config,
        metadata=metadata or {},
    )
    return registry.register(definition)
