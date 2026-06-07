"""Tools kernel: pydantic-ai toolset binding (optional [providers] extra).

This module is import-safe without ``pydantic_ai``. The real binding — converting
each :class:`~himmy.services.tools.models.ToolDefinition` into a pydantic-ai
``Tool`` with a *generated argument model* derived from ``args_json_schema`` (so
the model sees a real function signature and pydantic-ai validates arguments +
issues ``ModelRetry``) — is only attempted when the model is actually run through
pydantic-ai. Offline, the runtime uses the ``BoundTool`` path from
:meth:`ToolService.bound_tools` instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import create_model

from himmy.core.errors import HimmyError
from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.service import ToolService

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.tools.models import ToolDefinition

# Map JSON-schema primitive types to Python types for the generated arg model.
_JSON_TYPE_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def _py_type_for(sub_schema: dict[str, Any]) -> Any:
    """Resolve a Python type for a property sub-schema (falls back to ``Any``)."""
    json_type = sub_schema.get("type")
    if isinstance(json_type, list):
        # Union: pick the first known concrete type, default Any.
        for candidate in json_type:
            if candidate in _JSON_TYPE_TO_PY and candidate != "null":
                return _JSON_TYPE_TO_PY[candidate]
        return Any
    return _JSON_TYPE_TO_PY.get(json_type, Any) if isinstance(json_type, str) else Any


def build_arg_model(name: str, args_json_schema: dict[str, Any]) -> type:
    """Build a pydantic model class from a tool's object ``args_json_schema``.

    Each declared property becomes a field; properties in ``required`` are
    required, the rest default to ``None``. Tools with no object schema get an
    empty model. Exposed (not gated) so it is unit-testable without pydantic-ai.
    """
    model_name = f"{name.title().replace('_', '')}Args"
    if (
        not isinstance(args_json_schema, dict)
        or args_json_schema.get("type") != "object"
    ):
        return cast(type, create_model(model_name))  # type: ignore[call-overload]

    properties = args_json_schema.get("properties", {}) or {}
    required = set(args_json_schema.get("required", []) or [])
    fields: dict[str, tuple[Any, Any]] = {}
    for prop_name, sub in properties.items():
        if not isinstance(sub, dict):
            sub = {}
        py_type = _py_type_for(sub)
        if prop_name in required:
            fields[prop_name] = (py_type, ...)
        else:
            fields[prop_name] = (py_type | None if py_type is not Any else Any, None)
    return cast(type, create_model(model_name, **fields))  # type: ignore[call-overload]


class ToolServiceToolset:
    """Adapter that exposes a :class:`ToolService` as a pydantic-ai toolset.

    Construction is import-safe; ``pydantic_ai`` is only required when
    :meth:`as_pydantic_ai_toolset` is invoked. Each pydantic-ai ``Tool`` advertises
    a typed argument model (built from ``args_json_schema``) and proxies its call
    back through :meth:`ToolService.execute` so policy hooks, validation, and
    events all still apply.
    """

    def __init__(
        self, tool_service: ToolService, *, names: list[str] | None = None
    ) -> None:
        self._tool_service = tool_service
        self._names = names

    def _selected(self) -> list[ToolDefinition]:
        """Return the tool definitions this toolset exposes."""
        registry = self._tool_service.registry
        if self._names is None:
            return registry.list()
        return [d for n in self._names if (d := registry.get(n)) is not None]

    @staticmethod
    def _require_pydantic_ai() -> Any:
        """Import ``pydantic_ai`` or raise a clear, actionable error."""
        try:
            import pydantic_ai  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised only with extra
            raise HimmyError(
                "pydantic-ai not installed; install the providers extra: "
                "pip install 'himmy[providers]'"
            ) from exc
        return pydantic_ai

    def as_pydantic_ai_toolset(self) -> Any:
        """Build a pydantic-ai toolset routing through this :class:`ToolService`.

        Requires the ``[providers]`` extra. Each generated tool exposes a typed
        argument model derived from ``args_json_schema`` so pydantic-ai validates
        arguments (and can ``ModelRetry``) before :meth:`ToolService.execute`.
        """
        self._require_pydantic_ai()
        from pydantic_ai.tools import Tool  # type: ignore  # pragma: no cover

        service = self._tool_service
        tools: list[Any] = []
        for definition in self._selected():  # pragma: no cover - needs extra
            arg_model = build_arg_model(definition.name, definition.args_json_schema)

            def _make_proxy(tool_name: str, model: type) -> Any:
                async def _proxy(args):  # type: ignore[no-untyped-def]
                    payload = (
                        args.model_dump(exclude_none=True)  # type: ignore[attr-defined]
                        if hasattr(args, "model_dump")
                        else dict(args)
                    )
                    result = await service.execute(
                        ToolInvocation(tool_name=tool_name, args=payload)
                    )
                    if result.outcome != "success":
                        # ModelRetry signals pydantic-ai to re-prompt the model.
                        from pydantic_ai import ModelRetry  # type: ignore

                        raise ModelRetry(
                            f"tool {tool_name!r} failed: {result.error_message}"
                        )
                    return result.result

                # This module uses PEP 563 (``from __future__ import annotations``),
                # which would stringify an inline ``args: model`` annotation to the
                # literal ``"model"`` — a closure local that pydantic-ai's
                # ``get_type_hints()`` can't resolve (NameError at bind time). Assign
                # the real generated arg-model class object directly so no eval runs.
                _proxy.__annotations__ = {"args": model, "return": Any}
                return _proxy

            tools.append(
                Tool(
                    _make_proxy(definition.name, arg_model),
                    name=definition.name,
                    description=definition.description,
                )
            )
        return tools
