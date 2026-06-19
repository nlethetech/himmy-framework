"""MCP kernel: register an MCP server's tools into the Himmy ToolService.

Each MCP tool becomes a LOCAL :class:`ToolDefinition` whose handler routes through
the :class:`MCPClient`, so MCP tools flow through the SAME pipeline as native ones
— arg validation against the MCP ``inputSchema``, approval gating, events, and
lineage. MCP tool errors are returned in-band (``is_error``) rather than raised,
matching the protocol's semantics.
"""

from __future__ import annotations

from collections.abc import Callable, Collection
from typing import Any

from himmy.core.errors import HimmyError
from himmy.services.mcp.client import MCPClient
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.schema_normalize import normalize_tool_schema


def _refuse_builtin_shadow(registry: ToolRegistry, tool_name: str) -> None:
    """Raise when ``tool_name`` would overwrite an existing non-MCP (built-in) tool.

    The registry replaces by name, so an MCP server advertising a ``web_fetch`` /
    ``sql_query`` / etc. with an empty ``prefix`` could silently shadow the built-in
    of that name and capture the agent's calls + arguments. Registering one MCP tool
    over another MCP tool (re-attaching the same server) stays allowed.
    """
    existing = registry.get(tool_name)
    if existing is not None and existing.metadata.get("backend") != "mcp":
        raise HimmyError(
            f"MCP tool {tool_name!r} would shadow a built-in tool of the same name; "
            "set a `prefix` (e.g. prefix: srv_) to namespace this server's tools"
        )


def _make_handler(client: MCPClient, mcp_name: str) -> Callable[[dict[str, Any]], Any]:
    """Build a tool handler that proxies one MCP tool through the client."""

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        result = await client.call_tool(mcp_name, args)
        return {
            "text": result.text,
            "is_error": result.is_error,
            "content": result.content,
            "structured": result.structured,
        }

    return _handler


async def register_mcp_tools(
    registry: ToolRegistry,
    client: MCPClient,
    *,
    prefix: str = "",
    requires_approval: bool = False,
    names: Collection[str] | None = None,
) -> list[str]:
    """Discover and register an MCP server's tools; return the registered names.

    ``prefix`` namespaces the tool names (e.g. ``"github_"``) to avoid collisions
    when bridging multiple servers; ``names`` restricts to a subset; each tool's
    MCP ``inputSchema`` becomes its ``args_json_schema``.
    """
    allow = set(names) if names is not None else None
    registered: list[str] = []
    for tool in await client.list_tools():
        if allow is not None and tool.name not in allow:
            continue
        tool_name = f"{prefix}{tool.name}"
        _refuse_builtin_shadow(registry, tool_name)
        # MCP servers commonly ship schemas with ``$ref``/``$defs`` (the single most
        # common cross-provider tool-schema failure). Run the provider-agnostic
        # normalizer (``None`` profile = ref-inlining + def-stripping only, no
        # provider narrowing) at registration; the per-provider egress normalizer
        # runs again at the manager (idempotent + widen-only, so this is safe).
        args_schema = normalize_tool_schema(tool.input_schema, None)
        register_local_tool(
            registry,
            name=tool_name,
            handler=_make_handler(client, tool.name),
            description=tool.description,
            args_json_schema=args_schema,
            requires_approval=requires_approval,
            metadata={"backend": "mcp", "mcp_tool": tool.name},
        )
        registered.append(tool_name)
    return registered


__all__ = ["register_mcp_tools"]
