"""Declarative MCP servers: plug any stdio MCP server into an agent via YAML.

The framework already speaks the Model Context Protocol over stdio
(:mod:`himmy.services.mcp`); this module makes it *declarative*, so an ``agent.yaml``
or ``team.yaml`` can pull in the whole MCP ecosystem — GitHub, Slack, a filesystem
server, a browser (Playwright), … — as ordinary agent tools without writing code::

    mcp_servers:
      - command: npx
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        prefix: fs_                # namespace the tool names (avoid collisions)
      - command: uvx
        args: ["mcp-server-git"]
        requires_approval: true    # gate this server's tools behind approval
        tools: [git_log, git_diff] # bind only a subset (empty = all)

Each MCP tool becomes a native :class:`ToolDefinition` (arg validation, approval
gating, events, lineage), so it flows through the same pipeline as built-in tools.
When the agent's ``tools:`` list is empty, every registered tool — MCP ones
included — is advertised to the model; otherwise list the (prefixed) MCP tool names
you want bound.
"""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from himmy.services.mcp.models import MCPServerSpec

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.mcp.client import MCPClient
    from himmy.services.tools.registry import ToolRegistry


class MCPServerConfig(BaseModel):
    """Declarative launch + binding config for one stdio MCP server."""

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    prefix: str = ""  # namespace for the tool names, e.g. "github_"
    requires_approval: bool = False  # gate this server's tools behind approval
    tools: list[str] = Field(default_factory=list)  # subset to bind (empty = all)

    def to_server_spec(self) -> MCPServerSpec:
        """Project to the transport-level :class:`MCPServerSpec` the client launches."""
        return MCPServerSpec(
            command=self.command, args=list(self.args), env=dict(self.env), cwd=self.cwd
        )


async def attach_mcp_servers(
    registry: ToolRegistry, configs: list[MCPServerConfig]
) -> list[MCPClient]:
    """Connect each configured MCP server and register its tools onto ``registry``.

    Returns the live clients so the caller can close them when the run finishes. The
    clients' reader tasks are bound to the *current* event loop — connect, run, and
    close must all happen inside one ``asyncio.run`` so the loop stays alive.
    """
    from himmy.services.mcp.client import MCPClient
    from himmy.services.mcp.connector import register_mcp_tools

    clients: list[MCPClient] = []
    try:
        for cfg in configs:
            client = await MCPClient.connect(cfg.to_server_spec())
            clients.append(client)
            await register_mcp_tools(
                registry,
                client,
                prefix=cfg.prefix,
                requires_approval=cfg.requires_approval,
                names=list(cfg.tools) or None,
            )
    except Exception:
        await close_mcp_clients(clients)
        raise
    return clients


async def close_mcp_clients(clients: list[MCPClient]) -> None:
    """Terminate every MCP server subprocess (best-effort, reverse order)."""
    for client in reversed(clients):
        with suppress(Exception):
            await client.aclose()


__all__ = ["MCPServerConfig", "attach_mcp_servers", "close_mcp_clients"]
