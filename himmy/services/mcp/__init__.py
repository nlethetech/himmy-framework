"""MCP kernel: a transport-direct Model Context Protocol stdio client.

Consume any stdio MCP server's tools through the Himmy ToolService — no SDK
dependency, fully offline-testable. ``MCPClient`` speaks newline-delimited
JSON-RPC 2.0; ``register_mcp_tools`` bridges a server's tools into a
:class:`~himmy.services.tools.registry.ToolRegistry`.
"""

from __future__ import annotations

from himmy.services.mcp.client import (
    DEFAULT_PROTOCOL_VERSION,
    MCPClient,
    MCPError,
)
from himmy.services.mcp.connector import register_mcp_tools
from himmy.services.mcp.models import MCPServerSpec, MCPTool, MCPToolResult

__all__ = [
    "MCPClient",
    "MCPError",
    "MCPServerSpec",
    "MCPTool",
    "MCPToolResult",
    "register_mcp_tools",
    "DEFAULT_PROTOCOL_VERSION",
]
