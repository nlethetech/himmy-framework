"""MCP kernel: server spec + tool/result types for the stdio client."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MCPServerSpec(BaseModel):
    """How to launch a stdio MCP server subprocess.

    ``env`` is overlaid on the inherited environment (MCP servers are
    operator-configured *trusted* processes — unlike the untrusted code the
    sandbox kernel runs — so they inherit the parent env by default).
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None


class MCPTool(BaseModel):
    """A tool advertised by an MCP server (its ``inputSchema`` is JSON Schema)."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class MCPToolResult(BaseModel):
    """The outcome of an MCP ``tools/call``.

    ``text`` is the concatenation of the result's text content blocks (the common
    case); ``content`` is the raw block list; ``structured`` is the optional
    ``structuredContent``; ``is_error`` mirrors the MCP ``isError`` flag.
    """

    text: str = ""
    content: list[dict[str, Any]] = Field(default_factory=list)
    structured: Any = None
    is_error: bool = False


__all__ = ["MCPServerSpec", "MCPTool", "MCPToolResult"]
