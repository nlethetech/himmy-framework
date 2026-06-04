"""Tests for the transport-direct MCP stdio client, against a mock server."""

from __future__ import annotations

import os
import sys
from typing import Any

import pytest

from himmy.services.mcp import (
    MCPClient,
    MCPError,
    MCPServerSpec,
    register_mcp_tools,
)
from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from tests.conftest import run_async

_MOCK_SERVER = os.path.join(os.path.dirname(__file__), "mock_mcp_server.py")


def _spec() -> MCPServerSpec:
    return MCPServerSpec(command=sys.executable, args=[_MOCK_SERVER])


def _with_client(scenario) -> Any:
    """Connect, run ``scenario(client)``, then always close the client."""

    async def _run() -> Any:
        client = await MCPClient.connect(_spec())
        try:
            return await scenario(client)
        finally:
            await client.aclose()

    return run_async(_run())


def test_handshake_populates_server_info_and_lists_tools() -> None:
    """connect() completes the MCP handshake; tools/list returns the catalog."""

    async def scenario(client: MCPClient) -> Any:
        return client.server_info, client.protocol_version, await client.list_tools()

    server_info, protocol_version, tools = _with_client(scenario)
    assert server_info["name"] == "mock-mcp"
    assert protocol_version == "2024-11-05"
    by_name = {t.name: t for t in tools}
    assert {"echo", "add"} <= set(by_name)
    assert "text" in by_name["echo"].input_schema["properties"]


def test_call_tool_echo_and_add() -> None:
    """tools/call returns normalized text content for both tools."""

    async def scenario(client: MCPClient) -> Any:
        echo = await client.call_tool("echo", {"text": "hello mcp"})
        add = await client.call_tool("add", {"a": 2, "b": 3})
        return echo, add

    echo, add = _with_client(scenario)
    assert echo.text == "hello mcp"
    assert echo.is_error is False
    assert add.text == "5"


def test_unknown_tool_is_reported_as_error_result() -> None:
    """A tool-level failure surfaces as is_error, not an exception."""

    async def scenario(client: MCPClient) -> Any:
        return await client.call_tool("nope", {})

    result = _with_client(scenario)
    assert result.is_error is True


def test_jsonrpc_error_raises_mcp_error() -> None:
    """A JSON-RPC error response is raised as an MCPError with its code."""

    async def scenario(client: MCPClient) -> Any:
        with pytest.raises(MCPError) as excinfo:
            await client.call_tool("explode", {})
        return excinfo.value.code

    assert _with_client(scenario) == -32000


def test_register_mcp_tools_bridges_into_toolservice() -> None:
    """MCP tools register into a ToolRegistry and run through ToolService."""

    async def scenario(client: MCPClient) -> Any:
        registry = ToolRegistry()
        names = await register_mcp_tools(registry, client, prefix="mock_")
        service = ToolService(registry)
        result = await service.execute(
            ToolInvocation(tool_name="mock_echo", args={"text": "via toolservice"})
        )
        return names, result

    names, result = _with_client(scenario)
    assert set(names) == {"mock_echo", "mock_add"}
    assert result.outcome == "success"
    assert result.result["text"] == "via toolservice"
    assert result.result["is_error"] is False


def test_aclose_is_idempotent() -> None:
    """Closing twice is safe."""

    async def _run() -> None:
        client = await MCPClient.connect(_spec())
        await client.aclose()
        await client.aclose()

    run_async(_run())
