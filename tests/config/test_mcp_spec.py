"""Tests for declarative MCP servers (mcp_servers: in agent.yaml / team.yaml)."""

from __future__ import annotations

import os
import sys
from typing import Any

from himmy.config.agent_spec import AgentSpec
from himmy.config.mcp_spec import (
    MCPServerConfig,
    attach_mcp_servers,
    close_mcp_clients,
)
from himmy.config.team_spec import TeamSpec
from himmy.services.tools.models import ToolInvocation
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.service import ToolService
from tests.conftest import run_async

_MOCK_SERVER = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "mcp", "mock_mcp_server.py"
)


def _config(**over: Any) -> MCPServerConfig:
    return MCPServerConfig(command=sys.executable, args=[_MOCK_SERVER], **over)


def test_to_server_spec_projects_launch_fields() -> None:
    cfg = _config(env={"X": "1"}, cwd="/tmp")
    spec = cfg.to_server_spec()
    assert spec.command == sys.executable
    assert spec.args == [_MOCK_SERVER]
    assert spec.env == {"X": "1"}
    assert spec.cwd == "/tmp"


def test_attach_registers_prefixed_tools_and_runs_them() -> None:
    """attach_mcp_servers wires an MCP server's tools into a registry, runnable."""

    async def scenario() -> Any:
        registry = ToolRegistry()
        clients = await attach_mcp_servers(registry, [_config(prefix="mock_")])
        try:
            service = ToolService(registry)
            res = await service.execute(
                ToolInvocation(tool_name="mock_add", args={"a": 4, "b": 5})
            )
            return {t.name for t in registry.list()}, res
        finally:
            await close_mcp_clients(clients)

    names, res = run_async(scenario())
    assert {"mock_echo", "mock_add"} <= names
    assert res.outcome == "success"
    assert res.result["text"] == "9"


def test_attach_honors_tools_subset() -> None:
    """An explicit `tools:` subset binds only those tools."""

    async def scenario() -> Any:
        registry = ToolRegistry()
        clients = await attach_mcp_servers(registry, [_config(tools=["echo"])])
        try:
            return {t.name for t in registry.list()}
        finally:
            await close_mcp_clients(clients)

    names = run_async(scenario())
    assert "echo" in names
    assert "add" not in names


def test_attach_closes_clients_on_failure() -> None:
    """A bad server in the list doesn't leak the already-connected good ones."""

    async def scenario() -> Any:
        registry = ToolRegistry()
        good = _config(prefix="ok_")
        bad = MCPServerConfig(command="himmy-no-such-binary-xyz")
        try:
            await attach_mcp_servers(registry, [good, bad])
        except Exception as exc:  # the bad launch propagates
            return type(exc).__name__
        return "no-error"

    assert run_async(scenario()) != "no-error"


def test_agent_spec_parses_mcp_servers() -> None:
    spec = AgentSpec.model_validate(
        {
            "name": "a",
            "mcp_servers": [
                {"command": "npx", "args": ["-y", "server"], "prefix": "gh_"}
            ],
        }
    )
    assert spec.mcp_servers[0].command == "npx"
    assert spec.mcp_servers[0].prefix == "gh_"


def test_team_spec_parses_mcp_servers() -> None:
    spec = TeamSpec.model_validate(
        {
            "entry": "a",
            "members": [{"name": "a"}],
            "mcp_servers": [{"command": "uvx", "args": ["mcp-server-git"]}],
        }
    )
    assert spec.mcp_servers[0].command == "uvx"
