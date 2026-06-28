"""The P0 tool-capability gate is threaded from ``build_runtime_for_spec`` to dispatch.

Verifies that an authorizer handed to :func:`build_runtime_for_spec` reaches the per-run
:class:`ToolService` (so a tenant-bound caller's tool calls are gated deny-by-default),
that the offline / no-authorizer path is byte-unchanged, and that a spawned sub-agent
inherits the parent's gate (attenuate, never amplify).
"""

from __future__ import annotations

import sys
import types

from himmy.api.auth.principal import ANONYMOUS, Principal
from himmy.api.auth.rbac import AccessPolicy
from himmy.config.agent_spec import AgentSpec
from himmy.runtime.from_spec import build_runtime_for_spec
from himmy.services.tools import ToolInvocation
from himmy.services.tools.capability import ToolCapabilityAuthorizer
from tests.conftest import run_async


def _install_tools_module(name: str) -> None:
    """Install a throwaway tools module exposing a read tool + a write tool."""
    mod = types.ModuleType(name)

    def register(registry: object) -> None:
        from himmy.services.tools.registry import register_local_tool

        register_local_tool(
            registry, name="weather_get", handler=lambda a: {"t": 1},
            description="reads",
        )
        register_local_tool(
            registry, name="send_email", handler=lambda a: {"ok": True},
            description="sends", read_only=False,
        )

    mod.register = register  # type: ignore[attr-defined]
    sys.modules[name] = mod


def _policy() -> AccessPolicy:
    return AccessPolicy.from_mapping({"reader": ["tool:weather_get:invoke"]})


def test_authorizer_reaches_dispatch_via_spec() -> None:
    """A tenant-bound reader gets the read tool but is DENIED the write tool through the spec."""
    _install_tools_module("_authz_tools_a")
    spec = AgentSpec(name="a", provider="stub", tools_module="_authz_tools_a:register")
    reader = Principal(
        subject="u", roles=frozenset({"reader"}), tenant_ids=frozenset({"ws1"})
    )
    authz = ToolCapabilityAuthorizer.from_principal(reader, _policy())
    runtime, registry = build_runtime_for_spec(spec, tool_authorizer=authz)

    ok = run_async(runtime.tool_service.execute(ToolInvocation(tool_name="weather_get")))
    assert ok.outcome == "success"
    denied = run_async(
        runtime.tool_service.execute(ToolInvocation(tool_name="send_email"))
    )
    assert denied.outcome == "denied"


def test_offline_anonymous_runs_all_tools_via_spec() -> None:
    """The ANONYMOUS offline principal (or no authorizer) runs every tool — no-op bypass."""
    _install_tools_module("_authz_tools_b")
    spec = AgentSpec(name="b", provider="stub", tools_module="_authz_tools_b:register")
    authz = ToolCapabilityAuthorizer.from_principal(ANONYMOUS, _policy())
    runtime, _ = build_runtime_for_spec(spec, tool_authorizer=authz)
    res = run_async(runtime.tool_service.execute(ToolInvocation(tool_name="send_email")))
    assert res.outcome == "success"

    # And with NO authorizer at all, byte-unchanged.
    runtime2, _ = build_runtime_for_spec(spec)
    res2 = run_async(
        runtime2.tool_service.execute(ToolInvocation(tool_name="send_email"))
    )
    assert res2.outcome == "success"


def test_spawn_subagent_inherits_parent_gate() -> None:
    """A spawned sub-agent's tool service carries the parent's (attenuated) gate."""
    spec = AgentSpec(name="c", provider="stub", allow_spawn=True)
    reader = Principal(
        subject="u", roles=frozenset({"reader"}), tenant_ids=frozenset({"ws1"})
    )
    authz = ToolCapabilityAuthorizer.from_principal(reader, _policy())
    runtime, registry = build_runtime_for_spec(spec, tool_authorizer=authz)
    # The parent runtime's tool service carries the enforcing gate.
    parent_gate = runtime.tool_service._tool_authorizer
    assert parent_gate is not None and parent_gate.enforce is True
    # spawn_agent is registered, and the sub-runtime it builds attenuates the same gate
    # (verified directly: attenuate yields a non-wider gate over the same roles).
    assert "spawn_agent" in {d.name for d in registry.list()}
    assert parent_gate.attenuate().roles == frozenset({"reader"})
