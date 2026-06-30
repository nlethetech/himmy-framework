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
            description="reads", read_only=True,
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


def test_skill_dispatch_subagent_inherits_parent_gate() -> None:
    """A dispatched skill's sub-agent ToolService carries the parent's (attenuated) gate.

    Red-team r2 confused-deputy fix: ``dispatch_skill`` previously built its sub-runtime
    with a bare ``tool_registry`` override (no authorizer) → a default UN-gated
    ToolService, so a narrowed caller could route WITHHELD write tools (``write_file``,
    ``run_python``) through a skill's tool-packs. The dispatcher must now build the sub
    ToolService with the parent's attenuated gate, exactly like ``spawn``.
    """
    from himmy.skills import build_skill_registry
    from himmy.skills.dispatch import SkillDispatcher

    spec = AgentSpec(name="d", provider="stub", allow_skill_dispatch=True)
    reader = Principal(
        subject="u", roles=frozenset({"reader"}), tenant_ids=frozenset({"ws1"})
    )
    authz = ToolCapabilityAuthorizer.from_principal(reader, _policy())
    runtime, registry = build_runtime_for_spec(spec, tool_authorizer=authz)
    # dispatch_skill is registered at the parent (gated by the parent's tool service).
    assert "dispatch_skill" in {d.name for d in registry.list()}

    # Drive the dispatcher's sub-runtime build and CAPTURE the overrides it passes to
    # build_runtime — the sub ToolService must carry the attenuated (enforcing) gate, not
    # a bare tool_registry override (which would yield an un-gated default ToolService).
    captured: dict[str, object] = {}

    def _fake_build_runtime(**overrides: object) -> tuple[object, object, object]:
        captured.update(overrides)

        class _Loop:
            final = None
            stopped_reason = "done"

        class _Runtime:
            async def run_agent_loop(self, *a: object, **k: object) -> object:
                return _Loop()

        return _Runtime(), None, None

    import himmy.runtime.builder as builder_mod

    orig = builder_mod.build_runtime
    builder_mod.build_runtime = _fake_build_runtime  # type: ignore[assignment]
    try:
        dispatcher = SkillDispatcher(
            inference=object(),
            skill_registry=build_skill_registry(),
            tool_authorizer=authz,
        )
        # ``file_ops`` bundles the ``files`` pack (write_file) — a write surface.
        run_async(dispatcher.run("file_ops", "do nothing"))
    finally:
        builder_mod.build_runtime = orig  # type: ignore[assignment]

    # The sub-runtime was built with a GATED ToolService, not a bare registry override.
    assert "tool_service" in captured
    assert "tool_registry" not in captured
    sub_service = captured["tool_service"]
    sub_gate = sub_service._tool_authorizer  # type: ignore[attr-defined]
    assert sub_gate is not None and sub_gate.enforce is True
    # The reader (no tool:write_file:write) cannot reach the write tool through the gate.
    assert sub_gate.is_authorized("write_file", False) is False


def test_skill_dispatch_offline_subagent_is_ungated_byte_unchanged() -> None:
    """INVARIANT: with no/NON-enforcing authorizer the sub-runtime is byte-unchanged.

    The offline ANONYMOUS principal yields a NON-enforcing authorizer; the dispatcher
    must fall back to the bare ``tool_registry`` override (no ToolService injection), so
    the zero-config dispatch path is identical to before the hardening.
    """
    from himmy.skills import build_skill_registry
    from himmy.skills.dispatch import SkillDispatcher

    captured: dict[str, object] = {}

    def _fake_build_runtime(**overrides: object) -> tuple[object, object, object]:
        captured.update(overrides)

        class _Loop:
            final = None
            stopped_reason = "done"

        class _Runtime:
            async def run_agent_loop(self, *a: object, **k: object) -> object:
                return _Loop()

        return _Runtime(), None, None

    import himmy.runtime.builder as builder_mod

    orig = builder_mod.build_runtime
    builder_mod.build_runtime = _fake_build_runtime  # type: ignore[assignment]
    try:
        # tool_authorizer=None (the offline default) → bare registry override, no gate.
        dispatcher = SkillDispatcher(
            inference=object(), skill_registry=build_skill_registry()
        )
        run_async(dispatcher.run("file_ops", "do nothing"))
    finally:
        builder_mod.build_runtime = orig  # type: ignore[assignment]

    assert "tool_registry" in captured
    assert "tool_service" not in captured
