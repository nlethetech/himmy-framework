"""centralize-tool-gate: the AMBIENT tool-capability authorizer at the chokepoint.

The companion to ``tests/tools/test_tool_capability_authz.py`` (which proves the EXPLICIT
per-``ToolService`` gate). This proves the ambient mechanism that makes it IMPOSSIBLE to
run a tool un-gated when an authorizer is in scope, regardless of whether the constructing
site remembered to thread the gate:

* a ``ToolService`` built WITHOUT an explicit authorizer still enforces the AMBIENT one set
  on the context (the "forgotten thread" case the redesign closes);
* the explicit and ambient gates INTERSECT — a tool needs BOTH to allow it, so an
  attenuated sub-agent gate can only narrow and a wider explicit gate cannot amplify;
* when auth is CONFIGURED and NEITHER gate is in scope, the chokepoint FAILS CLOSED
  (``CAPABILITY_DENIED``) — a future path that forgets both is safe by construction;
* the OFFLINE invariant: no auth configured + no ambient + no explicit → pure pass,
  byte-unchanged.
"""

from __future__ import annotations

import pytest

from himmy.api.auth.principal import ANONYMOUS, Principal
from himmy.api.auth.rbac import AccessPolicy
from himmy.services.tools import (
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_local_tool,
)
from himmy.services.tools.ambient import (
    auth_is_configured,
    current_authorizer,
    mark_auth_configured,
    resolve_effective_authorizer,
    use_tool_authorizer,
)
from himmy.services.tools.capability import ToolCapabilityAuthorizer
from tests.conftest import run_async


def _registry() -> ToolRegistry:
    """A registry with a read tool (``weather_get``) and a write tool (``send_email``)."""
    registry = ToolRegistry()

    def weather(args: dict) -> dict:
        return {"temp": 20}

    def send(args: dict) -> dict:
        return {"sent": True}

    register_local_tool(
        registry,
        name="weather_get",
        handler=weather,
        description="reads weather",
        read_only=True,
    )
    register_local_tool(
        registry,
        name="send_email",
        handler=send,
        description="sends an email",
        read_only=False,
    )
    return registry


def _policy() -> AccessPolicy:
    return AccessPolicy.from_mapping(
        {
            "reader": ["tool:weather_get:invoke"],
            "mailer": ["tool:send_email:invoke", "tool:send_email:write"],
            "anytool": ["tool:*"],
            "admin": ["*:*"],
        }
    )


def _authz(*roles: str) -> ToolCapabilityAuthorizer:
    return ToolCapabilityAuthorizer.from_principal(
        Principal(
            subject="u", roles=frozenset(roles), tenant_ids=frozenset({"ws1"})
        ),
        _policy(),
    )


@pytest.fixture(autouse=True)
def _reset_auth_flag():
    """Each test starts with auth NOT configured and no ambient gate; restore after."""
    mark_auth_configured(False)
    yield
    mark_auth_configured(False)


# ---- the core guarantee: a forgotten explicit thread still enforces ----------


def test_ambient_gate_enforces_a_service_built_without_explicit_authorizer() -> None:
    """A ToolService with NO explicit authorizer still consults the AMBIENT one.

    This is the by-construction failure mode the redesign kills: a construction site that
    forgot to pass ``tool_authorizer=`` no longer runs un-gated when an authorizer is in
    scope on the context.
    """
    svc = ToolService(_registry())  # forgot to thread the gate
    with use_tool_authorizer(_authz("reader")):
        ok = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
        denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert ok.outcome == "success"
    assert denied.outcome == "denied"
    assert denied.error_code is ToolErrorCode.CAPABILITY_DENIED


def test_ambient_binding_is_scoped_and_resets() -> None:
    """The ambient gate only applies inside the ``with`` block; it resets on exit."""
    svc = ToolService(_registry())
    assert current_authorizer() is None
    with use_tool_authorizer(_authz("reader")):
        assert current_authorizer() is not None
    assert current_authorizer() is None
    # Outside the block (and with no auth configured) the same forgot-the-thread service
    # is a pure pass again — byte-unchanged.
    res = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert res.outcome == "success"


# ---- intersection: attenuate-only, no amplification --------------------------


def test_explicit_and_ambient_intersect_no_amplification() -> None:
    """A WIDER explicit gate cannot amplify past a NARROWER ambient one (and vice-versa).

    The effective gate is the INTERSECTION: the tool needs BOTH to allow it. So even a
    buggy site that passes a wider explicit authorizer than the ambient principal's reach
    is still bounded by the ambient gate.
    """
    # Explicit gate can do everything (``anytool``); ambient gate is read-only (``reader``).
    svc = ToolService(_registry(), tool_authorizer=_authz("anytool"))
    with use_tool_authorizer(_authz("reader")):
        ok = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
        # The write tool is allowed by the explicit gate but DENIED by the ambient one →
        # the intersection denies it. No amplification.
        denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert ok.outcome == "success"
    assert denied.outcome == "denied"


def test_intersection_denies_when_explicit_is_narrower() -> None:
    """Symmetric: a narrow explicit gate is not widened by a broad ambient one."""
    svc = ToolService(_registry(), tool_authorizer=_authz("reader"))
    with use_tool_authorizer(_authz("anytool")):
        denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert denied.outcome == "denied"


def test_explicit_wins_alone_when_no_ambient_is_set() -> None:
    """With no ambient gate, the explicit arg is the sole gate (today's behaviour)."""
    svc = ToolService(_registry(), tool_authorizer=_authz("mailer"))
    res = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert res.outcome == "success"


# ---- fail-closed when auth is configured -------------------------------------


def test_fail_closed_when_auth_configured_and_no_gate_in_scope() -> None:
    """A path that forgot BOTH the arg and the ambient var DENIES under configured auth."""
    mark_auth_configured(True)
    svc = ToolService(_registry())  # no explicit gate
    # No ambient gate either → fail closed.
    res = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
    assert res.outcome == "denied"
    assert res.error_code is ToolErrorCode.CAPABILITY_DENIED


def test_configured_auth_with_ambient_gate_still_enforces_normally() -> None:
    """Under configured auth, an ambient gate restores normal grant-based enforcement."""
    mark_auth_configured(True)
    svc = ToolService(_registry())
    with use_tool_authorizer(_authz("reader")):
        ok = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
        denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert ok.outcome == "success"
    assert denied.outcome == "denied"


# ---- OFFLINE invariant: byte-unchanged ---------------------------------------


def test_offline_no_auth_no_ambient_is_pure_pass() -> None:
    """No authenticator + no ambient + no explicit → every tool runs (byte-unchanged)."""
    assert not auth_is_configured()
    svc = ToolService(_registry())
    for name in ("weather_get", "send_email"):
        res = run_async(svc.execute(ToolInvocation(tool_name=name)))
        assert res.outcome == "success", name


def test_offline_anonymous_ambient_is_pure_pass() -> None:
    """A NON-enforcing ANONYMOUS ambient gate (offline shape) is an unconditional pass."""
    svc = ToolService(_registry())
    anon = ToolCapabilityAuthorizer.from_principal(ANONYMOUS, _policy())
    assert anon.enforce is False
    with use_tool_authorizer(anon):
        res = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert res.outcome == "success"


# ---- resolver unit-level edge cases ------------------------------------------


def test_resolver_returns_none_offline() -> None:
    """resolve_effective_authorizer is None when nothing is in scope and auth is off."""
    assert resolve_effective_authorizer(None) is None


def test_resolver_returns_deny_all_when_configured_and_empty() -> None:
    """resolve_effective_authorizer is the DENY_ALL sentinel under configured-auth gap."""
    mark_auth_configured(True)
    eff = resolve_effective_authorizer(None)
    assert eff is not None
    assert eff.enforce is True
    assert eff.is_authorized("anything", read_only=True) is False


def test_resolver_identity_when_explicit_equals_ambient() -> None:
    """When explicit IS the ambient object, the resolver returns it directly (no wrap)."""
    authz = _authz("admin")
    with use_tool_authorizer(authz):
        assert resolve_effective_authorizer(authz) is authz
