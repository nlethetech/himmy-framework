"""Tests for P0 tool capability authorization (the confused-deputy fix).

Covers the four guarantees the work package introduces:

* a principal LACKING a tool's capability is DENIED that tool at runtime;
* an ANONYMOUS / all_tenants (offline) principal still runs EVERY tool (no-op bypass);
* the read/write intent split — a write tool needs the extra ``tool:<name>:write`` grant;
* a spawned sub-agent inherits the parent's gate verbatim (attenuate, never amplify).
"""

from __future__ import annotations

from himmy.api.auth.principal import ANONYMOUS, Principal
from himmy.api.auth.rbac import DEFAULT_POLICY, AccessPolicy
from himmy.services.tools import (
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_local_tool,
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
        registry, name="weather_get", handler=weather, description="reads weather"
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


def _bound(principal: Principal) -> ToolService:
    authz = ToolCapabilityAuthorizer.from_principal(principal, _policy())
    return ToolService(_registry(), tool_authorizer=authz)


def test_anonymous_offline_runs_every_tool() -> None:
    """ANONYMOUS (all_tenants) bypasses the gate entirely — every tool allowed (no-op)."""
    svc = _bound(ANONYMOUS)
    for name in ("weather_get", "send_email"):
        res = run_async(svc.execute(ToolInvocation(tool_name=name)))
        assert res.outcome == "success", name


def test_no_authorizer_is_byte_unchanged() -> None:
    """With no authorizer wired at all the tool service behaves exactly as before."""
    svc = ToolService(_registry())  # no tool_authorizer
    res = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert res.outcome == "success"


def test_principal_without_capability_is_denied() -> None:
    """A tenant-bound principal lacking a tool's capability is DENIED at runtime."""
    reader = Principal(
        subject="u", roles=frozenset({"reader"}), tenant_ids=frozenset({"ws1"})
    )
    svc = _bound(reader)
    # Granted the read tool.
    ok = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
    assert ok.outcome == "success"
    # Denied the write tool it has no grant for — deny-by-default.
    denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert denied.outcome == "denied"
    assert denied.error_code is ToolErrorCode.POLICY_BLOCKED


def test_invoke_without_write_grant_denies_write_tool() -> None:
    """An invoke-only grant is NOT enough to call a side-effecting (write) tool."""
    # A role granting only invoke on the write tool, but not the write capability.
    policy = AccessPolicy.from_mapping({"halfgrant": ["tool:send_email:invoke"]})
    p = Principal(
        subject="u", roles=frozenset({"halfgrant"}), tenant_ids=frozenset({"ws1"})
    )
    svc = ToolService(
        _registry(), tool_authorizer=ToolCapabilityAuthorizer.from_principal(p, policy)
    )
    denied = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert denied.outcome == "denied"
    assert denied.error_code is ToolErrorCode.POLICY_BLOCKED


def test_full_grant_allows_write_tool() -> None:
    """A role with both invoke + write may call the side-effecting tool."""
    mailer = Principal(
        subject="u", roles=frozenset({"mailer"}), tenant_ids=frozenset({"ws1"})
    )
    svc = _bound(mailer)
    ok = run_async(svc.execute(ToolInvocation(tool_name="send_email")))
    assert ok.outcome == "success"


def test_admin_and_wildcard_allow_all() -> None:
    """``admin`` (*:*) and ``tool:*`` both reach every tool (read AND write)."""
    for role in ("admin", "anytool"):
        p = Principal(
            subject="u", roles=frozenset({role}), tenant_ids=frozenset({"ws1"})
        )
        svc = _bound(p)
        for name in ("weather_get", "send_email"):
            res = run_async(svc.execute(ToolInvocation(tool_name=name)))
            assert res.outcome == "success", f"{role}:{name}"


def test_enforcing_authorizer_without_policy_fails_closed() -> None:
    """An enforcing authorizer with no policy denies everything (fail CLOSED)."""
    authz = ToolCapabilityAuthorizer(enforce=True, roles=frozenset({"reader"}), policy=None)
    svc = ToolService(_registry(), tool_authorizer=authz)
    denied = run_async(svc.execute(ToolInvocation(tool_name="weather_get")))
    assert denied.outcome == "denied"


def test_attenuate_returns_subset_gate() -> None:
    """A sub-agent's inherited gate cannot exceed the parent's (attenuate, never amplify)."""
    reader = Principal(
        subject="u", roles=frozenset({"reader"}), tenant_ids=frozenset({"ws1"})
    )
    parent = ToolCapabilityAuthorizer.from_principal(reader, _policy())
    child = parent.attenuate()
    # The child grants exactly what the parent grants — same roles, same enforcement.
    assert child.enforce is True
    assert child.roles == parent.roles
    assert child.is_authorized("weather_get", True) is True
    assert child.is_authorized("send_email", False) is False


def test_default_policy_operator_holds_tool_capability() -> None:
    """REGRESSION: the SHIPPED DEFAULT_POLICY must authorize operator for read+write tools.

    The routine/connector SERVICE principals AND operator humans run as ``operator``; if
    the default policy carried no ``tool:*`` grant, enabling an authenticator would deny
    EVERY tool to all of them. This pins the real policy (not a fabricated one) so the
    regression can never silently return.
    """
    op = Principal(
        subject="u", roles=frozenset({"operator"}), tenant_ids=frozenset({"ws1"})
    )
    authz = ToolCapabilityAuthorizer.from_principal(op, DEFAULT_POLICY)
    assert authz.enforce is True
    # A read tool and a (side-effecting) write tool are BOTH authorized for operator.
    assert authz.is_authorized("weather_get", True) is True
    assert authz.is_authorized("send_email", False) is True


def test_ambiguously_named_writer_requires_write_grant() -> None:
    """An ambiguously-named tool fails CLOSED to writer: invoke-only is NOT enough.

    ``process_payment`` has no ``read_only`` flag and a name ``classify_read_only`` cannot
    classify (-> None). An invoke-only grant must NOT reach it — it additionally requires
    ``tool:process_payment:write`` — so an operator granting a per-tool ``:invoke`` for a
    read-only reach cannot unknowingly hand write reach to an ambiguously-named writer.
    """
    invoke_only = AccessPolicy.from_mapping(
        {"halfgrant": ["tool:process_payment:invoke"]}
    )
    p = Principal(
        subject="u", roles=frozenset({"halfgrant"}), tenant_ids=frozenset({"ws1"})
    )
    gate = ToolCapabilityAuthorizer.from_principal(p, invoke_only)
    # read_only=None (no flag) + ambiguous name → treated as a writer → DENIED on invoke-only.
    assert gate.is_authorized("process_payment", None) is False
    # Granting the write capability too lets it through.
    full = AccessPolicy.from_mapping(
        {"full": ["tool:process_payment:invoke", "tool:process_payment:write"]}
    )
    pf = Principal(
        subject="u", roles=frozenset({"full"}), tenant_ids=frozenset({"ws1"})
    )
    assert (
        ToolCapabilityAuthorizer.from_principal(pf, full).is_authorized(
            "process_payment", None
        )
        is True
    )


def test_from_actor_rebuilds_enforcing_gate() -> None:
    """A persisted actor descriptor rebuilds the enforcing gate (dispatch-recovery path)."""
    actor = {"subject": "u", "roles": ["reader"], "tool_authz_enforce": True}
    authz = ToolCapabilityAuthorizer.from_actor(actor, _policy())
    assert authz.enforce is True
    assert authz.is_authorized("weather_get", True) is True
    assert authz.is_authorized("send_email", False) is False
    # An actor without the enforce flag (offline / all_tenants) is a pass-through.
    offline = ToolCapabilityAuthorizer.from_actor({"subject": "anon"}, _policy())
    assert offline.enforce is False
    assert offline.is_authorized("send_email", False) is True
