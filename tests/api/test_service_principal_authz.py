"""P0 #2: the side-doors (inbound connector, scheduled routine) carry a SERVICE identity.

Verifies the least-privilege SERVICE principals are tenant-bound (so their tool-capability
gate engages under a configured authenticator), carry an audit-stampable actor, and that
the connector inbound handler builds an enforcing gate only when auth is configured.
"""

from __future__ import annotations

from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import DEFAULT_POLICY
from himmy.api.auth.service_principal import (
    CONNECTOR_SERVICE_SUBJECT,
    ROUTINE_SERVICE_SUBJECT,
    bind_service_authorizer,
    connector_service_principal,
    routine_service_principal,
)
from himmy.api.connector_inbound import _connector_tool_authorizer
from himmy.services.storage.models import LOCAL_WORKSPACE
from himmy.services.tools.ambient import current_authorizer
from himmy.services.tools.capability import ToolCapabilityAuthorizer


def test_service_principals_are_tenant_bound_and_named() -> None:
    """Each side-door principal is tenant-bound (all_tenants False) + audit-named."""
    conn = connector_service_principal(workspace_id="ws-7")
    assert conn.subject == CONNECTOR_SERVICE_SUBJECT
    assert conn.all_tenants is False
    assert conn.tenant_ids == frozenset({"ws-7"})
    assert "operator" in conn.roles

    rout = routine_service_principal()
    assert rout.subject == ROUTINE_SERVICE_SUBJECT
    assert rout.all_tenants is False
    assert rout.tenant_ids == frozenset({LOCAL_WORKSPACE})


def test_service_actor_metadata_carries_enforce_flag() -> None:
    """A tenant-bound SERVICE actor carries roles + the tool-authz enforce flag for stamping."""
    actor = connector_service_principal(workspace_id="ws-7").actor_metadata()
    assert actor["subject"] == CONNECTOR_SERVICE_SUBJECT
    assert actor["tool_authz_enforce"] is True
    assert "operator" in actor["roles"]


def test_anonymous_actor_omits_enforce_flag() -> None:
    """The ANONYMOUS / all_tenants offline principal omits the enforce flag (no-op gate)."""
    from himmy.api.auth.principal import ANONYMOUS

    assert "tool_authz_enforce" not in ANONYMOUS.actor_metadata()


class _State:
    def __init__(self, authenticator: object, policy: object) -> None:
        self.authenticator = authenticator
        self.access_policy = policy


class _App:
    def __init__(self, authenticator: object, policy: object) -> None:
        self.state = _State(authenticator, policy)


def test_connector_gate_engages_only_with_authenticator() -> None:
    """The connector tool gate is enforcing iff an authenticator is configured.

    Uses the REAL shipped DEFAULT_POLICY (not a fabricated one) so this also pins that
    the default operator role actually carries ``tool:*`` — without it the connector
    service principal would be denied every tool the moment auth is on.
    """
    policy = DEFAULT_POLICY

    # No app / no authenticator → no gate (offline byte-unchanged).
    assert _connector_tool_authorizer(None) is None
    assert _connector_tool_authorizer(_App(authenticator=None, policy=policy)) is None

    # Authenticator configured → an enforcing gate over the connector's roles.
    gate = _connector_tool_authorizer(_App(authenticator=object(), policy=policy))
    assert gate is not None
    assert gate.enforce is True
    # The default operator role holds ``tool:*`` so the connector may invoke any tool.
    assert gate.is_authorized("send_email", False) is True


def test_bind_service_authorizer_enforces_under_configured_auth() -> None:
    """The shared off-request seam binds an ENFORCING ambient gate when a policy is wired.

    Missions + studio_telegram drive ``stream_agent_run`` off any request; this is the
    helper they wrap that drive in. Under a real policy it must put an enforcing
    least-privilege authorizer in the ambient contextvar so the chokepoint gates the run's
    tools to the SERVICE role (rather than wholesale-denying via the fail-closed default).
    """
    principal = connector_service_principal(workspace_id=LOCAL_WORKSPACE)
    assert current_authorizer() is None  # nothing bound outside the block
    with bind_service_authorizer(principal, DEFAULT_POLICY):
        gate = current_authorizer()
        assert gate is not None
        assert gate.enforce is True
        # operator holds tool:* in the default policy, so the service role may invoke tools.
        assert gate.is_authorized("send_email", False) is True
    assert current_authorizer() is None  # reset on exit


def test_bind_service_authorizer_is_inert_offline() -> None:
    """With NO policy wired (offline) the helper binds NOTHING — byte-unchanged pure pass.

    The mission / Telegram drive reads ``active_access_policy()`` which is ``None`` whenever
    no authenticator is configured, so the ambient contextvar stays unset and the offline
    single-box path is untouched.
    """
    principal = connector_service_principal(workspace_id=LOCAL_WORKSPACE)
    with bind_service_authorizer(principal, None):
        assert current_authorizer() is None
    assert current_authorizer() is None


def test_default_policy_operator_authorizes_read_and_write_tools() -> None:
    """REGRESSION: the shipped DEFAULT_POLICY authorizes operator for a read AND a write tool.

    The routine/connector SERVICE principals run as ``operator``; if the default policy
    lost its ``tool:*`` grant, enabling an authenticator would deny every tool to both
    side-doors. Asserting against DEFAULT_POLICY (not a fabricated mapping) keeps this
    honest.
    """
    op = Principal(
        subject="u", roles=frozenset({"operator"}), tenant_ids=frozenset({"ws1"})
    )
    gate = ToolCapabilityAuthorizer.from_principal(op, DEFAULT_POLICY)
    assert gate.is_authorized("weather_get", True) is True
    assert gate.is_authorized("send_email", False) is True
