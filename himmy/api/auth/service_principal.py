"""Least-privilege SERVICE principals for the unauthenticated side-doors (P0 #2).

Two paths fire agent runs that are NOT initiated by an authenticated human request:

* an INBOUND connector delivery (a webhook / Slack / Discord trigger), and
* a SCHEDULED routine fire.

Historically these ran with the agent's FULL authority and (for the connector path)
bypassed ``create_run`` entirely — no actor identity, no tenant binding, no RBAC. This
module gives each side-door a dedicated, named SERVICE principal: an explicit
non-human subject, a FIXED workspace, ``all_tenants=False`` (so it is tenant-bound and
its tool-capability gate ENGAGES under a configured authenticator), and a minimal role
set. Its :meth:`~himmy.api.auth.principal.Principal.actor_metadata` is what stamps the
connector/routine identity as the audit actor on every run it launches.

**Offline invariant.** On the single-box ``__local__`` workspace these principals still
carry ``all_tenants=False``, but the run service only builds an enforcing tool-capability
gate when an RBAC policy is wired (i.e. an authenticator is configured). With no
authenticator the actor's ``tool_authz_enforce`` flag is inert — no per-run authorizer is
built — so the offline connector/routine path is byte-unchanged.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

from himmy.api.auth.principal import Principal
from himmy.services.storage.models import LOCAL_WORKSPACE

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from fastapi import Request

    from himmy.api.auth.rbac import AccessPolicy

#: The subject stamped on a run launched by an inbound connector delivery.
CONNECTOR_SERVICE_SUBJECT = "service:connector-inbound"

#: The subject stamped on a run launched by a scheduled routine fire.
ROUTINE_SERVICE_SUBJECT = "service:routine"

#: The least-privilege role both service principals hold by default: ``operator`` reads
#: and writes operational data (the level an autonomous agent run needs) WITHOUT the
#: ``admin`` ``*:*`` super-grant or the audit/consent surfaces. A deployment that wants to
#: further restrict (or widen) a side-door's tool reach ships a custom RBAC role and maps
#: it here via the ``roles`` override.
DEFAULT_SERVICE_ROLES = frozenset({"operator"})


def connector_service_principal(
    *,
    workspace_id: str = LOCAL_WORKSPACE,
    roles: frozenset[str] = DEFAULT_SERVICE_ROLES,
) -> Principal:
    """The SERVICE principal a webhook/Slack/Discord delivery runs under.

    Tenant-bound to ``workspace_id`` (default the offline ``__local__`` box) with
    ``all_tenants=False`` so its tool-capability gate engages under a configured
    authenticator, and named ``service:connector-inbound`` so the run's audit actor
    records WHO acted. ``roles`` is the explicit allow-list of RBAC roles it may exercise.
    """
    return Principal(
        subject=CONNECTOR_SERVICE_SUBJECT,
        tenant_ids=frozenset({workspace_id}),
        roles=frozenset(roles),
        all_tenants=False,
        auth_method="service",
    )


def routine_service_principal(
    *,
    workspace_id: str = LOCAL_WORKSPACE,
    roles: frozenset[str] = DEFAULT_SERVICE_ROLES,
) -> Principal:
    """The SERVICE principal a scheduled routine fire runs under.

    Mirrors :func:`connector_service_principal` for the routine side-door: tenant-bound,
    least-privilege, named ``service:routine`` for audit attribution.
    """
    return Principal(
        subject=ROUTINE_SERVICE_SUBJECT,
        tenant_ids=frozenset({workspace_id}),
        roles=frozenset(roles),
        all_tenants=False,
        auth_method="service",
    )


@contextlib.contextmanager
def bind_service_authorizer(
    principal: Principal,
    policy: AccessPolicy | None,
) -> Iterator[None]:
    """Bind ``principal``'s least-privilege tool-capability gate AMBIENTLY for a run drive.

    centralize-tool-gate: the off-request background surfaces (a scheduled routine fire, a
    background Mission, a Telegram-triggered run) drive an agent via
    ``studio_service.stream_agent_run`` with NO HTTP request in scope, so the request
    -boundary seam cannot reach them. Each must therefore bind the tool gate itself or it
    inherits the chokepoint's process-level FAIL-CLOSED default — which, under a configured
    authenticator, denies EVERY tool (``CAPABILITY_DENIED``) and makes the surface
    non-functional rather than merely safe.

    This wraps that drive so its tools run under ``principal``'s least-privilege SERVICE
    role (deny-by-default to a named service identity) instead of the agent's full
    authority. The contextvar binding propagates into the ``to_thread`` /
    ``create_task`` children the runtime uses (the property the ambient design relies on)
    and is reset on exit.

    **Offline invariant — byte-unchanged.** ``policy`` is the active RBAC ``AccessPolicy``,
    which is ``None`` whenever no authenticator is configured (CLI / single box / offline
    server). With ``policy is None`` :meth:`ToolCapabilityAuthorizer.from_principal` returns
    a NON-enforcing authorizer (or the offline pure-pass), so the ambient binding is inert
    and the historical path is untouched. Were a future caller to forget this wrap entirely,
    the chokepoint's fail-closed default still denies under a configured deployment.
    """
    if policy is None:
        # No RBAC policy wired (offline): nothing to enforce, stay byte-unchanged. We still
        # enter the contextmanager so callers can use it unconditionally.
        yield
        return

    from himmy.services.tools.ambient import _active_authorizer
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    authorizer = ToolCapabilityAuthorizer.from_principal(principal, policy)
    token = _active_authorizer.set(authorizer)
    try:
        yield
    finally:
        _active_authorizer.reset(token)


def require_no_capability_amplification(
    request: Request,
    *,
    resource: str,
    action: str,
    service_roles: frozenset[str] = DEFAULT_SERVICE_ROLES,
) -> None:
    """Reject an arm-and-fire request whose FIRE-TIME authority would EXCEED the caller's.

    The shared confused-deputy / capability-amplification gate for every side-door that
    drives an agent under a FIXED service identity (a scheduled routine fire, a background
    Mission), not under the caller's own gate. Such a run's tools execute under
    ``service_roles`` (default :data:`DEFAULT_SERVICE_ROLES` ⇒ ``operator`` ⇒ ``tool:*``)
    regardless of the launcher's roles. Under the shipped :data:`DEFAULT_RBAC` the only
    holders of the launch permission already hold ``tool:*``, so the service identity never
    exceeds them. But a CUSTOM ``HIMMY_RBAC_FILE`` can decouple the launch permission (e.g.
    ``studio.missions:write`` or ``routine:write``) from broad tool reach — then the run
    would invoke EVERY tool the launcher was never granted (AMPLIFICATION, not attenuation).

    Because the run's tool authority is not creator-derived (no launcher roles are persisted
    on the mission/routine), we close the gap at the arm-and-fire surface itself: a principal
    that does not ALREADY hold every (resource, action) grant the service identity holds may
    not launch (403). The check is computed against the SERVICE identity's actual roles, so
    it stays correct if :data:`DEFAULT_SERVICE_ROLES` is ever retuned.

    Strict NO-OP on the offline path: with no authenticator the principal is the
    unrestricted operator (``all_tenants``), so the gate short-circuits and the zero-config
    launch is byte-unchanged. ``resource`` / ``action`` name the launching surface for the
    audit trail.
    """
    from fastapi import HTTPException

    from himmy.api.auth.context import get_principal
    from himmy.api.auth.rbac import DEFAULT_POLICY
    from himmy.api.security_audit import audit_event

    principal = get_principal(request)
    if principal.all_tenants:
        return  # offline / unrestricted: RBAC inert, byte-unchanged
    policy: Any = getattr(request.app.state, "access_policy", None) or DEFAULT_POLICY
    service_perms: set[tuple[str, str]] = set()
    for role in service_roles:
        service_perms |= set(policy.role_permissions.get(role, frozenset()))
    for svc_resource, svc_action in sorted(service_perms):
        if not policy.authorize(principal, svc_resource, svc_action):
            audit_event(
                request,
                event_type="authz_denied",
                outcome="deny",
                resource=resource,
                action=action,
                detail=(
                    f"{resource} launch denied: caller lacks the "
                    f"{svc_resource}:{svc_action} authority the run would execute with"
                ),
            )
            raise HTTPException(
                status_code=403,
                detail=(
                    f"{resource} denied: it would run with broader authority than "
                    "you hold (capability amplification)"
                ),
            )


__all__ = [
    "CONNECTOR_SERVICE_SUBJECT",
    "ROUTINE_SERVICE_SUBJECT",
    "DEFAULT_SERVICE_ROLES",
    "connector_service_principal",
    "routine_service_principal",
    "bind_service_authorizer",
    "require_no_capability_amplification",
]
