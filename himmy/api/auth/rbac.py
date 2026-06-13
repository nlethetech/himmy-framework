"""Role-based access control: roles → permissions over (resource, action) pairs.

A :class:`AccessPolicy` maps each role to a set of ``resource:action`` permissions
(``*`` wildcards allowed). The :func:`require_permission` dependency guards a route:
the authenticated principal must hold a role that grants the route's permission, else
403. Permissions are **data** (a JSON policy file via ``HIMMY_RBAC_FILE``), so an
operator can customize roles without code.

Offline-first is preserved: when no authenticator is configured, RBAC is bypassed
(the zero-config path is unchanged). Enforcement only kicks in once auth is on — and a
principal with no matching role is denied by default (deny-by-default).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from himmy.api.auth.context import get_principal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

    from himmy.api.auth.principal import Principal

#: The built-in role catalogue. ``admin`` is unrestricted; ``viewer`` reads
#: operational data; ``operator`` reads + writes it; ``auditor`` additionally reads the
#: audit surface AND runs the WS4.7 privacy audit (``audit:run``). ``data_subject`` is a
#: self-scoped role for a person exercising
#: their own consent/erasure rights (the router additionally restricts it to its own
#: ``subject_id``). Operators ship their own via ``HIMMY_RBAC_FILE``.
DEFAULT_RBAC: dict[str, list[str]] = {
    "viewer": [
        "run:read",
        "recommendation:read",
        "context:read",
        "dashboard:read",
        "evaluation:read",
        "connector:read",
    ],
    "operator": [
        "run:read",
        "run:write",
        "recommendation:read",
        "recommendation:write",
        "context:read",
        "context:write",
        "dashboard:read",
        "evaluation:read",
        "evaluation:write",
        "consent:read",
        "consent:write",
        "connector:read",
    ],
    "auditor": [
        "run:read",
        "recommendation:read",
        "context:read",
        "dashboard:read",
        "evaluation:read",
        "audit:read",
        "audit:run",
        "consent:read",
        "connector:read",
    ],
    # Self-scoped: holds only consent:read (so it can read its own decision/history and
    # exercise withdrawal/erasure). The /v1/consent router enforces it may touch ONLY its
    # own subject_id; it has no operational/write reach.
    "data_subject": [
        "consent:read",
    ],
    "admin": ["*:*"],
}


def _parse_perm(spec: str) -> tuple[str, str]:
    """Parse a ``"resource:action"`` permission string."""
    resource, _, action = spec.partition(":")
    return resource.strip() or "*", (action.strip() or "*")


@dataclass(frozen=True)
class AccessPolicy:
    """An immutable role → permissions map with wildcard-aware authorization."""

    role_permissions: dict[str, frozenset[tuple[str, str]]]

    @classmethod
    def from_mapping(cls, mapping: dict[str, list[str]]) -> AccessPolicy:
        """Build a policy from ``{role: ["resource:action", ...]}``."""
        return cls(
            {
                str(role): frozenset(_parse_perm(p) for p in perms)
                for role, perms in mapping.items()
            }
        )

    def authorize(self, principal: Principal, resource: str, action: str) -> bool:
        """Whether any of the principal's roles grants ``(resource, action)``."""
        for role in principal.roles:
            perms = self.role_permissions.get(role)
            if perms and _covers(perms, resource, action):
                return True
        return False


def _covers(perms: frozenset[tuple[str, str]], resource: str, action: str) -> bool:
    """Whether ``perms`` grants ``(resource, action)`` (``*`` wildcards match)."""
    return any(r in (resource, "*") and a in (action, "*") for (r, a) in perms)


#: The default policy (used when no ``HIMMY_RBAC_FILE`` is configured).
DEFAULT_POLICY = AccessPolicy.from_mapping(DEFAULT_RBAC)


def load_policy(path: str | Path) -> AccessPolicy:
    """Load an :class:`AccessPolicy` from a JSON ``{role: [perm, ...]}`` file."""
    raw = json.loads(Path(path).expanduser().read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"RBAC file {path} must be a JSON object")
    return AccessPolicy.from_mapping(raw)


def build_access_policy() -> AccessPolicy:
    """Select the policy from env (``HIMMY_RBAC_FILE``) or the built-in default."""
    import os

    path = os.environ.get("HIMMY_RBAC_FILE")
    return load_policy(path) if path else DEFAULT_POLICY


def require_permission(
    resource: str, action: str
) -> Callable[[Request], Awaitable[None]]:
    """A route dependency enforcing ``resource:action`` for the request's principal.

    Bypassed when no authenticator is configured (offline-first). Otherwise the
    principal must hold a role granting the permission, else 403.
    """

    async def _dep(request: Request) -> None:
        if getattr(request.app.state, "authenticator", None) is None:
            return  # no auth configured → RBAC off (offline-first)
        policy: AccessPolicy = (
            getattr(request.app.state, "access_policy", None) or DEFAULT_POLICY
        )
        if not policy.authorize(get_principal(request), resource, action):
            from himmy.api.security_audit import audit_event

            audit_event(
                request,
                event_type="authz_denied",
                outcome="deny",
                resource=resource,
                action=action,
                workspace_id=request.query_params.get("workspace_id"),
                detail=f"permission denied: {resource}:{action}",
            )
            raise HTTPException(
                status_code=403,
                detail=f"permission denied: {resource}:{action}",
            )

    return _dep


__all__ = [
    "AccessPolicy",
    "DEFAULT_RBAC",
    "DEFAULT_POLICY",
    "load_policy",
    "build_access_policy",
    "require_permission",
]
