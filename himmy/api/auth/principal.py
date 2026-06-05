"""The authenticated caller: a Principal carries identity, tenants, and roles.

A :class:`Principal` is the verified answer to "who is calling, and what may they
touch". It is produced by an :class:`~himmy.api.auth.base.Authenticator` (API key,
OIDC token, or client cert) and threaded through the request so routers and the
service layer can enforce tenant isolation (WS1.0), RBAC (WS1.2), and actor
stamping (WS1.3) from a single trusted source instead of trusting client input.

``all_tenants=True`` marks an unrestricted principal — the offline/unconfigured
default (``ANONYMOUS``) and the trusted shared-key boundary — so the framework
stays zero-config and offline-first. A principal bound to specific ``tenant_ids``
(a mapped API key or an OIDC token with a tenant claim) is what actually closes
the cross-tenant (IDOR) hole.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Principal:
    """A verified caller identity with its tenant + role entitlements."""

    subject: str
    tenant_ids: frozenset[str] = frozenset()
    roles: frozenset[str] = frozenset()
    scopes: frozenset[str] = frozenset()
    all_tenants: bool = False
    auth_method: str = "anonymous"
    source_ip: str | None = None
    claims: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        subject: str,
        *,
        tenant_ids: Iterable[str] = (),
        roles: Iterable[str] = (),
        scopes: Iterable[str] = (),
        all_tenants: bool = False,
        auth_method: str = "anonymous",
        source_ip: str | None = None,
        claims: dict[str, Any] | None = None,
    ) -> Principal:
        """Construct a Principal from plain iterables (frozen sets are built here)."""
        return cls(
            subject=subject,
            tenant_ids=frozenset(tenant_ids),
            roles=frozenset(roles),
            scopes=frozenset(scopes),
            all_tenants=all_tenants,
            auth_method=auth_method,
            source_ip=source_ip,
            claims=dict(claims or {}),
        )

    def may_access(self, workspace_id: str) -> bool:
        """Whether this principal is entitled to the given workspace/tenant."""
        return self.all_tenants or workspace_id in self.tenant_ids

    def default_tenant(self) -> str | None:
        """The implied tenant when a request omits one (only if exactly one)."""
        if self.all_tenants or len(self.tenant_ids) != 1:
            return None
        return next(iter(self.tenant_ids))

    def has_role(self, role: str) -> bool:
        """Whether the principal holds ``role``."""
        return role in self.roles

    def actor_metadata(self) -> dict[str, Any]:
        """Compact, log-safe descriptor of the actor (for run/entity stamping)."""
        meta: dict[str, Any] = {
            "subject": self.subject,
            "auth_method": self.auth_method,
        }
        if self.roles:
            meta["roles"] = sorted(self.roles)
        if self.source_ip:
            meta["source_ip"] = self.source_ip
        return meta


#: The unrestricted default principal used when no authenticator is configured
#: (offline-first): unchanged behavior — the caller controls the workspace.
ANONYMOUS = Principal(subject="anonymous", all_tenants=True, auth_method="anonymous")


__all__ = ["Principal", "ANONYMOUS"]
