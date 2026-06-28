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

``subject_scoped`` is the OPT-IN object-level (BOLA) narrowing within a shared
tenant (WS-bola): by default a workspace is the trust boundary — its members read
each other's runs/threads (``subject_scoped=False``, the historical behavior, so
every existing principal is byte-unchanged). When a deployment marks a principal
``subject_scoped=True``, the principal may only read resources whose data subject
matches its own ``subject`` — UNLESS it also holds the ``tenant_admin`` role, which
crosses subjects WITHIN its own tenant (the tenant axis is still enforced
separately via :func:`~himmy.api.auth.context.resolve_workspace`). ANONYMOUS /
``all_tenants`` principals are never subject-scoped, so the offline path is unaffected.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

#: The role permitted to cross data subjects WITHIN its own tenant (BOLA). A
#: ``subject_scoped`` principal holding this role reads every subject's runs/threads
#: in the tenant it is bound to (the tenant axis is still enforced by
#: ``resolve_workspace``); without it, a subject-scoped principal sees only its OWN.
TENANT_ADMIN_ROLE = "tenant_admin"


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
    #: Opt-in object-level (BOLA) narrowing: when True, the principal may read only
    #: resources whose data subject matches its own ``subject`` (unless it holds the
    #: ``tenant_admin`` role). Defaults False so a workspace stays the trust boundary and
    #: every existing/ANONYMOUS principal is byte-unchanged. See :meth:`may_access_subject`.
    subject_scoped: bool = False

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
        subject_scoped: bool = False,
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
            subject_scoped=subject_scoped,
        )

    def may_access(self, workspace_id: str) -> bool:
        """Whether this principal is entitled to the given workspace/tenant."""
        return self.all_tenants or workspace_id in self.tenant_ids

    def may_access_subject(self, subject_id: str | None) -> bool:
        """Whether this principal may read a resource owned by ``subject_id`` (BOLA).

        The object-level (BOLA) counterpart of :meth:`may_access` (which is the
        tenant/workspace axis). Returns True — i.e. NO narrowing — for every
        principal except an opt-in ``subject_scoped`` one, so a workspace stays the
        trust boundary by default and the offline / ``all_tenants`` / existing path
        is byte-unchanged:

        * ``all_tenants`` (ANONYMOUS / trusted shared key) → always True;
        * not ``subject_scoped`` (the historical multi-user-workspace default) → True;
        * holds the ``tenant_admin`` role → True (crosses subjects within its tenant —
          the tenant axis is enforced separately by ``resolve_workspace``);
        * a resource with no recorded subject (``subject_id is None``, legacy / pre-
          subject-binding data) → True (no subject to narrow on);
        * otherwise the resource's ``subject_id`` must equal this principal's ``subject``.
        """
        if self.all_tenants or not self.subject_scoped:
            return True
        if TENANT_ADMIN_ROLE in self.roles:
            return True
        if subject_id is None:
            return True
        return subject_id == self.subject

    def default_tenant(self) -> str | None:
        """The implied tenant when a request omits one (only if exactly one)."""
        if self.all_tenants or len(self.tenant_ids) != 1:
            return None
        return next(iter(self.tenant_ids))

    def has_role(self, role: str) -> bool:
        """Whether the principal holds ``role``."""
        return role in self.roles

    def actor_metadata(self) -> dict[str, Any]:
        """Compact, log-safe descriptor of the actor (for run/entity stamping).

        Carries the actor's ``roles`` (and ``scopes``, when present) and — for a
        tenant-bound principal — a ``tool_authz_enforce`` flag (P0). The flag is what
        lets a run's tool-capability gate be REBUILT from the persisted descriptor on the
        leased-dispatch recovery path (a fresh process has no live :class:`Principal`);
        ``scopes`` is serialized so that rebuilt gate applies the SAME delegated
        scope-narrowing the live gate did (a read-only-scoped token cannot regain write
        tools across a process boundary). An unrestricted / ANONYMOUS principal
        (``all_tenants``) that carries NO scopes omits the enforce flag, so the rebuilt
        authorizer is a non-enforcing pass-through and the offline path is byte-unchanged.

        A SCOPE-NARROWED ``all_tenants`` token (e.g. an admin-role OIDC/CI token minted with
        narrowing ``scopes`` but no tool scope) DOES carry the enforce flag: like
        :meth:`ToolCapabilityAuthorizer.from_principal`, its tool reach is narrowed by those
        scopes on the live path, so the rebuilt gate must enforce too — otherwise the
        leased-dispatch recovery path would re-open the attenuation escape across a process
        boundary. Only a truly scope-LESS all_tenants principal stays a pass-through.
        """
        meta: dict[str, Any] = {
            "subject": self.subject,
            "auth_method": self.auth_method,
        }
        if self.roles:
            meta["roles"] = sorted(self.roles)
        if self.scopes:
            meta["scopes"] = sorted(self.scopes)
        if self.source_ip:
            meta["source_ip"] = self.source_ip
        if not self.all_tenants or self.scopes:
            meta["tool_authz_enforce"] = True
        return meta


#: The unrestricted default principal used when no authenticator is configured
#: (offline-first): unchanged behavior — the caller controls the workspace.
ANONYMOUS = Principal(subject="anonymous", all_tenants=True, auth_method="anonymous")


__all__ = ["Principal", "ANONYMOUS", "TENANT_ADMIN_ROLE"]
