"""API-key authentication: a header key → a Principal (constant-time, rotatable).

Two modes, both timing-safe:

* **Shared trusted-boundary key** (back-compat): one or more keys in
  ``HIMMY_INTERNAL_API_KEY`` (comma-separated for rotation) each map to an
  unrestricted, all-tenants principal — the existing "internal trusted boundary"
  behavior. It does NOT bind a caller to a tenant.
* **Mapped keys** (closes the cross-tenant hole): a JSON map of key → tenants/roles
  (``HIMMY_API_KEYS_FILE``) yields a Principal *bound* to specific tenants, so a key
  scoped to tenant A cannot read tenant B.
"""

from __future__ import annotations

import hmac
import json
from pathlib import Path
from typing import TYPE_CHECKING

from himmy.api.auth.base import AuthError, client_ip
from himmy.api.auth.principal import Principal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request

#: Default header carrying the API key.
DEFAULT_HEADER = "x-himmy-internal-key"


#: Default roles a DEMOTED shared key receives under the multi-tenant posture (G1).
#: ``operator`` reads + writes the operational surface (run/agent/knowledge/model/…)
#: but holds NO tenant_ids and is ``all_tenants=False``, so ``resolve_workspace``
#: 403s it off any tenant's data — it is useful for ops/diagnostics, not a tenant
#: super-user. (Verified against ``DEFAULT_RBAC``: ``operator`` grants run/agent/
#: knowledge/model/diagnostics read+write, so the demoted key is NOT 403-everywhere.)
DEMOTED_SHARED_KEY_ROLES: tuple[str, ...] = ("operator",)


class ApiKeyAuthenticator:
    """Authenticate a request by a header API key (shared or tenant-mapped)."""

    def __init__(
        self,
        *,
        shared_keys: set[str] | None = None,
        key_principals: dict[str, Principal] | None = None,
        header_name: str = DEFAULT_HEADER,
        shared_key_roles: tuple[str, ...] | None = None,
    ) -> None:
        """Wire shared keys (→ all-tenants) and/or mapped keys (→ bound principals).

        ``shared_key_roles`` controls the posture of a *shared* (unmapped) key match:

        * ``None`` (default) keeps the historical single-box behavior — a shared key
          maps to an unrestricted ``all_tenants`` ``admin`` principal (the "internal
          trusted boundary").
        * A concrete role tuple (e.g. :data:`DEMOTED_SHARED_KEY_ROLES`) DEMOTES the
          shared key to a tenant-bound-by-absence principal: ``all_tenants=False``,
          no ``tenant_ids``, and exactly those roles. The multi-tenant fail-closed
          posture (G2) passes this so a shared-key match can no longer act as a
          cross-tenant admin, while remaining useful for ops/diagnostics routes.

        Mapped keys (``key_principals``) are unaffected by this — they always bind to
        their declared tenants.
        """
        self._shared = set(shared_keys or set())
        self._mapped = dict(key_principals or {})
        self._header = header_name
        self._shared_key_roles = shared_key_roles
        if not self._shared and not self._mapped:
            raise AuthError("ApiKeyAuthenticator needs at least one configured key")

    @property
    def binds_tenants(self) -> bool:
        """Whether this authenticator binds callers to concrete tenants (G1).

        True iff at least one tenant-mapped key is configured — those principals
        carry ``tenant_ids`` and close the cross-tenant hole. A shared-key-ONLY
        authenticator binds nobody (every shared match is all-tenants or, when
        demoted, tenant-LESS), so it returns ``False`` and the multi-tenant posture
        (G2) refuses to start on it.
        """
        return bool(self._mapped)

    def openapi_security_scheme(self) -> dict[str, dict[str, object]]:
        """Advertise the API-key header as an OpenAPI security scheme (for docs)."""
        return {
            "himmyApiKey": {
                "type": "apiKey",
                "in": "header",
                "name": self._header,
            }
        }

    async def authenticate(self, request: Request) -> Principal:
        """Resolve the key header to a Principal (raises AuthError if invalid)."""
        provided = request.headers.get(self._header)
        if not provided:
            raise AuthError("missing api key")
        ip = client_ip(request)
        # Mapped keys first (a key bound to tenants is more specific than a shared one).
        for key, principal in self._mapped.items():
            if hmac.compare_digest(provided, key):
                return _with_ip(principal, ip)
        for key in self._shared:
            if hmac.compare_digest(provided, key):
                # Default (shared_key_roles is None): the historical unrestricted
                # all-tenants admin. Demoted (a role tuple supplied, e.g. under the
                # G2 multi-tenant posture): NO tenant_ids, all_tenants=False, only the
                # given roles — so resolve_workspace 403s it off any tenant's data.
                roles = self._shared_key_roles
                demoted = roles is not None
                return Principal.build(
                    subject=f"apikey:{_fingerprint(key)}",
                    all_tenants=not demoted,
                    roles=roles if roles is not None else ("admin",),
                    auth_method="apikey",
                    source_ip=ip,
                )
        raise AuthError("invalid api key")


def _fingerprint(key: str) -> str:
    """A short, non-reversible tag for a key (for the subject id / logs)."""
    import hashlib

    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _with_ip(principal: Principal, ip: str | None) -> Principal:
    """Return ``principal`` stamped with the request source IP."""
    if ip is None or principal.source_ip == ip:
        return principal
    return Principal(
        subject=principal.subject,
        tenant_ids=principal.tenant_ids,
        roles=principal.roles,
        scopes=principal.scopes,
        all_tenants=principal.all_tenants,
        auth_method=principal.auth_method,
        source_ip=ip,
        claims=principal.claims,
        subject_scoped=principal.subject_scoped,
    )


def load_key_principals(path: str | Path) -> dict[str, Principal]:
    """Load a key→Principal map from a JSON file.

    Schema: ``{"<key>": {"subject": str, "tenant_ids": [..], "roles": [..]}}``. Each
    entry yields a Principal bound to those tenants (auth_method ``apikey``).
    """
    raw = json.loads(Path(path).expanduser().read_text())
    if not isinstance(raw, dict):
        raise AuthError(f"api-keys file {path} must be a JSON object")
    out: dict[str, Principal] = {}
    for key, spec in raw.items():
        spec = spec or {}
        out[str(key)] = Principal.build(
            subject=str(spec.get("subject") or f"apikey:{_fingerprint(str(key))}"),
            tenant_ids=[str(t) for t in (spec.get("tenant_ids") or [])],
            roles=[str(r) for r in (spec.get("roles") or [])],
            scopes=[str(s) for s in (spec.get("scopes") or [])],
            all_tenants=bool(spec.get("all_tenants", False)),
            auth_method="apikey",
            subject_scoped=bool(spec.get("subject_scoped", False)),
        )
    return out


__all__ = [
    "ApiKeyAuthenticator",
    "load_key_principals",
    "DEFAULT_HEADER",
    "DEMOTED_SHARED_KEY_ROLES",
]
