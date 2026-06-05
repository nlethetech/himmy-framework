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


class ApiKeyAuthenticator:
    """Authenticate a request by a header API key (shared or tenant-mapped)."""

    def __init__(
        self,
        *,
        shared_keys: set[str] | None = None,
        key_principals: dict[str, Principal] | None = None,
        header_name: str = DEFAULT_HEADER,
    ) -> None:
        """Wire shared keys (→ all-tenants) and/or mapped keys (→ bound principals)."""
        self._shared = set(shared_keys or set())
        self._mapped = dict(key_principals or {})
        self._header = header_name
        if not self._shared and not self._mapped:
            raise AuthError("ApiKeyAuthenticator needs at least one configured key")

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
                return Principal.build(
                    subject=f"apikey:{_fingerprint(key)}",
                    all_tenants=True,
                    roles=("admin",),
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
        )
    return out


__all__ = ["ApiKeyAuthenticator", "load_key_principals", "DEFAULT_HEADER"]
