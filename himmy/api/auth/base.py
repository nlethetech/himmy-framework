"""The authentication seam: resolve a request into a verified Principal.

An :class:`Authenticator` turns transport credentials (an API-key header, an OIDC
Bearer token, a client certificate) into a :class:`Principal`, or raises
:class:`AuthError` (surfaced as 401). This is the inference ``ClientManager``-style
seam for identity: the BFF picks an implementation from config and the rest of the
stack only ever sees a Principal.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from himmy.core.errors import HimmyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request

    from himmy.api.auth.principal import Principal


class AuthError(HimmyError):
    """Authentication failed (missing/invalid credentials) → HTTP 401."""

    def __init__(self, detail: str = "authentication required") -> None:
        super().__init__(detail)
        self.detail = detail


@runtime_checkable
class Authenticator(Protocol):
    """Resolves a request into a verified :class:`Principal` (or raises AuthError)."""

    @property
    def binds_tenants(self) -> bool:
        """Whether this authenticator binds callers to specific tenants (G1).

        ``True`` means a verified principal carries concrete ``tenant_ids`` (OIDC, or
        tenant-mapped API keys) — the signal the multi-tenant fail-closed posture (G2)
        keys on to refuse the two all-tenants-admin footguns (a shared-key-only deploy
        that mints every caller an unrestricted admin, or an ANONYMOUS all-tenants
        default). A shared-key-ONLY API-key authenticator binds nobody (``False``).

        Declared read-only so a computed ``@property`` implementation satisfies the
        Protocol. The posture check reads it via ``getattr(authenticator,
        "binds_tenants", False)`` so a custom/legacy authenticator lacking the member
        fails CLOSED (rejected), never ``AttributeError``.
        """
        ...

    async def authenticate(self, request: Request) -> Principal:
        """Return the verified principal, or raise :class:`AuthError` on failure."""
        ...


def _trusted_proxies() -> frozenset[str]:
    """The peer IPs allowed to set ``X-Forwarded-For`` (from ``HIMMY_TRUSTED_PROXIES``)."""
    return frozenset(
        p.strip()
        for p in (os.environ.get("HIMMY_TRUSTED_PROXIES") or "").split(",")
        if p.strip()
    )


def peer_ip(request: Request) -> str | None:
    """The real transport peer IP (``request.client.host``), ignoring any header."""
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client else None


def client_ip(request: Request) -> str | None:
    """Source IP for audit + rate-limit keying.

    Honors ``X-Forwarded-For`` **only** when the immediate peer is a configured
    trusted proxy (``HIMMY_TRUSTED_PROXIES``); otherwise the header is attacker-
    controllable and is ignored in favour of the real ``request.client.host``. This
    closes the rate-limit-bypass / audit-IP-spoofing hole where any client could
    forge a fresh source by rotating the header.
    """
    peer = peer_ip(request)
    trusted = _trusted_proxies()
    if peer in trusted and trusted:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            first = forwarded.split(",")[0].strip()
            if first:
                return first
    return peer


__all__ = ["Authenticator", "AuthError", "client_ip", "peer_ip"]
