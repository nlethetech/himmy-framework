"""The authentication seam: resolve a request into a verified Principal.

An :class:`Authenticator` turns transport credentials (an API-key header, an OIDC
Bearer token, a client certificate) into a :class:`Principal`, or raises
:class:`AuthError` (surfaced as 401). This is the inference ``ClientManager``-style
seam for identity: the BFF picks an implementation from config and the rest of the
stack only ever sees a Principal.
"""

from __future__ import annotations

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

    async def authenticate(self, request: Request) -> Principal:
        """Return the verified principal, or raise :class:`AuthError` on failure."""
        ...


def client_ip(request: Request) -> str | None:
    """Best-effort source IP for audit (honors a single proxy ``X-Forwarded-For``)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", None) if client else None


__all__ = ["Authenticator", "AuthError", "client_ip"]
