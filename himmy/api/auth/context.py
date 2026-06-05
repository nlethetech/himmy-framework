"""Request-time identity: build the authenticator, attach the Principal, scope tenants.

This wires the auth seam into FastAPI:

* :func:`build_authenticator` selects an :class:`Authenticator` from environment
  (mapped/shared API keys today; OIDC plugs in here). ``None`` ⇒ no auth ⇒ the
  offline-first default where every request is :data:`ANONYMOUS` (all tenants).
* :func:`principal_dependency` authenticates each request and stashes the
  :class:`Principal` on ``request.state`` (401 on failure).
* :func:`resolve_workspace` is the **single tenant-isolation choke point** (WS1.0):
  the effective ``workspace_id`` comes from the verified principal, never blindly
  from client input — closing the cross-tenant (IDOR) hole.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from fastapi import HTTPException, Request

from himmy.api.auth.base import AuthError
from himmy.api.auth.principal import ANONYMOUS, Principal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.api.auth.base import Authenticator


def build_authenticator() -> Authenticator | None:
    """Select an authenticator from env, or ``None`` for the offline (no-auth) default.

    Precedence: OIDC (``HIMMY_AUTH_MODE=oidc``) ▸ API keys (mapped file and/or shared
    ``HIMMY_INTERNAL_API_KEY``) ▸ none. Mapped keys (``HIMMY_API_KEYS_FILE``) and a
    shared key can coexist.
    """
    mode = os.environ.get("HIMMY_AUTH_MODE", "").lower()
    if mode == "oidc":  # pragma: no cover - lands in the OIDC increment
        raise AuthError(
            "HIMMY_AUTH_MODE=oidc is not available yet; use API-key auth for now"
        )

    from himmy.api.auth.apikey import (
        DEFAULT_HEADER,
        ApiKeyAuthenticator,
        load_key_principals,
    )

    header = os.environ.get("HIMMY_INTERNAL_HEADER", DEFAULT_HEADER)
    shared = {
        k.strip()
        for k in os.environ.get("HIMMY_INTERNAL_API_KEY", "").split(",")
        if k.strip()
    }
    mapped = {}
    keys_file = os.environ.get("HIMMY_API_KEYS_FILE")
    if keys_file:
        mapped = load_key_principals(keys_file)
    if shared or mapped:
        return ApiKeyAuthenticator(
            shared_keys=shared, key_principals=mapped, header_name=header
        )
    return None


async def principal_dependency(request: Request) -> None:
    """Authenticate the request and attach the Principal to ``request.state`` (401)."""
    authenticator = getattr(request.app.state, "authenticator", None)
    if authenticator is None:
        request.state.principal = ANONYMOUS
        return
    try:
        principal = await authenticator.authenticate(request)
    except AuthError as exc:
        from himmy.api.security_audit import audit_event

        audit_event(
            request, event_type="auth_failure", outcome="deny", detail=exc.detail
        )
        raise HTTPException(
            status_code=401,
            detail=exc.detail,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
    request.state.principal = principal


def get_principal(request: Request) -> Principal:
    """Return the Principal attached to the request (ANONYMOUS if unset)."""
    return getattr(request.state, "principal", ANONYMOUS)


def resolve_workspace(request: Request, requested: str | None) -> str | None:
    """Return the effective, authorized ``workspace_id`` for this request (WS1.0).

    * An unrestricted principal (offline default / trusted shared key) keeps the
      legacy behavior — the caller-supplied value is used as-is.
    * A tenant-bound principal may only use a workspace it is entitled to: a
      mismatch is **403**, and omitting the workspace is allowed only when the
      principal has exactly one tenant (otherwise **400**).
    """
    principal = get_principal(request)
    if principal.all_tenants:
        return requested
    if requested is None:
        default = principal.default_tenant()
        if default is not None:
            return default
        raise HTTPException(
            status_code=400,
            detail="workspace_id is required for a multi-tenant principal",
        )
    if not principal.may_access(requested):
        raise HTTPException(status_code=403, detail="workspace access denied")
    return requested


def require_workspace(request: Request, requested: str) -> str:
    """Like :func:`resolve_workspace` for write paths that always carry a workspace.

    The request supplies a concrete ``workspace_id`` (e.g. in a create body); this
    authorizes it against the principal and returns it (never ``None``).
    """
    resolved = resolve_workspace(request, requested)
    if resolved is None:  # pragma: no cover - requested is non-None, so this can't fire
        raise HTTPException(status_code=400, detail="workspace_id is required")
    return resolved


__all__ = [
    "build_authenticator",
    "principal_dependency",
    "get_principal",
    "resolve_workspace",
    "require_workspace",
]
