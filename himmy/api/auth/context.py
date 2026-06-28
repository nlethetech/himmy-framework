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

# fastapi is the [api] extra; this module must still IMPORT on a core (offline) install — its
# request-time helpers below are only ever called when the API server is running.
try:
    from fastapi import HTTPException, Request
except ModuleNotFoundError:  # pragma: no cover - exercised only when the API server runs
    HTTPException = Request = None  # type: ignore[assignment, misc]

from himmy.api.auth.base import AuthError
from himmy.api.auth.principal import ANONYMOUS, Principal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.api.auth.base import Authenticator


def is_multi_tenant() -> bool:
    """Whether the deployment has declared a multi-tenant fail-closed posture (G2).

    The single source of truth both :func:`build_authenticator` (to DEMOTE a shared
    key) and the startup posture check (to REFUSE a non-tenant-binding authenticator)
    read, so the two never disagree. Truthy when ``HIMMY_MULTI_TENANT`` is set OR an
    explicit ``HIMMY_AUTH_MODE`` (anything but empty/``none``) is configured — i.e.
    ANY non-empty auth mode (incl. the ``apikey`` example in values.yaml) engages
    strictness. Default (no env) is single-box, byte-unchanged.
    """
    if os.environ.get("HIMMY_MULTI_TENANT", "").lower() in ("1", "true", "yes"):
        return True
    return os.environ.get("HIMMY_AUTH_MODE", "").lower() not in ("", "none")


def build_authenticator() -> Authenticator | None:
    """Select an authenticator from env, or ``None`` for the offline (no-auth) default.

    Precedence: OIDC (``HIMMY_AUTH_MODE=oidc``) ▸ API keys (mapped file and/or shared
    ``HIMMY_INTERNAL_API_KEY``) ▸ none. Mapped keys (``HIMMY_API_KEYS_FILE``) and a
    shared key can coexist.

    Under the multi-tenant posture (:func:`is_multi_tenant`) a *shared* key is built
    DEMOTED to operator-only (no tenant reach) so a shared-key match can no longer act
    as a cross-tenant admin (G1); a shared-key-ONLY deploy is then additionally refused
    at startup because it still binds nobody.
    """
    mode = os.environ.get("HIMMY_AUTH_MODE", "").lower()
    if mode == "oidc":
        from himmy.api.auth.oidc import OidcAuthenticator

        return OidcAuthenticator.from_env()

    from himmy.api.auth.apikey import (
        DEFAULT_HEADER,
        DEMOTED_SHARED_KEY_ROLES,
        ApiKeyAuthenticator,
        load_key_principals,
    )
    from himmy.config.secrets import get_secret

    header = os.environ.get("HIMMY_INTERNAL_HEADER", DEFAULT_HEADER)
    shared = {
        k.strip()
        for k in (get_secret("HIMMY_INTERNAL_API_KEY") or "").split(",")
        if k.strip()
    }
    mapped = {}
    keys_file = os.environ.get("HIMMY_API_KEYS_FILE")
    if keys_file:
        mapped = load_key_principals(keys_file)
    if shared or mapped:
        return ApiKeyAuthenticator(
            shared_keys=shared,
            key_principals=mapped,
            header_name=header,
            shared_key_roles=(
                DEMOTED_SHARED_KEY_ROLES if is_multi_tenant() else None
            ),
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


def authorize_object(request: Request, subject_id: str | None) -> bool:
    """Object-level (BOLA) gate: may this request's principal read ``subject_id``?

    The subject-axis companion to :func:`resolve_workspace` (the tenant axis). Returns
    the principal's :meth:`~himmy.api.auth.principal.Principal.may_access_subject`
    verdict, which is **True for every principal except an opt-in ``subject_scoped``
    one** — so the offline / ``all_tenants`` / historical multi-user-workspace path is a
    NO-OP and byte-unchanged. A ``subject_scoped`` principal is True only for its OWN
    ``subject`` (or any subject when it holds ``tenant_admin``, or a subject-less legacy
    resource).

    Callers fold a False verdict into a **404** (not 403) so a subject-scoped tenant
    cannot even probe the existence of another subject's object — mirroring the
    not-found-on-cross-tenant convention of :func:`get_run` / :func:`load_owned_thread`.
    """
    return get_principal(request).may_access_subject(subject_id)


def narrow_subject(request: Request, requested: str | None) -> str | None:
    """Pin a LIST query to the caller's own subject when it is ``subject_scoped`` (BOLA).

    The list-path companion to :func:`authorize_object` (the by-id gate). For a
    ``subject_scoped`` principal WITHOUT the ``tenant_admin`` role, the effective
    ``subject_id`` filter is forced to the principal's own ``subject`` (a requested value
    for ANOTHER subject is ignored, never honored — so a subject-scoped tenant cannot
    enumerate another subject's runs). Every other principal — ``all_tenants`` / offline /
    the historical multi-user-workspace default / a ``tenant_admin`` — keeps the
    caller-supplied ``requested`` as-is, so the zero-config path is byte-unchanged.
    """
    principal = get_principal(request)
    if principal.all_tenants or not principal.subject_scoped:
        return requested
    from himmy.api.auth.principal import TENANT_ADMIN_ROLE

    if TENANT_ADMIN_ROLE in principal.roles:
        return requested
    return principal.subject


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
    "is_multi_tenant",
    "principal_dependency",
    "get_principal",
    "resolve_workspace",
    "require_workspace",
    "authorize_object",
    "narrow_subject",
]
