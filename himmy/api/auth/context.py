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
from himmy.config.flags import env_truthy

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
    if env_truthy("HIMMY_MULTI_TENANT"):
        return True
    return os.environ.get("HIMMY_AUTH_MODE", "").lower() not in ("", "none")


def build_authenticator() -> Authenticator | None:
    """Select an authenticator from env, or ``None`` for the offline (no-auth) default.

    Precedence: OIDC (``HIMMY_AUTH_MODE=oidc``) ▸ API keys (mapped file and/or shared
    ``HIMMY_INTERNAL_API_KEY``) ▸ none. Mapped keys (``HIMMY_API_KEYS_FILE``) and a
    shared key can coexist.

    Under the multi-tenant posture a *shared* key is built DEMOTED to operator-only (no
    tenant reach) so a shared-key match can no longer act as a cross-tenant admin (G1); a
    shared-key-ONLY deploy is then additionally refused at startup because it still binds
    nobody. The posture engages when EITHER the operator declared it explicitly via env
    (:func:`is_multi_tenant`) OR a tenant-mapped keys file is present (``bool(records)``)
    — a per-tenant ``HIMMY_API_KEYS_FILE`` is multi-tenant IN FACT even without the env
    flag, so the demotion must not silently skip it (else a co-configured shared key would
    stay a cross-tenant super-admin — the exact G1 hole this closes).
    """
    mode = os.environ.get("HIMMY_AUTH_MODE", "").lower()
    if mode == "oidc":
        from himmy.api.auth.oidc import OidcAuthenticator

        return OidcAuthenticator.from_env()

    from himmy.api.auth.apikey import (
        DEFAULT_HEADER,
        DEMOTED_SHARED_KEY_ROLES,
        ApiKeyAuthenticator,
        load_key_records,
    )
    from himmy.config.secrets import get_secret

    header = os.environ.get("HIMMY_INTERNAL_HEADER", DEFAULT_HEADER)
    shared = {
        k.strip()
        for k in (get_secret("HIMMY_INTERNAL_API_KEY") or "").split(",")
        if k.strip()
    }
    # Mapped keys are loaded as lifecycle-aware records (hashed in-memory, carrying any
    # per-key expires_at/disabled) rather than bare principals, so the file's expiry +
    # disabled metadata is enforced at authenticate time.
    records = []
    keys_file = os.environ.get("HIMMY_API_KEYS_FILE")
    if keys_file:
        records = load_key_records(keys_file)
    if shared or records:
        # A tenant-mapped keys file makes the deploy multi-tenant IN FACT (binds_tenants),
        # so engage the demotion even when the env flag is absent — never trust the env
        # flag alone to detect a multi-tenant posture (G1/G2 fail-open fix).
        multi_tenant = is_multi_tenant() or bool(records)
        return ApiKeyAuthenticator(
            shared_keys=shared,
            records=records,
            header_name=header,
            shared_key_roles=(
                DEMOTED_SHARED_KEY_ROLES if multi_tenant else None
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


def enforce_subject_write(request: Request, subject_id: str | None) -> None:
    """Object-level (BOLA) WRITE gate: may this principal CREATE/MUTATE under ``subject_id``?

    The write-path companion to :func:`authorize_object` (by-id read) and
    :func:`narrow_subject` (list read). The read gates were applied symmetrically across
    the runs/threads readers, but the CREATE/UPSERT paths took ``subject_id`` straight from
    the request body — letting an opt-in ``subject_scoped`` principal STAMP a run/thread/PII
    field under ANOTHER data subject (an attribution / lineage / erasure-scope poisoning
    BOLA). This closes that asymmetry: a ``subject_scoped`` principal WITHOUT the
    ``tenant_admin`` role may only write under its OWN ``subject``; a mismatch is **403**.

    Like every other gate here this is a strict NO-OP for the offline / ``all_tenants`` /
    historical-multi-user-workspace / ``tenant_admin`` path — those principals are not
    ``subject_scoped``, so the check short-circuits and the zero-config write path is
    byte-unchanged. A ``None`` subject (a subject-less legacy write) is always allowed.
    """
    principal = get_principal(request)
    if principal.may_access_subject(subject_id):
        return
    from himmy.api.security_audit import audit_event

    audit_event(
        request,
        event_type="authz_denied",
        outcome="deny",
        resource="subject",
        action="write",
        detail="subject_scoped principal may only write under its own subject_id",
    )
    raise HTTPException(
        status_code=403,
        detail="subject access denied",
    )


def studio_tenant_filter(request: Request) -> frozenset[str] | None:
    """The set of workspaces a Studio LIST reader may surface, or ``None`` for ALL.

    Studio's global run/thread/file readers historically showed EVERY canonical
    object regardless of tenant (it was a single-user-local console). Once an
    authenticator binds a caller to specific tenants, that becomes a cross-tenant
    read: a tenant-bound principal could browse another tenant's runs in the Studio
    GUI. This returns the allow-list a list-reader must intersect against:

    * ``None`` for an unrestricted principal (offline default / ANONYMOUS / trusted
      shared key) — meaning "no filtering, show everything", so the zero-config /
      single-box path is **byte-unchanged**; and
    * the principal's own ``tenant_ids`` for a tenant-bound principal, so it sees only
      runs in workspaces it is entitled to.
    """
    principal = get_principal(request)
    if principal.all_tenants:
        return None
    return principal.tenant_ids


def studio_subject_filter(request: Request) -> str | None:
    """The data subject a Studio LIST reader must pin to, or ``None`` for ALL (BOLA).

    The subject-axis companion to :func:`studio_tenant_filter` (the tenant axis) for the
    Studio run/list readers. Studio historically showed EVERY run regardless of data
    subject; once a deployment opts a principal into ``subject_scoped``, that becomes a
    cross-subject read WITHIN the tenant — the exact BOLA the ``/v1`` and missions readers
    already close via :func:`narrow_subject` / :func:`authorize_object`. Returns:

    * ``None`` for every principal except an opt-in ``subject_scoped`` one (offline /
      ``all_tenants`` / the historical multi-user-workspace default / a ``tenant_admin``),
      meaning "no subject filtering, show everything" — so the zero-config / single-box
      path is **byte-unchanged**; and
    * the principal's own ``subject`` for a ``subject_scoped`` principal WITHOUT the
      ``tenant_admin`` role, so it sees only runs attributed to its own data subject.
    """
    principal = get_principal(request)
    if principal.all_tenants or not principal.subject_scoped:
        return None
    from himmy.api.auth.principal import TENANT_ADMIN_ROLE

    if TENANT_ADMIN_ROLE in principal.roles:
        return None
    return principal.subject


def studio_write_workspace(request: Request) -> str | None:
    """The single ``workspace_id`` a Studio singleton-store WRITE must be stamped with.

    The write-path companion to :func:`studio_tenant_filter` (the LIST read path) for the
    cwd-keyed singleton Studio stores (tasks/notes/calendar/cookbook/notify), which key a
    row by a single owning workspace rather than a per-request body field. Returns:

    * ``None`` for an unrestricted principal (offline default / ANONYMOUS / a trusted shared
      key), so a write lands UNSTAMPED exactly as before — the zero-config / single-box path
      is **byte-unchanged** and the shared tool packs (which never pass a workspace) keep
      writing ``NULL``-tenant rows;
    * the principal's sole tenant for a tenant-bound principal, so its writes are stamped to
      the workspace it is entitled to and become invisible to other tenants;
    * **400** when a tenant-bound principal binds MORE THAN ONE tenant — a Studio write
      cannot guess which workspace to stamp (mirrors :func:`resolve_workspace`'s
      require-a-default rule), so the ambiguity is a client error, not a silent mis-stamp.
    """
    principal = get_principal(request)
    if principal.all_tenants:
        return None
    default = principal.default_tenant()
    if default is not None:
        return default
    raise HTTPException(
        status_code=400,
        detail="workspace_id is required for a multi-tenant principal",
    )


def authorize_studio_object(request: Request, workspace_id: str | None) -> bool:
    """By-id (BOLA-style) gate for a Studio object: may this caller read ``workspace_id``?

    The by-id companion to :func:`studio_tenant_filter` (the list path). Returns True
    for an unrestricted principal (offline / ``all_tenants``) so the single-box path is
    a NO-OP, and otherwise the principal's :meth:`~Principal.may_access` verdict on the
    object's owning workspace. A ``None`` workspace (a legacy/uncategorized run with no
    tenant stamp) is allowed — it predates tenant binding and belongs to no tenant.

    Callers fold a False verdict into a **404** (not 403) so a tenant cannot even probe
    the existence of another tenant's run by id — mirroring the not-found-on-cross-tenant
    convention of the ``/v1`` :func:`get_run` reader.
    """
    if workspace_id is None:
        return True
    return get_principal(request).may_access(workspace_id)


def reveal_host_posture(request: Request) -> bool:
    """Whether this request may see OPERATOR DEPLOYMENT POSTURE on a ``/v1`` infra read.

    The ``/v1`` analogue of Studio's :func:`~himmy.api.routers.studio._caller_holds_console_write`
    posture gate, but keyed on the tenant axis rather than the ``studio.console:write`` grant —
    because the ``/v1`` infra reads (``GET /v1/diagnostics`` / ``GET /v1/models``) carry no
    Studio role dimension, only a principal's tenant binding. "Host posture" is the
    reconnaissance class red-team r6/r9 withheld from tenant browse roles on the Studio twins:
    which LLM binaries are installed + their absolute filesystem paths, the provider-key
    PRESENCE inventory, the live local Ollama model inventory, the storage backend DSN host, and
    the canonical ``.himmy`` SQLite file paths.

    Returns the principal's :attr:`~himmy.api.auth.principal.Principal.all_tenants` flag, so:

    * an unrestricted principal — the offline default / :data:`ANONYMOUS` (``all_tenants=True``)
      / a trusted shared key — gets the FULL posture, so the zero-config / single-box path is
      **byte-unchanged** (the report the CLI ``himmy doctor`` / ``himmy models`` prints); and
    * a tenant-bound principal (``all_tenants=False``) gets a POSTURE-FREE view — it may no
      longer learn the operator's installed binaries, key inventory, DB host, or live local
      model list, mirroring exactly what the hardened Studio ``/doctor`` / ``/models`` withhold.
    """
    return get_principal(request).all_tenants


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
    "enforce_subject_write",
    "studio_tenant_filter",
    "studio_subject_filter",
    "studio_write_workspace",
    "authorize_studio_object",
    "reveal_host_posture",
]
