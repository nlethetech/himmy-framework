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
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# fastapi is the [api] extra; this module must still IMPORT on a core (offline) install — the
# route dependency below is only ever built when the API server is running.
try:
    from fastapi import HTTPException, Request
except ModuleNotFoundError:  # pragma: no cover - exercised only when the API server runs
    HTTPException = Request = None  # type: ignore[assignment, misc]

from himmy.api.auth.context import get_principal
from himmy.core.errors import HimmyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Awaitable, Callable

    from himmy.api.auth.principal import Principal

#: Loud, operator-facing warnings about a policy that *parses* but is dangerously
#: permissive or suspect (an empty lock-out, a wildcard on a non-admin role, a typo'd
#: token). Surfaced both to the log AND collected by :func:`lint_policy` for the
#: ``himmy rbac validate`` CLI so an operator sees them before shipping.
logger = logging.getLogger("himmy.api.auth.rbac")

#: The built-in role catalogue. ``admin`` is unrestricted; ``viewer`` reads
#: operational data (incl. the read-only model catalog + global diagnostics);
#: ``operator`` reads + writes it (``model:write`` covers the run-fanning compare);
#: ``auditor`` additionally reads the audit surface AND runs the WS4.7 privacy audit
#: (``audit:run``). ``data_subject`` is a self-scoped role for a person exercising
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
        "agent:read",
        "knowledge:read",
        "routine:read",
        "model:read",
        "diagnostics:read",
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
        "agent:read",
        "agent:write",
        "knowledge:read",
        "knowledge:write",
        "routine:read",
        "routine:write",
        "model:read",
        "model:write",
        "diagnostics:read",
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
        "agent:read",
        "knowledge:read",
        "routine:read",
        "model:read",
        "diagnostics:read",
    ],
    # Self-scoped: holds only consent:read (so it can read its own decision/history and
    # exercise withdrawal/erasure). The /v1/consent router enforces it may touch ONLY its
    # own subject_id; it has no operational/write reach.
    "data_subject": [
        "consent:read",
    ],
    "admin": ["*:*"],
}


class RbacPolicyError(HimmyError):
    """A malformed or unsafe RBAC policy (bad shape, bad permission spec).

    Raised by :func:`_parse_perm`, :meth:`AccessPolicy.from_mapping` and
    :func:`load_policy` so a hand-edited policy fails CLOSED at parse/startup time
    with a clear, named message instead of silently widening access (fail-open) or
    crashing later as a raw ``TypeError`` 500. Subclasses :class:`HimmyError` so the
    API's global handler surfaces it as a structured 400 and the CLI/startup paths can
    catch it broadly.
    """


def _parse_perm(spec: str) -> tuple[str, str]:
    """Parse a STRICT ``"resource:action"`` permission string into a tuple.

    Fail CLOSED on anything ambiguous: a typo'd ``"run:"`` would, under a naive
    ``or "*"`` fallback, silently WIDEN to ``("run", "*")`` (every action) and
    ``":read"`` to ``("*", "read")`` (every resource) — a hand-edited policy quietly
    granting more than intended. So we require both halves to be present and
    non-empty; a wildcard must be written as the LITERAL ``"*"`` (it can never be
    produced by emptiness). Anything else raises :class:`RbacPolicyError`.
    """
    if not isinstance(spec, str):
        raise RbacPolicyError(
            f"RBAC permission must be a string, got {type(spec).__name__}: {spec!r}"
        )
    if ":" not in spec:
        raise RbacPolicyError(
            f"RBAC permission {spec!r} is malformed: expected 'resource:action' "
            "(missing ':')"
        )
    resource, _, action = spec.partition(":")
    resource, action = resource.strip(), action.strip()
    if not resource or not action:
        raise RbacPolicyError(
            f"RBAC permission {spec!r} is malformed: both 'resource' and 'action' "
            "are required (use the literal '*' for a wildcard, never an empty half)"
        )
    return resource, action


@dataclass(frozen=True)
class AccessPolicy:
    """An immutable role → permissions map with wildcard-aware authorization."""

    role_permissions: dict[str, frozenset[tuple[str, str]]]

    @classmethod
    def from_mapping(cls, mapping: dict[str, list[str]]) -> AccessPolicy:
        """Build a policy from ``{role: ["resource:action", ...]}``, validated strictly.

        Fail CLOSED on a malformed shape — the error message NAMES the offending role
        so an operator can find it: a non-dict top level, a role whose value is not a
        list, or a permission that is not a string all raise :class:`RbacPolicyError`
        instead of silently dropping/widening grants. Permission specs are parsed via
        the strict :func:`_parse_perm` (no emptiness-as-wildcard).

        Loud (log-only, non-fatal) WARNINGS — surfaced via :func:`lint_policy`/the CLI
        — flag a policy that *parses* but is dangerous: an empty ``{}`` (locks
        everyone out) or a NON-``admin`` role granted a ``*:*`` / ``*:<action>``
        wildcard (a likely over-grant). See :func:`lint_policy` for the full report.
        """
        if not isinstance(mapping, dict):
            raise RbacPolicyError(
                "RBAC policy must be a JSON object mapping role → [permissions], got "
                f"{type(mapping).__name__}"
            )
        role_permissions: dict[str, frozenset[tuple[str, str]]] = {}
        for role, perms in mapping.items():
            role_name = str(role)
            if not isinstance(perms, list):
                raise RbacPolicyError(
                    f"RBAC role {role_name!r}: permissions must be a list of "
                    f"'resource:action' strings, got {type(perms).__name__}"
                )
            parsed: list[tuple[str, str]] = []
            for perm in perms:
                try:
                    parsed.append(_parse_perm(perm))
                except RbacPolicyError as exc:
                    raise RbacPolicyError(f"RBAC role {role_name!r}: {exc}") from exc
            role_permissions[role_name] = frozenset(parsed)
        # Loud warnings for parses-but-DANGEROUS policies (never fatal): an empty
        # lock-out and non-admin wildcards. Typo warnings are intentionally NOT logged
        # here (they compare against the SERVER catalogue and would spuriously fire for
        # legitimately-different vocabularies, e.g. the CLI policy) — the
        # ``himmy rbac validate`` CLI surfaces those via :func:`lint_policy`.
        for message in _policy_warnings(role_permissions, include_typos=False):
            logger.warning("%s", message)
        return cls(role_permissions)

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


#: The role name that is *expected* to hold the ``*:*`` super-grant; a wildcard on
#: any OTHER role is flagged as a likely over-grant.
_ADMIN_ROLE = "admin"


def _known_tokens() -> tuple[frozenset[str], frozenset[str]]:
    """The known ``(resources, actions)`` vocab drawn from :data:`DEFAULT_RBAC`.

    Used only as a TYPO catcher (warning, never an error): a custom policy is free to
    introduce its own resources/actions, but a token that matches none of the built-in
    catalogue is more often a typo (``"recommendaton:read"``) than an intentional new
    surface, so :func:`lint_policy` warns on it.
    """
    resources: set[str] = set()
    actions: set[str] = set()
    for perms in DEFAULT_RBAC.values():
        for spec in perms:
            res, _, act = spec.partition(":")
            resources.add(res)
            actions.add(act)
    return frozenset(resources), frozenset(actions)


def _policy_warnings(
    role_permissions: dict[str, frozenset[tuple[str, str]]],
    *,
    include_typos: bool = True,
) -> list[str]:
    """Loud (non-fatal) warnings about a policy that PARSES but is dangerous/suspect.

    Catches the silent footguns: an empty ``{}`` (locks everyone out), a non-``admin``
    role granted a ``*:*`` / ``*:<action>`` wildcard (a likely over-grant), and — when
    ``include_typos`` is set — tokens that match no known resource/action in
    :data:`DEFAULT_RBAC` (likely typos). These are warnings only — a custom deployment
    may legitimately define new roles/resources — but they are surfaced loudly so an
    operator notices.

    ``include_typos`` is off for the auto-log in :meth:`AccessPolicy.from_mapping`
    (the typo check compares against the SERVER vocab and would spuriously fire for a
    legitimately different one, e.g. the CLI policy); the ``himmy rbac validate`` CLI
    turns it on.
    """
    warnings: list[str] = []
    if not role_permissions:
        warnings.append(
            "RBAC policy is EMPTY: no roles are defined, so every authenticated "
            "caller is denied by default (a total lock-out). Did you mean to ship a "
            "policy with at least one role?"
        )
    known_resources, known_actions = _known_tokens()
    for role in sorted(role_permissions):
        perms = role_permissions[role]
        if role != _ADMIN_ROLE:
            for res, act in sorted(perms):
                if res == "*":
                    warnings.append(
                        f"RBAC role {role!r} is granted a wildcard resource "
                        f"('*:{act}') — every resource is reachable. Only 'admin' is "
                        "expected to hold a wildcard; double-check this is intended."
                    )
        if not include_typos:
            continue
        for res, act in sorted(perms):
            if res != "*" and res not in known_resources:
                warnings.append(
                    f"RBAC role {role!r}: resource {res!r} matches no known resource "
                    f"in the built-in catalogue — possible typo (in permission "
                    f"'{res}:{act}')."
                )
            if act != "*" and act not in known_actions:
                warnings.append(
                    f"RBAC role {role!r}: action {act!r} matches no known action in "
                    f"the built-in catalogue — possible typo (in permission "
                    f"'{res}:{act}')."
                )
    return warnings


def lint_policy(
    mapping: object,
) -> tuple[AccessPolicy | None, list[str], list[str]]:
    """Validate + lint a raw policy mapping; return ``(policy, errors, warnings)``.

    The non-raising counterpart of :meth:`AccessPolicy.from_mapping`, for the
    ``himmy rbac validate`` CLI: it collects the FATAL errors (bad shape, malformed
    perm specs — each naming the offending role) and the non-fatal WARNINGS (empty
    lock-out, non-admin wildcards, unknown-token typos) instead of raising on the
    first error.

    ``policy`` is the built :class:`AccessPolicy` when there were no errors, else
    ``None``. A caller exits non-zero iff ``errors`` is non-empty.
    """
    errors: list[str] = []
    if not isinstance(mapping, dict):
        return (
            None,
            [
                "RBAC policy must be a JSON object mapping role → [permissions], got "
                f"{type(mapping).__name__}"
            ],
            [],
        )
    role_permissions: dict[str, frozenset[tuple[str, str]]] = {}
    for role, perms in mapping.items():
        role_name = str(role)
        if not isinstance(perms, list):
            errors.append(
                f"RBAC role {role_name!r}: permissions must be a list of "
                f"'resource:action' strings, got {type(perms).__name__}"
            )
            continue
        parsed: list[tuple[str, str]] = []
        for perm in perms:
            try:
                parsed.append(_parse_perm(perm))
            except RbacPolicyError as exc:
                errors.append(f"RBAC role {role_name!r}: {exc}")
        role_permissions[role_name] = frozenset(parsed)
    warnings = _policy_warnings(role_permissions)
    policy = AccessPolicy(role_permissions) if not errors else None
    return policy, errors, warnings


#: The default policy (used when no ``HIMMY_RBAC_FILE`` is configured).
DEFAULT_POLICY = AccessPolicy.from_mapping(DEFAULT_RBAC)


def load_policy(path: str | Path) -> AccessPolicy:
    """Load an :class:`AccessPolicy` from a JSON ``{role: [perm, ...]}`` file.

    Fails CLOSED with a clear :class:`RbacPolicyError` (a :class:`HimmyError`, mapped
    to a structured 400 / caught at startup) on unreadable/invalid JSON or a non-object
    top level — never a raw ``TypeError``/``JSONDecodeError`` 500. Shape + per-role
    validation happens in :meth:`AccessPolicy.from_mapping`.
    """
    target = Path(path).expanduser()
    try:
        text = target.read_text()
    except OSError as exc:
        raise RbacPolicyError(f"RBAC file {path}: cannot read ({exc})") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RbacPolicyError(f"RBAC file {path}: invalid JSON ({exc})") from exc
    if not isinstance(raw, dict):
        raise RbacPolicyError(f"RBAC file {path} must be a JSON object")
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
    "RbacPolicyError",
    "DEFAULT_RBAC",
    "DEFAULT_POLICY",
    "lint_policy",
    "load_policy",
    "build_access_policy",
    "require_permission",
]
