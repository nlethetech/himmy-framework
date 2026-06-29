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
#: ``operator`` reads + writes it (``model:write`` covers the run-fanning compare) AND
#: holds ``tool:*`` — the every-tool, read+write capability an autonomous agent run
#: needs, so the tool-capability gate (see :mod:`himmy.services.tools.capability`) keeps
#: today's behavior for operator humans AND the routine/connector SERVICE principals
#: (which run as ``operator``) once an authenticator is configured. Read-only roles
#: (``viewer``/``auditor``) carry NO ``tool:*`` because the single-colon wildcard grammar
#: cannot express "every tool, read-only"; a deployment that wants those roles to invoke
#: look-up tools lists them explicitly (``tool:<name>:invoke``) via ``HIMMY_RBAC_FILE``.
#: ``auditor`` additionally reads the audit surface AND runs the WS4.7 privacy audit
#: (``audit:run``). ``data_subject`` is a self-scoped role for a person exercising
#: their own consent/erasure rights (the router additionally restricts it to its own
#: ``subject_id``). Operators ship their own via ``HIMMY_RBAC_FILE``.
#: The Studio console's per-surface RBAC resources (mirroring the ``/v1`` per-resource
#: pattern). Each maps to one Studio surface/sub-router (see
#: :mod:`himmy.api.routers.studio_common`); a read-only role is granted ``read`` on ALL
#: of them so it can browse the whole console, while every ``write`` / ``manage`` action
#: stays admin-only (the single-colon ``studio.*`` wildcard the grammar would need
#: cannot be expressed, so reads are enumerated explicitly here — a deployment adding a
#: NEW Studio surface lists its ``studio.<surface>:read`` here or via ``HIMMY_RBAC_FILE``).
#: ``console`` is the low-privilege "may open Studio at all" baseline the main router
#: carries; the rest are the granular surfaces.
STUDIO_READ_RESOURCES: tuple[str, ...] = (
    "studio.console",
    "studio.runs",
    "studio.connections",
    "studio.google",
    "studio.approvals",
    "studio.models",
    # red-team reattack-r6: the BENIGN model CATALOG (the tenant-facing "model picker",
    # GET /api/studio/models -> build_model_catalog) split off ``studio.models`` so the
    # catalog can stay tenant-grantable while ``studio.models`` (the operator
    # credential-status surface — GET /api/studio/models/providers) is withheld to
    # admin-only via :data:`_STUDIO_GLOBAL_STORE_RESOURCES`. Same catalog ``GET /v1/models``
    # (``model:read``) and ``himmy models`` render; carries NO provider key posture.
    "studio.modelcatalog",
    "studio.agents",
    "studio.tasks",
    "studio.chats",
    "studio.cookbook",
    "studio.notes",
    "studio.calendar",
    "studio.memory",
    "studio.knowledge",
    "studio.evals",
    "studio.workflows",
    "studio.files",
    "studio.mcp",
    "studio.eval",
    "studio.guardrails",
    "studio.kb",
    "studio.learning",
    "studio.lineage",
    "studio.missions",
    "studio.notify",
    "studio.privacy",
    "studio.projects",
    "studio.routines",
    "studio.seclog",
    "studio.teams",
    "studio.telegram",
)

#: Studio surfaces backed by a PROCESS-WIDE store that has NO tenant or subject axis
#: the reader can intersect against. Two flavours, same exposure:
#:
#: * the *single-user-local* stores (memory/notes/tasks/chats/cookbook/calendar/
#:   knowledge/projects/files readers all enumerate one cwd-keyed store shared across
#:   every tenant); and
#: * the *process-wide spine readers* (``seclog``/``lineage``/``privacy``) that walk the
#:   shared ``.himmy/spine.db`` entity registry / security-audit log by id or kind with
#:   no tenant filter — ``seclog`` returns EVERY tenant's actor subjects, source IPs and
#:   denied actions; ``lineage`` resolves any run/entity payload + metadata by id;
#:   ``privacy`` lists every data subject and their consent chains/actors across all
#:   tenants.
#:
#: Unlike the run readers — which were retrofitted with ``studio_tenant_filter`` — none
#: of these can intersect against the caller's tenant because the store has no workspace
#: partition to intersect ON. Granting ``studio.<surface>:read`` to a TENANT-FACING role
#: (viewer/operator/auditor — the roles a mapped API key is bound with) would therefore
#: let any paying tenant read every OTHER tenant's/subject's private memory, transcripts,
#: notes, files, security-audit events, lineage payloads, and privacy/consent records via
#: the Studio console (a cross-tenant/cross-subject BOLA / IDOR). So these surfaces are
#: withheld from the default browse roles and remain ``admin``-only (Studio is documented
#: as a network-isolated OPERATOR console); a deployment that deliberately exposes one to
#: a tenant role can still grant it explicitly via ``HIMMY_RBAC_FILE``.
#: The OFFLINE path is unaffected — ``require_permission`` no-ops without an authenticator,
#: so the single-box console reads everything byte-unchanged.
_STUDIO_GLOBAL_STORE_RESOURCES: frozenset[str] = frozenset(
    {
        "studio.memory",
        "studio.notes",
        "studio.tasks",
        "studio.chats",
        "studio.cookbook",
        "studio.calendar",
        "studio.knowledge",
        "studio.projects",
        "studio.files",
        # Process-wide spine readers exposing cross-tenant security/PII/lineage data
        # (no per-tenant partition to filter on) — admin-only, same rationale as above.
        "studio.seclog",
        "studio.lineage",
        "studio.privacy",
        # red-team r6: the deployment's OWN operator-connector surfaces — there is one
        # process-wide connection store (the operator's SMTP/Telegram/web creds) and one
        # process-wide Google OAuth connection (the operator's connected Gmail/Calendar),
        # neither of which has a per-tenant partition to intersect against. Granting their
        # ``:read`` to a TENANT-FACING browse role (viewer/operator/auditor — the roles a
        # mapped API key binds) let a paying tenant read the operator's connection status +
        # masked field hints and drive/read the operator's connected Google account through
        # the Studio console (documented as a network-isolated OPERATOR surface, never
        # tenant-facing). Withheld from the default browse roles and kept ``admin``-only,
        # same rationale as the global stores above; a deployment that deliberately exposes
        # one to a tenant role can still grant it explicitly via ``HIMMY_RBAC_FILE``. The
        # OFFLINE path is unaffected (``require_permission`` no-ops without an authenticator).
        "studio.connections",
        "studio.google",
        # red-team reattack-r1: four more PROCESS-WIDE operator surfaces missed by the
        # r1/r2/r6 sweep — same class (a single app.state/module-level singleton with NO
        # tenant axis to intersect on), so granting their ``:read`` to a tenant-facing
        # browse role (viewer/operator/auditor) was a cross-tenant BOLA/IDOR:
        #   * ``studio.missions`` — the process-wide mission registry
        #     (:func:`himmy.api.missions.get_registry`); Mission objects carry NO
        #     workspace_id, so list/by-id/SSE-stream leaked every other tenant's prompt
        #     and live agent output.
        #   * ``studio.notify`` — the module-level notification ring
        #     (:mod:`himmy.api.routers.studio_notify`) accumulating every tenant's
        #     routine/mission/project run-output previews.
        #   * ``studio.kb`` — the knowledge-UPLOAD sub-router backing the SAME
        #     process-wide ``studio_knowledge._service()`` singleton as the (already
        #     withheld) ``studio.knowledge`` twin; leaked cross-tenant document metadata.
        #   * ``studio.telegram`` — the process-wide Telegram listener singleton's
        #     operator config metadata (agent_path, provider/model, allowed chat ids).
        # Withheld from the default browse roles (admin-only, Studio is a network-isolated
        # operator console); a deployment that deliberately exposes one to a tenant role
        # can still grant it explicitly via ``HIMMY_RBAC_FILE``. The OFFLINE path is
        # unaffected — ``require_permission`` no-ops without an authenticator.
        "studio.missions",
        "studio.notify",
        "studio.kb",
        "studio.telegram",
        # red-team reattack-r4: two more PROCESS-WIDE operator surfaces of the SAME class
        # the r1/r2/r6/reattack-r1 sweep was closing — a single cwd-keyed store with NO
        # per-tenant axis to intersect on, so granting their ``:read`` to a tenant-facing
        # browse role (viewer/operator/auditor) was a cross-tenant BOLA/IDOR disclosure of
        # operator infrastructure:
        #   * ``studio.mcp`` — the process-wide MCP server registry
        #     (``.himmy/mcp_servers.json`` under :func:`project_root`, no workspace field);
        #     its ``ServerOut`` readers leaked every registered server's ``command``/``args``/
        #     ``cwd`` (absolute filesystem paths), ``env_keys`` (the operator's secret NAMES),
        #     ``prefix`` and cached tool schemas — operator-infrastructure reconnaissance a
        #     tenant must never see. (Write/manage stay ``studio.mcp:manage`` = admin-only.)
        #   * ``studio.approvals`` — the process-wide HITL checkpoint store
        #     (``.himmy/approvals.db`` via :func:`get_checkpoint_store`, single shared
        #     ``"studio"`` tenant on Postgres, no per-caller partition); its readers leaked
        #     every paused Studio run's prompt, agent name, pending tool-call names + redacted
        #     args and a 4-message thread preview across tenants. (Approve/reject/write stay
        #     ``studio.approvals:write`` = admin-only, so this was disclosure-only.)
        # Withheld from the default browse roles (admin-only, Studio is a network-isolated
        # operator console); a deployment that deliberately exposes one to a tenant role can
        # still grant it explicitly via ``HIMMY_RBAC_FILE``. The OFFLINE path is unaffected —
        # ``require_permission`` no-ops without an authenticator.
        "studio.mcp",
        "studio.approvals",
        # red-team reattack-r6: ``studio.models`` is the operator PROVIDER-CREDENTIAL
        # posture surface, NOT a tenant model picker. GET /api/studio/models/providers
        # (build_studio_router("models"), gated only by ``studio.models:read``) enumerates
        # every key-based provider (openrouter/anthropic/openai/pydantic-ai) with
        # ``configured`` + ``detected_via`` ('secret'|'env'), plus ``secrets_writable`` and
        # the default provider — disclosing WHICH paid LLM providers the operator has wired
        # and whether each key lives in the secret store vs an env var. This is the SAME
        # operator-infrastructure-reconnaissance class deliberately withheld for the sibling
        # ``studio.connections`` (SMTP/Telegram/web creds status) and ``studio.google``
        # (Gmail/Calendar connection status) surfaces above, but it slipped the r6 sweep
        # because it looked like a benign "model picker". No key bytes leak (presence only)
        # and writes already require ``studio.models:write`` (admin), so this was
        # disclosure-only — but it is operator deployment posture a tenant must not read.
        # Withheld from the default browse roles (admin-only, Studio is a network-isolated
        # operator console); a deployment that deliberately exposes it to a tenant role can
        # still grant ``studio.models:read`` explicitly via ``HIMMY_RBAC_FILE``. The BENIGN
        # model catalog stays tenant-readable under the separate ``studio.modelcatalog``
        # resource (see :data:`STUDIO_READ_RESOURCES`). The OFFLINE path is unaffected
        # (``require_permission`` no-ops without an authenticator).
        "studio.models",
        # red-team reattack-r7: two more SINGLE-USER-LOCAL operator surfaces of the SAME
        # class the r1/r4/r6 sweep was closing — a process-wide store hard-scoped to the
        # operator's ``__local__`` workspace with NO per-tenant axis the Studio list can
        # intersect on, so granting their ``:read`` to a tenant-facing browse role
        # (viewer/operator/auditor — the roles a mapped API key binds) was a cross-boundary
        # disclosure of operator infrastructure:
        #   * ``studio.routines`` — the routines store (``.himmy/routines.db``). The Studio
        #     LIST (GET /api/studio/routines -> list_routines) hard-codes
        #     ``workspace_id=LOCAL_WORKSPACE`` and has NO ``studio_tenant_filter`` /
        #     ``authorize_studio_object`` gate (unlike the by-id paths, which 404 a foreign
        #     ``__local__`` row), so it returned EVERY operator-local routine's ``name``,
        #     ``agent_path`` (an absolute server filesystem path = infrastructure recon),
        #     ``prompt``, ``provider``/``model``, ``last_status``/``last_error`` and
        #     ``last_preview`` (up to 400 chars of the routine RUN OUTPUT). The list cannot
        #     intersect on a tenant axis (it is hard-scoped to ``__local__``), so the fix is
        #     to withhold the resource rather than retrofit a filter. (Write/manage stay
        #     ``studio.routines:write`` = admin, so this was disclosure-only.)
        #   * ``studio.eval`` — the eval-suite discovery surface (GET /api/studio/eval/suites
        #     -> list_suites). ``discover_suites()`` enumerates the operator's local
        #     filesystem eval-suite NAMES and PATHS (``RunnableSuite.path``/``source`` =
        #     absolute/project paths), the same operator-local FS reconnaissance class.
        #     (Run/write stay ``studio.eval:write`` = admin, so this was disclosure-only.)
        # Withheld from the default browse roles (admin-only, Studio is a network-isolated
        # operator console); a deployment that deliberately exposes one to a tenant role can
        # still grant ``studio.routines:read`` / ``studio.eval:read`` explicitly via
        # ``HIMMY_RBAC_FILE``. The OFFLINE path is unaffected — ``require_permission``
        # no-ops without an authenticator.
        "studio.routines",
        "studio.eval",
        # red-team reattack-r8: two operator-local DISCOVERY surfaces of the SAME class the
        # r7 sweep closed for ``studio.eval`` — a ``project_root()``-globbing read with NO
        # per-tenant axis to intersect on, so granting their ``:read`` to a tenant-facing
        # browse role (viewer/operator/auditor) leaked operator-local filesystem inventory:
        #   * ``studio.evals`` — GET /api/studio/evals (studio.eval_suites) is the
        #     un-hardened twin of the r7-locked GET /api/studio/eval/suites; both call
        #     ``studio_eval.discover_suites()``, enumerating the operator's local eval-suite
        #     NAMES/PATHS/case-counts. (Run/write stay ``studio.evals:write`` = admin.)
        #   * ``studio.workflows`` — GET /api/studio/workflows (studio.workflows ->
        #     ``discover_workflows()``) enumerates the operator's workflow specs:
        #     project-relative path + name + the step/tool graph (orchestration topology).
        #     (Run/write stay ``studio.workflows:write`` = admin.)
        # Both globs are hard-scoped to the operator project root with no tenant partition, so
        # the fix is to WITHHOLD the resource (admin-only) rather than retrofit a filter,
        # mirroring the r7 treatment of ``studio.eval``/``studio.routines``. A deployment that
        # deliberately exposes one to a tenant role can still grant ``studio.evals:read`` /
        # ``studio.workflows:read`` explicitly via ``HIMMY_RBAC_FILE``. The OFFLINE path is
        # unaffected — ``require_permission`` no-ops without an authenticator.
        "studio.evals",
        "studio.workflows",
        # red-team reattack-r10: two more operator-local DISCOVERY surfaces of the EXACT
        # class the r7/r8 sweep closed for ``studio.eval``/``studio.evals``/
        # ``studio.workflows``/``studio.routines`` — a ``project_root()``-globbing read with
        # NO per-tenant axis to intersect on — that slipped because they "looked like" benign
        # spec pickers:
        #   * ``studio.agents`` — GET /api/studio/agents (studio_service.list_agents) globs
        #     the single operator project root and returns each ``AgentSummary``'s
        #     project-relative server filesystem ``path`` (= infrastructure recon), ``name``,
        #     ``provider``/``model`` and tool/skill presence. (Edit/validate stay
        #     ``studio.agents:write`` = admin.)
        #   * ``studio.teams`` — GET /api/studio/teams (studio_service.list_teams) returns
        #     each ``TeamSummary``'s project-relative ``path``, ``name``, ``entry`` and the
        #     member graph (providers/models/delegates/handoffs = orchestration topology).
        # Both lists are hard-scoped to the operator project root with NO tenant partition,
        # so the fix is to WITHHOLD the resource (admin-only) — and their main-router GET
        # routes were retrofitted with the per-surface ``studio.agents:read`` /
        # ``studio.teams:read`` guard (previously they collapsed to the coarse
        # ``studio.console:read`` baseline every browse role holds). A deployment that
        # deliberately exposes one to a tenant role can still grant ``studio.agents:read`` /
        # ``studio.teams:read`` explicitly via ``HIMMY_RBAC_FILE``. The OFFLINE path is
        # unaffected — ``studio_permission`` no-ops without an authenticator.
        "studio.agents",
        "studio.teams",
    }
)

#: The read grants every browse-capable role (viewer/operator/auditor) holds on Studio:
#: ``studio.<surface>:read`` for each surface above EXCEPT the un-partitionable global-store
#: surfaces (:data:`_STUDIO_GLOBAL_STORE_RESOURCES`, which stay admin-only). Spliced into
#: those roles below so the read-only console works once an authenticator is configured,
#: while mutating ``studio.*:write`` / ``studio.mcp:manage`` permissions remain exclusively
#: ``admin``.
_STUDIO_READ_PERMS: list[str] = [
    f"{res}:read"
    for res in STUDIO_READ_RESOURCES
    if res not in _STUDIO_GLOBAL_STORE_RESOURCES
]


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
        *_STUDIO_READ_PERMS,
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
        # Every-tool, read+write capability for autonomous agent/service runs: without it,
        # enabling an authenticator would deny EVERY tool call to operator humans AND to
        # the routine/connector SERVICE principals (which run as ``operator``). The
        # tool-capability gate folds the tool name + intent into the action under the
        # ``tool`` resource (``tool:<name>:invoke`` / ``:write``), so ``tool:*`` covers
        # both. (``res='tool'`` here, so the non-admin wildcard lint — which flags a
        # wildcard RESOURCE ``*:...`` — does not fire.)
        "tool:*",
        *_STUDIO_READ_PERMS,
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
        *_STUDIO_READ_PERMS,
    ],
    # Self-scoped: holds only consent:read (so it can read its own decision/history and
    # exercise withdrawal/erasure). The /v1/consent router enforces it may touch ONLY its
    # own subject_id; it has no operational/write reach.
    "data_subject": [
        "consent:read",
    ],
    "admin": ["*:*"],
}


#: The reserved RBAC *resources* of the built-in catalogue (:data:`DEFAULT_RBAC`),
#: excluding the ``*`` wildcard. These are ALWAYS recognized for scope-narrowing
#: (:meth:`AccessPolicy.known_resources`) regardless of whether the active custom policy
#: enumerates them, so a deliberately-attenuated token scope naming a built-in resource
#: (notably ``tool:<name>:invoke``) always narrows — closing the fail-open where a custom
#: ``HIMMY_RBAC_FILE`` that grants tools ONLY via admin ``*:*`` would silently disable
#: tool-scope attenuation. Derived from :data:`DEFAULT_RBAC` so it stays in lock-step with
#: the built-in vocabulary (``tool``/``run``/``context``/``model``/… ). Splitting on the
#: FIRST colon mirrors :func:`_parse_perm`'s grammar.
_BUILTIN_RESOURCES: frozenset[str] = frozenset(
    spec.split(":", 1)[0]
    for perms in DEFAULT_RBAC.values()
    for spec in perms
    if spec.split(":", 1)[0] != "*"
)


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

    def known_resources(self) -> frozenset[str]:
        """The set of RBAC *resources* recognized for scope-narrowing (excluding ``*``).

        Used to gate scope-narrowing: only a scope whose parsed resource is one we
        recognize is allowed to engage narrowing (see :func:`_scope_permissions`), so a
        colon-bearing OIDC resource-scope an IdP emits (``api://…`` parses to resource
        ``api``, ``urn:…`` to ``urn``, ``https://…`` to ``https``) — none of which are RBAC
        resources — cannot masquerade as a permission-scope and zero out the grant. The
        wildcard ``*`` is excluded because a *scope* of ``*:<action>`` is, by the OAuth
        attenuation contract, never something a real IdP issues as a narrowing token and we
        do not want a stray ``*`` to re-admit garbage.

        The recognized set is the union of (a) the resources THIS policy actually names
        and (b) the BUILT-IN reserved resources of :data:`DEFAULT_RBAC`
        (:data:`_BUILTIN_RESOURCES`). Including the built-ins closes a fail-open in
        scope-narrowing: a custom :data:`HIMMY_RBAC_FILE` whose tool reach comes ONLY
        through admin's ``*:*`` (no role names a ``tool:…`` permission) names no ``tool``
        resource, so a deliberately-attenuated ``tool:<name>:invoke`` scope would parse but
        be DROPPED — leaving the admin ``*:*`` un-narrowed and the token able to invoke
        EVERY tool, the opposite of the operator's intent. By treating the reserved RBAC
        resources (``tool``/``run``/``context``/… from the built-in catalogue) as always
        recognized, a colon-bearing scope naming one of them always narrows, regardless of
        whether the active custom policy happens to enumerate it. Genuinely foreign IdP
        resource-scopes (``api``/``urn``/``https``/…) are still dropped — they are neither
        named by the policy nor reserved built-ins.
        """
        return frozenset(
            res
            for perms in self.role_permissions.values()
            for (res, _action) in perms
            if res != "*"
        ) | _BUILTIN_RESOURCES

    def authorize(self, principal: Principal, resource: str, action: str) -> bool:
        """Whether the principal may perform ``(resource, action)``.

        The verdict is the role-derived grant **intersected with the token's
        scopes** — scopes can only NARROW, never widen, the reach a role confers
        (delegated least-privilege, the OAuth contract: a token chooses to act with
        a SUBSET of its bearer's authority). Concretely:

        1. The principal must hold a role granting ``(resource, action)`` (the
           historical check) — a scope alone never grants anything a role does not.
        2. THEN, when the principal carries a non-empty set of *permission-scopes*
           (scopes naming a resource this policy recognizes, see below), that grant
           must ALSO be covered by one of them.

        **What counts as a permission-scope (and why it is opt-in, not guessed).**
        A scope engages narrowing ONLY when it parses as the ``resource:action``
        grammar AND its resource is one the active policy actually names
        (:meth:`known_resources`) — e.g. ``run:read``, ``context:write``, ``tool:*``.
        This is deliberately strict: mainstream IdPs (Entra ID / Keycloak / ZITADEL)
        emit colon-BEARING resource scopes such as ``api://himmy/access_as_user``,
        ``urn:zitadel:iam:org:project:role`` or ``https://graph.microsoft.com/User.Read``
        which DO parse on the first colon (``('api', '//…')`` etc.) but name no RBAC
        resource — were we to honor them, a single such scope would make the
        permission-scope set non-empty and narrow EVERY real grant to nothing,
        locking out every role (admin included). Gating on known resources drops them,
        so they narrow nothing.

        Empty (or all-unrecognized) permission-scopes ⇒ NO narrowing: the principal
        keeps its full role reach (the existing behavior — every API-key / ANONYMOUS /
        role-only principal is byte-unchanged, since those carry no scopes; a token
        carrying ONLY standard/garbage scopes likewise keeps full reach rather than
        being locked out).
        """
        granted_by_role = any(
            (perms := self.role_permissions.get(role)) is not None
            and _covers(perms, resource, action)
            for role in principal.roles
        )
        if not granted_by_role:
            return False
        scope_perms = _scope_permissions(principal.scopes, self.known_resources())
        if not scope_perms:
            return True  # no recognizable scopes → no narrowing (full role reach)
        return _covers(scope_perms, resource, action)


def _covers(perms: frozenset[tuple[str, str]], resource: str, action: str) -> bool:
    """Whether ``perms`` grants ``(resource, action)`` (``*`` wildcards match)."""
    return any(r in (resource, "*") and a in (action, "*") for (r, a) in perms)


def _scope_permissions(
    scopes: frozenset[str], known_resources: frozenset[str]
) -> frozenset[tuple[str, str]]:
    """Parse a token's ``scopes`` into the ``(resource, action)`` pairs they grant.

    Scopes are intersected with the role grant in :meth:`AccessPolicy.authorize` to
    realise delegated least-privilege (a scope can only NARROW). A scope is honored as
    a permission-scope ONLY when BOTH hold:

    1. it parses as the strict ``resource:action`` grammar (:func:`_parse_perm`,
       wildcards allowed); AND
    2. its resource is one the active policy actually names (``known_resources``,
       from :meth:`AccessPolicy.known_resources`), so it is an *opt-in*, recognized
       RBAC resource — never guessed from the colon grammar alone.

    Everything else is DROPPED, not treated as an error and not allowed to narrow
    anything. This drops TWO classes of scope that would otherwise be catastrophic or
    inert:

    * the standard OIDC ``openid`` / ``profile`` / ``email`` (no colon → fail
      :func:`_parse_perm`); and
    * mainstream IdP resource-scopes that DO parse on the first colon but name no RBAC
      resource — ``api://himmy/access_as_user`` → ``('api', '//…')``,
      ``urn:zitadel:…`` → ``('urn', '…')``, ``https://graph…`` → ``('https', '//…')``.

    Were the second class honored, one such scope would make the set non-empty and
    narrow every real grant to nothing — a total authorization lock-out (admin
    included). Enforcement is therefore fail-open *per-scope*: an unrelated/garbage
    scope cannot silently lock a caller out of every resource (the deny decision still
    rests on the role check in :meth:`AccessPolicy.authorize`). An all-unrecognized
    scope set yields the empty set, which the caller reads as "no narrowing".
    """
    parsed: set[tuple[str, str]] = set()
    for scope in scopes:
        try:
            resource, action = _parse_perm(scope)
        except RbacPolicyError:
            continue  # not a resource:action grant (e.g. 'openid') → ignore
        if resource not in known_resources:
            continue  # parses, but names no RBAC resource (e.g. OIDC 'api://…') → ignore
        parsed.add((resource, action))
    return frozenset(parsed)


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


def policy_fingerprint(policy: AccessPolicy) -> str:
    """A stable, non-reversible content HASH of a policy's grants (for the audit log).

    The audit/forensic question "did the running policy change?" needs a content
    digest, NOT the permission strings themselves — emitting the verbatim grants into
    the audit trail would copy the deployment's whole authorization matrix into a log
    that may have wider read access than the policy file, an information-disclosure
    footgun (which roles can do what, where the wildcards are). So we hash a
    canonicalized, order-independent serialization of the ``role → {(resource, action)}``
    map (roles + their perm tuples both sorted, so two equal policies always hash
    identically regardless of source ordering) and surface only the first 16 hex chars
    of a SHA-256 — enough to detect drift, impossible to invert into the grants.
    """
    import hashlib

    canonical = sorted(
        (role, sorted(perms))
        for role, perms in policy.role_permissions.items()
    )
    digest = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest[:16]}"


def policy_source() -> str:
    """A human-readable label for WHERE the active policy came from (no secrets).

    Either the configured ``HIMMY_RBAC_FILE`` path or the literal ``"<built-in>"`` for
    the default catalogue. Surfaced in the ``policy_loaded`` audit event so an operator
    can correlate a content-hash change with a file edit; the path is config, never a
    secret.
    """
    import os

    return os.environ.get("HIMMY_RBAC_FILE") or "<built-in>"


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
        # Observability (P1): record the verdict into the process-wide metrics. This
        # runs ONLY past the offline bypass above, so the zero-config path is
        # untouched. A DENY is always counted; a GRANT is sampled for privileged
        # resources (see :func:`record_authz_decision`). Best-effort and never raises.
        from himmy.services.observability.metrics import record_authz_decision

        granted = policy.authorize(get_principal(request), resource, action)
        record_authz_decision(resource, action, granted=granted)
        if not granted:
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

    # Tag the closure so the route-coverage gate (tests/api/test_rbac_route_coverage.py)
    # can recognise — at runtime, by walking each route's dependency tree — that a route
    # carries a require_permission guard, and which (resource, action) it enforces. This
    # is purely introspection metadata; it never affects the dependency's behavior.
    _dep._rbac_permission = (resource, action)  # type: ignore[attr-defined]
    return _dep


__all__ = [
    "AccessPolicy",
    "RbacPolicyError",
    "DEFAULT_RBAC",
    "DEFAULT_POLICY",
    "lint_policy",
    "load_policy",
    "build_access_policy",
    "policy_fingerprint",
    "policy_source",
    "require_permission",
]
