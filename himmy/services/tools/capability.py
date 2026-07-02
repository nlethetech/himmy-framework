"""Tool capability authorization: which tools may THIS run's principal invoke?

This closes the confused-deputy hole where the RBAC that guards a ``/v1`` route
("may you create a run?") did not follow the request down into TOOL EXECUTION
("may you invoke *this tool*?"). Without it, any caller authorized to start a run
could make that run call ANY tool the agent declares — RBAC stopped at the door and
the run acted with the agent's full authority, not the caller's.

A :class:`ToolCapabilityAuthorizer` models tool capability as first-class RBAC. Tool
capability lives under the dedicated ``tool`` RBAC *resource*, with the tool name folded
into the *action* so it composes with the existing single-colon ``resource:action``
grammar (``himmy.api.auth.rbac._parse_perm`` splits on the FIRST colon). A tool ``foo``
requires the permission ``tool:foo:invoke``, and a WRITE / side-effecting tool
additionally requires ``tool:foo:write`` — its read/write intent seeded from
:func:`himmy.services.tools.access.classify_read_only` / the tool's explicit
``read_only`` flag. An operator grants every tool with the action wildcard ``tool:*``
(and the unrestricted ``admin`` ``*:*`` covers it). The authorizer is consulted by the
tools kernel just before dispatch, deny-by-default — a principal lacking the capability
is refused the tool.

**Offline / zero-config invariant.** When no authenticator is configured the request
principal is ANONYMOUS (``all_tenants=True``); in that case ``enforce`` is ``False`` and
EVERY authorization is an unconditional pass — byte-for-byte the historical behavior, so
the offline single-box path and the ~2000-test suite are untouched. Enforcement engages
ONLY once an authenticator is configured AND the principal is tenant-bound, exactly
mirroring the established :func:`himmy.api.auth.rbac.require_permission` bypass pattern.

**Attenuation, never amplification.** A spawned sub-agent / orchestration member
inherits the parent's authorizer verbatim (:meth:`attenuate`), so a sub-agent's
capability set is always a subset of its parent's — it can never reach a tool the
parent could not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.api.auth.principal import Principal
    from himmy.api.auth.rbac import AccessPolicy
    from himmy.services.tools.models import ToolDefinition

#: The RBAC *resource* every tool capability lives under. The tool name + intent are
#: folded into the *action* (``<name>:invoke`` / ``<name>:write``) so the grant composes
#: with the existing single-colon ``resource:action`` grammar and its ``*`` wildcards
#: (``tool:*`` grants every tool; ``admin`` ``*:*`` covers it).
TOOL_RESOURCE = "tool"

#: The action granting permission to call a tool at all (read OR write).
INVOKE_ACTION = "invoke"

#: The extra action a WRITE / side-effecting tool additionally requires. Seeded from the
#: tool's read/write intent so a reader needs only ``invoke`` while a writer needs both —
#: an operator can grant ``tool:weather:invoke`` (a look-up) without also granting writes.
WRITE_ACTION = "write"


@dataclass(frozen=True)
class ToolCapabilityAuthorizer:
    """Deny-by-default gate deciding whether a run's principal may invoke a tool.

    Built once per run from the request principal + the active :class:`AccessPolicy`
    (see :meth:`from_principal`) and threaded into the per-run tool service. ``enforce``
    is the single offline-bypass switch: ``False`` (an unrestricted / ANONYMOUS principal,
    i.e. the zero-config default) makes every :meth:`authorize` an unconditional pass, so
    nothing changes for offline deployments; ``True`` (a tenant-bound principal under a
    configured authenticator) turns on the deny-by-default capability check.
    """

    enforce: bool
    roles: frozenset[str]
    policy: AccessPolicy | None = None
    #: The token's OAuth scopes, carried so the synthetic probe in :meth:`_grants`
    #: reflects the SAME scope-narrowing the route layer applies (delegated
    #: least-privilege). Empty for every offline / ANONYMOUS / API-key principal, so the
    #: gate is byte-unchanged for them; a scoped OIDC token narrows its tool reach here
    #: exactly as it narrows its ``/v1`` route reach. See :func:`himmy.api.auth.rbac`.
    scopes: frozenset[str] = frozenset()

    @classmethod
    def from_principal(
        cls, principal: Principal | None, policy: AccessPolicy | None
    ) -> ToolCapabilityAuthorizer:
        """Build an authorizer for ``principal`` under ``policy``.

        Mirrors :func:`himmy.api.auth.rbac.require_permission`'s bypass exactly: a
        ``None`` principal (no auth seam) or an unrestricted ``all_tenants`` principal
        that carries NO scopes (the ANONYMOUS offline default, or a trusted shared key)
        yields a NON-enforcing authorizer — every tool is allowed, byte-identical to
        before. A tenant-bound principal yields an enforcing one carrying its roles +
        scopes + the policy, so a scope-narrowed token narrows its TOOL reach exactly as
        it narrows its route reach.

        **Scope-narrowed all_tenants tokens still enforce (attenuation escape fix).** An
        ``all_tenants`` principal is the trusted/admin shape, but a DELIBERATELY-ATTENUATED
        delegated token (e.g. an OIDC service/CI token carrying an admin role but
        ``scopes=["run:read","run:write"]`` and NO tool scope) is all_tenants AND scoped:
        the operator minted it to start runs but NOT to invoke side-effecting tools, and
        :class:`~himmy.api.auth.rbac.AccessPolicy` ALREADY narrows its ``/v1`` route reach
        by those scopes (``tool:*`` denied). Returning a non-enforcing gate for it would
        let a run it starts invoke ANY tool — the route layer narrows but the tool gate is
        off, an amplification the module's 'Attenuation, never amplification' invariant
        forbids. So an all_tenants principal that carries scopes builds an ENFORCING
        authorizer too; the synthetic probe in :meth:`_grants` then applies the SAME
        scope-narrowing the route layer does. Only a truly scope-LESS all_tenants principal
        (offline / ANONYMOUS / unscoped shared key / role-only admin) is the byte-unchanged
        non-enforcing pass-through.
        """
        scopes = frozenset(getattr(principal, "scopes", ()) or ()) if principal else frozenset()
        if principal is None or (principal.all_tenants and not scopes):
            return cls(enforce=False, roles=frozenset())
        return cls(
            enforce=True,
            roles=frozenset(principal.roles),
            policy=policy,
            scopes=scopes,
        )

    @classmethod
    def from_actor(
        cls, actor: dict[str, Any] | None, policy: AccessPolicy | None
    ) -> ToolCapabilityAuthorizer:
        """Rebuild an authorizer from a serialized actor descriptor (recovery path).

        The leased-dispatch queue persists a run's launch input and re-executes it from a
        possibly-fresh process, where the live :class:`Principal` is gone — only the
        ``actor`` metadata (``Principal.actor_metadata()`` + the enforce flag) survives. A
        descriptor carrying ``tool_authz_enforce: True`` rebuilds the enforcing authorizer
        from its recorded ``roles``; anything else (no descriptor, the offline default,
        legacy runs) is NON-enforcing — fail-open is safe here because enforcement only
        ever ADDS denials, and the offline path must stay byte-unchanged.
        """
        if not actor or not actor.get("tool_authz_enforce"):
            return cls(enforce=False, roles=frozenset())
        roles = actor.get("roles") or []
        scopes = actor.get("scopes") or []
        return cls(
            enforce=True,
            roles=frozenset(str(r) for r in roles),
            policy=policy,
            scopes=frozenset(str(s) for s in scopes),
        )

    @classmethod
    def from_request(cls, request: Any) -> ToolCapabilityAuthorizer | None:
        """Build the gate for an in-flight HTTP request from its verified principal.

        The request-boundary seam (centralize-tool-gate): the Studio chat / team /
        workflow streaming surfaces and the routine ``agent_path`` seam build their runtime
        WITHOUT threading a ``tool_authorizer`` explicitly — they instead bind THIS gate
        ambiently (:func:`himmy.services.tools.ambient.use_tool_authorizer`) around the run
        drive, so the tool chokepoint enforces the request principal's tool reach without
        every router having to plumb the authorizer through ``build_runtime_for_spec``.

        Returns ``None`` when no authenticator is configured (``app.state.authenticator``
        is ``None`` — the offline / loopback Studio default) so the ambient binding is inert
        and tool dispatch is byte-unchanged. Otherwise it is the verified principal's
        :meth:`from_principal` authorizer over the app's :class:`AccessPolicy` — a
        non-enforcing pass-through for an unrestricted principal, an enforcing
        deny-by-default gate for a tenant-bound one. Mirrors the connector-inbound and
        ``require_permission`` posture exactly: enforcement engages when auth does.
        """
        app = getattr(request, "app", None)
        state = getattr(app, "state", None)
        authenticator = getattr(state, "authenticator", None)
        policy = getattr(state, "access_policy", None)
        if authenticator is None or policy is None:
            return None
        from himmy.api.auth.context import get_principal

        return cls.from_principal(get_principal(request), policy)

    def attenuate(self) -> ToolCapabilityAuthorizer:
        """Return the authorizer a spawned sub-agent inherits (never wider than ``self``).

        A sub-agent runs with the SAME capability set as its parent — capability can only
        ATTENUATE down a spawn chain, never amplify — so this returns ``self`` (the frozen
        authorizer is immutable and shareable). Kept as an explicit seam so the
        "sub ⊆ parent" guarantee is named at every spawn site rather than implied.
        """
        return self

    def is_authorized(self, name: str, read_only: bool | None) -> bool:
        """Whether the principal may invoke tool ``name`` (deny-by-default when enforcing).

        Non-enforcing (offline / unrestricted) → always ``True``. Enforcing → the
        principal must hold ``tool:<name>:invoke`` and, unless the tool is PROVABLY
        read-only, additionally ``tool:<name>:write``. An ``admin``-style ``*:*`` grant
        covers both via the policy's own wildcard matching. With no policy wired (a
        misconfiguration) an enforcing authorizer denies — fail CLOSED.

        **What counts as "provably read-only" (and why name-inference does NOT).**
        ONLY an EXPLICIT author flag ``read_only=True`` waives the ``:write`` sub-grant.
        Name-inference (:func:`classify_read_only`) is deliberately NOT trusted to waive
        it: that classifier is first-token-wins, so a side-effecting tool whose name
        merely STARTS with a read verb — ``get_or_create_invoice`` (first token ``get``),
        ``check_payment`` (``check``) — is inferred read-only even though it mutates state
        (``create`` is literally a later token in the first example). Were inference
        allowed to waive ``:write``, a least-privilege role granted only
        ``tool:<name>:invoke`` (expecting a look-up) could fire that write tool it was
        never meant to. So the gate fails CLOSED for everything that is not EXPLICITLY
        flagged read-only — both an ambiguous name (``process_payment`` →
        :func:`classify_read_only` ``None``) AND a name merely inferred read-only require
        the ``:write`` grant. Name-inference still feeds the model-facing description hint
        (:func:`himmy.services.tools.access.describe_for_model`); it just never relaxes a
        capability check. Tool authors set ``read_only=True`` on genuine look-up tools to
        grant invoke-only reach.
        """
        if not self.enforce:
            return True
        if self.policy is None:
            return False
        if not self._grants(name, INVOKE_ACTION):
            return False
        # Waive the ``:write`` sub-grant ONLY for an EXPLICIT author ``read_only=True``
        # flag — never for a name-inferred verdict (see docstring: first-token-wins
        # mis-classifies read-verb-prefixed writers like ``get_or_create``). So anything
        # whose intent is not the explicit-True flag (ambiguous, inferred-read, or write)
        # additionally requires ``tool:<name>:write`` — fail CLOSED.
        if read_only is not True and not self._grants(name, WRITE_ACTION):
            return False
        return True

    def authorize_definition(self, definition: ToolDefinition) -> bool:
        """Authorize a :class:`ToolDefinition`, reading its read/write intent.

        Only an AUTHORITATIVE author assertion (``read_only=True`` AND
        ``read_only_authoritative``) waives the ``tool:<name>:write`` sub-grant —
        exactly like the retry (:meth:`ToolService._is_retry_side_effect_safe`) and
        parallelism (:func:`himmy.services.inference.local._is_parallel_safe`) gates.
        A method-derived read-only (a bare HTTP ``GET`` whose ``read_only`` is inferred
        ``True`` but whose ``read_only_authoritative`` is ``False``) may still be
        side-effecting (``GET /trigger``), so it does NOT waive ``:write`` — it is
        passed as ``None`` and the capability check fails CLOSED.
        """
        read_only = (
            True
            if (definition.read_only is True and definition.read_only_authoritative)
            else None
        )
        return self.is_authorized(definition.name, read_only)

    def _grants(self, name: str, action: str) -> bool:
        """Whether any of the principal's roles grants ``tool:<name>:<action>``.

        Probes the policy with resource ``tool`` and action ``<name>:<action>`` — the
        exact tuple a policy entry ``"tool:<name>:<action>"`` parses to (first-colon
        split), so the wildcard ``tool:*`` (and ``admin`` ``*:*``) match too.
        """
        assert self.policy is not None  # guarded by is_authorized
        from himmy.api.auth.principal import Principal

        probe = Principal(
            subject="__tool_authz__", roles=self.roles, scopes=self.scopes
        )
        return self.policy.authorize(probe, TOOL_RESOURCE, f"{name}:{action}")


__all__ = [
    "ToolCapabilityAuthorizer",
    "TOOL_RESOURCE",
    "INVOKE_ACTION",
    "WRITE_ACTION",
]
