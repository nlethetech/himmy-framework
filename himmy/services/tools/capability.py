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

from himmy.services.tools.access import classify_read_only

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

    @classmethod
    def from_principal(
        cls, principal: Principal | None, policy: AccessPolicy | None
    ) -> ToolCapabilityAuthorizer:
        """Build an authorizer for ``principal`` under ``policy``.

        Mirrors :func:`himmy.api.auth.rbac.require_permission`'s bypass exactly: a
        ``None`` principal (no auth seam) or an unrestricted ``all_tenants`` principal
        (the ANONYMOUS offline default, or a trusted shared key) yields a NON-enforcing
        authorizer — every tool is allowed, byte-identical to before. A tenant-bound
        principal yields an enforcing one carrying its roles + the policy.
        """
        if principal is None or principal.all_tenants:
            return cls(enforce=False, roles=frozenset())
        return cls(enforce=True, roles=frozenset(principal.roles), policy=policy)

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
        return cls(enforce=True, roles=frozenset(str(r) for r in roles), policy=policy)

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
        read-only (``read_only`` is ``True``, or inferred ``True`` from the name),
        additionally ``tool:<name>:write`` — so an ambiguously-named tool whose intent
        cannot be inferred fails CLOSED to writer (requires the write grant too). An
        ``admin``-style ``*:*`` grant covers both via the policy's own wildcard matching.
        With no policy wired (a misconfiguration) an enforcing authorizer denies — fail
        CLOSED.
        """
        if not self.enforce:
            return True
        if self.policy is None:
            return False
        if not self._grants(name, INVOKE_ACTION):
            return False
        intent = read_only if read_only is not None else classify_read_only(name)
        # Fail CLOSED on the write sub-grant: require ``tool:<name>:write`` for anything
        # NOT provably read-only. ``intent`` is ``True`` only for a tool flagged
        # ``read_only=True`` or inferred read-only from its name; an AMBIGUOUS name
        # (``classify_read_only`` -> ``None``, e.g. ``process_payment``, ``submit_order``)
        # is treated as a writer so an operator granting a per-tool ``:invoke`` for a
        # read-only reach cannot unknowingly hand a writer write reach. Tool authors should
        # set explicit ``read_only=True`` on look-up tools to avoid needing the write grant.
        if intent is not True and not self._grants(name, WRITE_ACTION):
            return False
        return True

    def authorize_definition(self, definition: ToolDefinition) -> bool:
        """Authorize a :class:`ToolDefinition`, reading its read/write intent."""
        return self.is_authorized(definition.name, definition.read_only)

    def _grants(self, name: str, action: str) -> bool:
        """Whether any of the principal's roles grants ``tool:<name>:<action>``.

        Probes the policy with resource ``tool`` and action ``<name>:<action>`` — the
        exact tuple a policy entry ``"tool:<name>:<action>"`` parses to (first-colon
        split), so the wildcard ``tool:*`` (and ``admin`` ``*:*``) match too.
        """
        assert self.policy is not None  # guarded by is_authorized
        from himmy.api.auth.principal import Principal

        probe = Principal(subject="__tool_authz__", roles=self.roles)
        return self.policy.authorize(probe, TOOL_RESOURCE, f"{name}:{action}")


__all__ = [
    "ToolCapabilityAuthorizer",
    "TOOL_RESOURCE",
    "INVOKE_ACTION",
    "WRITE_ACTION",
]
