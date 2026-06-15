"""Role-based access control for the himmy CLI — real enforcement, not cosmetic.

The framework already carries the RBAC primitives used by the HTTP surface
(:mod:`himmy.api.auth.rbac`): an :class:`~himmy.api.auth.rbac.AccessPolicy` maps
roles to ``(resource, action)`` permissions (``*`` wildcards allowed) and a
:class:`~himmy.api.auth.principal.Principal` carries the caller's roles. This
module reuses *those same* primitives so the CLI enforces access with identical
semantics to the server — no parallel if/else ladder.

A local single-user CLI defaults to ``admin`` (full access) so nothing changes
for existing users; roles only ever **restrict**, and only once one is
configured. The active role is resolved by precedence::

    --role flag  >  HIMMY_ROLE env  >  himmy.toml [rbac] role  >  "admin"

The policy comes from :func:`~himmy.api.auth.rbac.build_access_policy` (which
honours ``HIMMY_RBAC_FILE``); absent a custom file we fall back to a CLI-shaped
default catalogue (:data:`CLI_RBAC`) that maps the three CLI roles —
``admin`` / ``operator`` / ``viewer`` — onto CLI resources (run, chat, tools,
code, comms, …). :func:`gate_tool_for_role` is the enforcement hook: it turns a
``(principal, policy, tool, requires_approval)`` question into
``"allow" | "prompt" | "deny"`` so a viewer hitting a gated tool fails closed.

CLI resource vocabulary (the ``(resource, action)`` pairs enforcement actually
checks — distinct from the SERVER's resource names in :mod:`himmy.api.auth.rbac`).
A custom ``HIMMY_RBAC_FILE`` for the CLI should grant these:

* ``tool:execute``        — run a PLAIN, never-gated tool (calculator, current_time,
  read_file). A role WITHOUT this loses even ungated tools.
* ``tool_gated:execute``  — run an APPROVAL-GATED tool (it still prompts unless the
  role is unrestricted). A role without this is DENIED gated tools (fail closed).
* ``run:execute`` / ``chat:execute`` — drive the agent loop / interactive chat.
* ``inspect:read``        — read-only introspection (/tools, /why, /cost, /whoami).
* ``code:execute`` / ``comms:execute`` / ``telegram:execute`` — shell-exec / outbound
  messaging surfaces.

So a read-only ``viewer`` grants ``tool:execute`` (keeps ungated tools) + the
``read`` actions, but NOT ``tool_gated:execute`` (no approval-gated execution).
"""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from himmy.api.auth.principal import Principal
from himmy.api.auth.rbac import AccessPolicy, build_access_policy
from himmy.cli.ui import styles
from himmy.config.project import find_project_root

#: The CLI role catalogue, expressed as DATA over ``(resource, action)`` perms.
#:
#: * ``admin`` — unrestricted (``*:*``), the local single-user default.
#: * ``operator`` — runs/chats and may execute tools (including ones that need
#:   approval, which still *prompt*), but may not *manage* the agent surface
#:   (mcp / rbac config).
#: * ``viewer`` — read-only: run/chat/inspect, but any tool that ``requires
#:   approval`` is DENIED, and the code / comms / telegram resources are off.
#:
#: Resources surfaced to the CLI: ``run`` (agent loop), ``chat``, ``tool``
#: (generic tool execution), ``tool_gated`` (a tool that requires approval),
#: ``code`` (shell/exec tools), ``comms``/``telegram`` (outbound messaging),
#: ``inspect`` (read-only introspection: /tools, /why, /cost), ``mcp`` and
#: ``rbac`` (management). Actions are ``read`` / ``execute`` / ``manage``.
CLI_RBAC: dict[str, list[str]] = {
    "admin": ["*:*"],
    "operator": [
        "run:execute",
        "chat:execute",
        "inspect:read",
        "tool:execute",
        "tool_gated:execute",
        "code:execute",
        "comms:execute",
        "telegram:execute",
    ],
    "viewer": [
        "run:read",
        "chat:read",
        "inspect:read",
        "tool:execute",
    ],
}

#: A CLI-shaped policy built from :data:`CLI_RBAC`, used when no custom
#: ``HIMMY_RBAC_FILE`` is configured.
CLI_DEFAULT_POLICY = AccessPolicy.from_mapping(CLI_RBAC)

#: The role used when none is configured anywhere (full local access).
DEFAULT_ROLE = "admin"


def _toml_role(start: str | Path | None = None) -> str | None:
    """The ``[rbac] role`` from the project's ``himmy.toml`` (project root), or ``None``.

    Mirrors :func:`himmy.cli.permissions.load_auto_approve`: reads only the local
    ``himmy.toml`` and never crashes on a missing/odd file. Resolves the file via
    :func:`~himmy.config.project.find_project_root` so a run launched from a
    SUBDIRECTORY still enforces the project's configured role instead of silently
    falling back to ``admin``.
    """
    path = find_project_root(start) / "himmy.toml"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    table = data.get("rbac")
    if not isinstance(table, dict):
        return None
    role = table.get("role")
    return role.strip() if isinstance(role, str) and role.strip() else None


def cli_principal(args: Any = None, *, start: str | Path | None = None) -> Principal:
    """Resolve the active CLI caller as a :class:`Principal`.

    Role precedence: ``--role`` flag > ``HIMMY_ROLE`` env > ``himmy.toml``
    ``[rbac] role`` > :data:`DEFAULT_ROLE` (``admin``). The resulting principal
    carries exactly that one role; ``admin`` additionally gets ``all_tenants``
    (unrestricted), matching the offline-first single-user default.
    """
    role: str | None = None
    flag = getattr(args, "role", None) if args is not None else None
    if isinstance(flag, str) and flag.strip():
        role = flag.strip()
    if role is None:
        env = os.environ.get("HIMMY_ROLE")
        if env and env.strip():
            role = env.strip()
    if role is None:
        role = _toml_role(start)
    if role is None:
        role = DEFAULT_ROLE
    return Principal.build(
        subject=f"cli:{role}",
        roles=[role],
        all_tenants=(role == "admin"),
        auth_method="cli",
    )


def cli_access_policy() -> AccessPolicy:
    """The active CLI policy: a custom ``HIMMY_RBAC_FILE`` if set, else CLI default.

    Delegates to :func:`himmy.api.auth.rbac.build_access_policy`, which loads
    ``HIMMY_RBAC_FILE`` when present. When it is *not* present that helper returns
    the server-shaped ``DEFAULT_POLICY``, which has no CLI roles, so we substitute
    the CLI-shaped :data:`CLI_DEFAULT_POLICY` instead.
    """
    if os.environ.get("HIMMY_RBAC_FILE"):
        return build_access_policy()
    return CLI_DEFAULT_POLICY


def cli_permits(
    principal: Principal, policy: AccessPolicy, resource: str, action: str
) -> bool:
    """The single authorization check — delegates to the real policy logic.

    Thin wrapper over :meth:`AccessPolicy.authorize` (which uses the framework's
    wildcard-aware ``_covers``) so callers never re-implement the check.
    """
    return policy.authorize(principal, resource, action)


def gate_tool_for_role(
    principal: Principal,
    policy: AccessPolicy,
    tool_name: str,
    requires_approval: bool,
) -> str:
    """Decide a gated tool's fate for the caller's role: allow / prompt / deny.

    The enforcement hook the approval path consults *before* its existing
    ``requires_approval`` prompt:

    * ``"deny"``  — the role may not run this tool at all (fail closed). A viewer
      hitting an approval-gated tool lands here.
    * ``"prompt"`` — the role may run it but it needs approval (the normal flow):
      the integration shows its usual prompt.
    * ``"allow"`` — run it without a prompt (admin / a tool that needs no
      approval / an allowlisted tool whose ``requires_approval`` was flipped off).

    The decision is computed from the REAL :class:`AccessPolicy`: an
    approval-gated tool is checked against ``tool_gated:execute`` and a plain tool
    against ``tool:execute`` (``admin``'s ``*:*`` covers both). It never hardcodes
    role names.
    """
    # The permission a gated vs. plain tool demands.
    resource = "tool_gated" if requires_approval else "tool"
    if not cli_permits(principal, policy, resource, "execute"):
        return "deny"
    if not requires_approval:
        return "allow"
    # The tool is gated AND the role may run it. An *unrestricted* role (one
    # whose policy holds the ``*:*`` wildcard — admin) bypasses the prompt; a
    # role that holds only ``tool_gated:execute`` (operator) still prompts.
    # Probed via the real policy against a sentinel resource so admin's wildcard
    # is what grants the bypass, not a hardcoded role name.
    if _is_unrestricted(principal, policy):
        return "allow"
    return "prompt"


def _is_unrestricted(principal: Principal, policy: AccessPolicy) -> bool:
    """Whether the principal holds the ``*:*`` wildcard (admin-equivalent).

    Probes the REAL policy with a sentinel ``(resource, action)`` that appears in
    no explicit grant: only a ``*:*`` wildcard can authorize it.
    """
    return cli_permits(principal, policy, "__himmy_probe__", "__himmy_probe__")


def born_gated_names(registry: Any) -> set[str]:
    """The set of tools that are approval-gated AS BUILT (before any ungating).

    A snapshot of every :class:`ToolDefinition` whose ``requires_approval`` is on at
    registry-build time, *before* the permission profile (allowlist / --yolo) flips
    any off. :func:`enforce_role_on_registry` needs this to tell an inherently
    approval-gated tool (gate against ``tool_gated``) from a plain util/read tool
    (gate against ``tool``): once the allowlist has ungated a born-gated tool the live
    ``requires_approval`` no longer reveals which it was. Captured BEFORE
    ``apply_allowlist``/``grant_all_approvals`` run, then handed to
    ``enforce_role_on_registry``.
    """
    if registry is None:
        return set()
    return {d.name for d in registry.list() if d.requires_approval}


def enforce_role_on_registry(
    registry: Any,
    *,
    role: str | None,
    on_line: Any = None,
    born_gated: set[str] | None = None,
) -> list[str]:
    """Hard-deny a restricted role's tools by re-gating only the ones it may not run.

    Applied AFTER the permission profile (allowlist / --yolo / --safe) so a
    restricted role can never run a gated tool it lacks — even one that was
    allowlisted or --yolo'd. Each tool is gated against the permission it really
    demands: a BORN-gated tool (in ``born_gated``, captured before any ungating)
    against ``tool_gated:execute``; a plain util/read tool against ``tool:execute``.
    So a viewer whose policy grants ``tool:execute`` KEEPS plain tools (calculator,
    current_time, read_file) and only loses approval-gated execution.

    For each tool the resolved principal may NOT run (``gate_tool_for_role`` →
    ``deny``), ``requires_approval`` is forced back ON so the non-interactive path
    fails it closed (POLICY_BLOCKED) and the interactive path is denied at the prompt.
    Returns the names re-gated; a no-op for ``admin`` (the unrestricted default), so
    existing single-user sessions are unaffected.

    When ``born_gated`` is not supplied it falls back to the registry's CURRENT
    ``requires_approval`` — correct unless an allowlist already ungated a born-gated
    tool, which the callers avoid by snapshotting before the profile runs.
    """
    if registry is None:
        return []
    principal = cli_principal(argparse.Namespace(role=role))
    policy = cli_access_policy()
    if _is_unrestricted(principal, policy):
        return []  # admin / unrestricted: nothing to restrict
    if born_gated is None:
        born_gated = {d.name for d in registry.list() if d.requires_approval}
    denied_gated: list[str] = []
    denied_plain: list[str] = []
    for definition in registry.list():
        was_gated = definition.name in born_gated
        # Gate each tool against the permission it genuinely demands: born-gated →
        # tool_gated:execute; plain util/read → tool:execute. A viewer that holds
        # tool:execute keeps plain tools and only loses approval-gated ones.
        if gate_tool_for_role(principal, policy, definition.name, was_gated) != "deny":
            continue
        if not definition.requires_approval:
            definition.requires_approval = True
        (denied_gated if was_gated else denied_plain).append(definition.name)
    if on_line is not None:
        if denied_gated:
            on_line(
                deny_message(principal, ", ".join(sorted(denied_gated)), gated=True)
            )
        if denied_plain:
            on_line(
                deny_message(principal, ", ".join(sorted(denied_plain)), gated=False)
            )
    # Record each denial to the durable security spine (fail open — never gates).
    for name in (*denied_gated, *denied_plain):
        _record_authz_denied(principal, name)
    return [*denied_gated, *denied_plain]


def _record_authz_denied(principal: Principal, tool_name: str) -> None:
    """Persist one ``authz_denied`` security event for a re-gated tool (fail open)."""
    from himmy.cli.security_audit_cmd import record_security_event

    role = next(iter(sorted(principal.roles)), DEFAULT_ROLE)
    record_security_event(
        event_type="authz_denied",
        outcome="deny",
        actor={"subject": role, "roles": sorted(principal.roles), "auth_method": "cli"},
        action="tool_call",
        resource=tool_name,
        detail="rbac re-gated: role may not run this tool",
    )


def deny_message(
    principal: Principal, tool_name: str, *, gated: bool = True, stream: Any = None
) -> str:
    """A styled 'role may not run <tool>' line.

    ``gated`` distinguishes the two real reasons a tool is denied: an approval-gated
    tool the role lacks ``tool_gated:execute`` for (the usual case), vs. a plain
    tool the role is denied even ``tool:execute`` for. The message never calls a
    plain tool "approval-gated".
    """
    c = styles(stream)
    role = next(iter(sorted(principal.roles)), DEFAULT_ROLE)
    why = (
        "approval-gated; raise your role to operator/admin"
        if gated
        else "this role lacks tool:execute"
    )
    return (
        f"{c['crimson']}✗ role {role!r} may not run {tool_name}{c['reset']} "
        f"{c['faint']}({why}){c['reset']}"
    )


def _granted_summary(policy: AccessPolicy, role: str) -> list[str]:
    """A sorted, human-readable list of permissions a role grants."""
    perms = policy.role_permissions.get(role)
    if not perms:
        return []
    return sorted(f"{r}:{a}" for (r, a) in perms)


def cmd_whoami(args: argparse.Namespace) -> int:
    """``himmy whoami`` — print the active role and the permissions it grants."""
    out = sys.stdout
    c = styles(out)
    principal = cli_principal(args)
    policy = cli_access_policy()
    role = next(iter(sorted(principal.roles)), DEFAULT_ROLE)

    grants = _granted_summary(policy, role)
    print(f"{c['snow']}role:{c['reset']} {c['gold']}{role}{c['reset']}", file=out)
    if role == "admin" or any(g == "*:*" for g in grants):
        print(
            f"{c['green']}grants:{c['reset']} {c['ink']}* (unrestricted){c['reset']}",
            file=out,
        )
    elif not grants:
        print(
            f"{c['crimson']}grants:{c['reset']} {c['faint']}(none — this role is "
            f"not in the active policy){c['reset']}",
            file=out,
        )
    else:
        print(f"{c['green']}grants:{c['reset']}", file=out)
        for perm in grants:
            print(f"  {c['ink']}· {perm}{c['reset']}", file=out)

    # A practical line: can this role run an approval-gated tool?
    gated = gate_tool_for_role(principal, policy, "<gated-tool>", True)
    verb = {"allow": "run", "prompt": "run (with approval)", "deny": "NOT run"}[gated]
    tone = {"allow": c["green"], "prompt": c["gold"], "deny": c["crimson"]}[gated]
    print(
        f"{c['faint']}gated tools: {tone}{verb}{c['reset']}",
        file=out,
    )
    return 0


__all__ = [
    "CLI_RBAC",
    "CLI_DEFAULT_POLICY",
    "DEFAULT_ROLE",
    "cli_principal",
    "cli_access_policy",
    "cli_permits",
    "gate_tool_for_role",
    "enforce_role_on_registry",
    "deny_message",
    "cmd_whoami",
]
