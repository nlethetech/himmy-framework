"""Sanitize a tenant-submitted :class:`AgentSpec` against the RCE/SSRF surface (T0.3).

A multi-tenant ``/v1`` deployment accepts ``AgentSpec`` rows from untrusted callers
(the ``/v1/agents`` store, a team member, a routine). Three spec fields are
operator-owned attack surfaces and MUST NOT be honored from a tenant request:

* ``tools_module`` — :func:`himmy.runtime.from_spec.resolve_tools_module` does
  ``importlib.import_module`` on a tenant-controlled dotted path and *calls* its
  ``register`` attribute (arbitrary code execution).
* ``http_tools`` — declarative server-side HTTP egress the runtime makes on the
  tenant's behalf (server-side request forgery toward internal services).
* ``mcp_servers`` — each entry spawns a stdio subprocess (``command``/``args``),
  i.e. tenant-driven arbitrary process spawn.

The CLI and Himmy Studio are single-user-local: the operator *is* the caller, so
they keep full power (this module is never applied there). The choke point is the
``/v1`` write path: a tenant spec carrying any of these three fields is either
rejected (the default, fail-closed) or stripped with a recorded warning, unless the
spec is *operator-provisioned* (an explicit trust signal — see
:func:`operator_specs_allowed`). The same single validator is reused by every ``/v1``
path that accepts a spec, so there is one choke point, not per-route ad-hoc checks.

Offline note: with no authenticator configured (the zero-config default) every
caller is the unrestricted operator, so ``operator==everyone`` and the sanitizer is
effectively inert for a local deployment — exactly the standalone-CLI posture. It
only bites once an authenticator (and thus a tenant boundary) is configured.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from himmy.config.agent_spec import AgentSpec
from himmy.core.errors import HimmyError

#: The three operator-owned fields stripped/rejected from a tenant-submitted spec.
#: ``tools_module`` = arbitrary import+call (RCE); ``http_tools`` = server-side HTTP
#: egress (SSRF); ``mcp_servers`` = stdio subprocess spawn.
PRIVILEGED_SPEC_FIELDS: tuple[str, ...] = ("tools_module", "http_tools", "mcp_servers")

#: Env flag (truthy) that lets the operator opt a /v1 deployment into ACCEPTING the
#: privileged fields from a spec marked operator-provisioned. Default off ⇒ even an
#: operator-provisioned spec is sanitized unless this is set, so the dangerous fields
#: cannot be reached by configuration accident.
_OPERATOR_SPECS_FLAG = "HIMMY_ALLOW_OPERATOR_SPEC_TOOLS"

#: Env flag (truthy) that makes the sanitizer STRIP the privileged fields (recording
#: a warning) instead of REJECTING the whole write. Default off ⇒ fail-closed reject,
#: which is the safer posture for an untrusted multi-tenant surface.
_STRIP_FLAG = "HIMMY_SANITIZE_SPEC_STRIP"


class SpecSanitizationError(HimmyError):
    """A tenant-submitted spec carried a privileged field and was rejected (T0.3)."""

    def __init__(self, fields: list[str]) -> None:
        """Record exactly which privileged fields tripped the rejection."""
        self.fields = list(fields)
        joined = ", ".join(self.fields)
        super().__init__(
            f"agent spec carries operator-only field(s) not permitted for a "
            f"tenant-submitted spec: {joined} "
            f"(tools_module=import+call, http_tools=server-side HTTP, "
            f"mcp_servers=subprocess spawn). Provision the spec as operator or remove "
            f"these fields."
        )


@dataclass(frozen=True)
class SanitizationResult:
    """The outcome of sanitizing one spec: the (possibly stripped) spec + a report."""

    spec: AgentSpec
    #: Privileged fields that were present on the input (whether kept or stripped).
    flagged: list[str] = field(default_factory=list)
    #: True when the privileged fields were stripped rather than kept.
    stripped: bool = False
    #: True when the spec was operator-provisioned and the fields were kept as-is.
    operator_provisioned: bool = False

    @property
    def changed(self) -> bool:
        """Whether the returned spec differs from the input (fields were stripped)."""
        return self.stripped and bool(self.flagged)


def _truthy(name: str) -> bool:
    """Whether an env flag is set to a truthy value (1/true/yes/on)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def operator_specs_allowed() -> bool:
    """Whether operator-provisioned specs may carry the privileged fields.

    Gated on :data:`_OPERATOR_SPECS_FLAG` so the dangerous fields stay unreachable
    unless an operator has *explicitly* opted in, even for a spec the caller asserts
    is operator-provisioned. Default ``False``.
    """
    return _truthy(_OPERATOR_SPECS_FLAG)


def flagged_fields(spec: AgentSpec) -> list[str]:
    """Return the privileged fields actually populated on ``spec`` (order-stable).

    A field counts only when it carries a value: ``tools_module`` non-empty, or a
    non-empty ``http_tools`` / ``mcp_servers`` list. An empty/default field is not a
    surface and is never flagged.
    """
    present: list[str] = []
    if spec.tools_module:
        present.append("tools_module")
    if spec.http_tools:
        present.append("http_tools")
    if spec.mcp_servers:
        present.append("mcp_servers")
    return present


def sanitize_tenant_spec(
    spec: AgentSpec,
    *,
    operator_provisioned: bool = False,
    strip: bool | None = None,
) -> SanitizationResult:
    """Validate/strip the privileged fields from a tenant-submitted spec (T0.3).

    Behaviour:

    * No privileged field present ⇒ the spec is returned unchanged (no-op for the
      overwhelmingly common case; an inline persona / tool_pack-only spec is never
      touched).
    * ``operator_provisioned`` AND :func:`operator_specs_allowed` ⇒ the fields are
      KEPT (an operator may wire its own tools/http/mcp); the result records them as
      ``flagged`` for audit but leaves the spec byte-identical.
    * Otherwise the privileged fields are a tenant attack surface. With
      ``strip=True`` (or :data:`_STRIP_FLAG` set) they are cleared and the sanitized
      copy is returned (``stripped=True``); with ``strip=False`` (the default,
      fail-closed) a :class:`SpecSanitizationError` is raised so the write is
      rejected outright.

    The spec is never mutated in place — a stripped result is a ``model_copy`` so the
    caller's input object is preserved.
    """
    present = flagged_fields(spec)
    if not present:
        return SanitizationResult(spec=spec)

    if operator_provisioned and operator_specs_allowed():
        return SanitizationResult(
            spec=spec, flagged=present, stripped=False, operator_provisioned=True
        )

    do_strip = _truthy(_STRIP_FLAG) if strip is None else strip
    if not do_strip:
        raise SpecSanitizationError(present)

    cleaned = spec.model_copy(
        update={"tools_module": None, "http_tools": [], "mcp_servers": []}
    )
    return SanitizationResult(spec=cleaned, flagged=present, stripped=True)


__all__ = [
    "PRIVILEGED_SPEC_FIELDS",
    "SpecSanitizationError",
    "SanitizationResult",
    "operator_specs_allowed",
    "flagged_fields",
    "sanitize_tenant_spec",
]
