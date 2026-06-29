"""Ambient tool-capability gate: the authorizer that follows execution everywhere.

**Why this exists (the by-construction failure mode it kills).** Tool-capability
enforcement lives at a single chokepoint —
:meth:`himmy.services.tools.service.ToolService._execute_invocation` — but it only fires
if THAT particular :class:`~himmy.services.tools.service.ToolService` instance was
constructed with a ``tool_authorizer``. Every execution path that builds a runtime
(``/v1`` single-agent, multi-agent orchestration, skill dispatch, toolkit spawn, durable
recovery, connector inbound, scheduled routines, Studio chat / team / workflow routers)
therefore has to REMEMBER to thread the authorizer down to that constructor. Most do; a
few historically did not (the Studio routers, the routine ``agent_path`` seam), leaving a
confused-deputy hole where a least-privilege caller could invoke any tool the agent
declares. A new path that forgets the thread re-opens the class of bug.

This module removes the requirement to remember. An **ambient authorizer** is set on a
:class:`contextvars.ContextVar` at exactly one place per principal scope (the run-drive
entry, the request boundary, the scheduler). Because ``contextvars`` propagate into
``asyncio.to_thread`` and ``loop.create_task`` (which copy the current context), the
ambient authorizer reaches the off-loop ``build_runtime_for_spec`` and the background
run-task the framework already uses. The chokepoint then resolves the *effective*
authorizer from BOTH the explicit constructor arg AND the ambient var, so a
``ToolService`` built by a site that forgot the arg still consults the gate.

**Precedence & the no-amplification guarantee.** The effective gate is the INTERSECTION
of the explicit and ambient authorizers (:func:`resolve_effective_authorizer`): a tool is
permitted only if BOTH allow it. This keeps two invariants simultaneously:

* a sub-agent's *attenuated* explicit authorizer can only narrow the ambient one (it can
  never reach a tool the ambient gate denies), so attenuation still holds; and
* a buggy site passing a WIDER explicit authorizer than the ambient one cannot AMPLIFY
  past the ambient principal's reach (the intersection still denies what ambient denies).

**Fail-closed when auth is configured.** :func:`mark_auth_configured` records, process
-wide, that an authenticator is wired. When it is, and a tool reaches the chokepoint with
NEITHER an explicit NOR an ambient authorizer in scope, the gate DENIES (treats it as
enforce-with-no-grants) instead of passing — so a future execution path that forgets both
the arg and the contextvar fails closed rather than silently un-gated. This is the
``DENY_ALL`` sentinel.

**Offline invariant — byte-unchanged.** When no authenticator is configured,
``auth_is_configured()`` is ``False``, the contextvar default is ``None``, and a site that
passes no explicit authorizer resolves to an effective authorizer of ``None`` — the
chokepoint's pre-existing ``if self._tool_authorizer is not None`` short-circuits to a
pure pass, exactly as before. The contextvar is never set on CLI / eval / benchmark /
offline-Studio paths (they configure no authenticator), so those stay a strict no-op and
the historical single-box path is untouched. A NON-enforcing ANONYMOUS / all_tenants
ambient authorizer (``enforce=False``) is likewise an unconditional pass.
"""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.services.tools.capability import ToolCapabilityAuthorizer
    from himmy.services.tools.models import ToolDefinition

#: The ambient authorizer for the current execution context. ``None`` (the default, and
#: every offline / CLI / eval path that never sets it) means "no ambient gate" — the
#: chokepoint falls back to the explicit constructor arg, which is itself ``None`` offline,
#: yielding the byte-unchanged pure-pass behaviour.
_active_authorizer: contextvars.ContextVar[ToolCapabilityAuthorizer | None] = (
    contextvars.ContextVar("himmy_tool_authorizer", default=None)
)

#: Process-level flag: has an authenticator been configured on this server? Set once at
#: app startup (:func:`mark_auth_configured`). When ``True``, a tool reaching the
#: chokepoint with no authorizer at all (explicit or ambient) FAILS CLOSED — the
#: fail-closed-by-default that makes a forgotten future path safe by construction. It is a
#: module-global rather than a contextvar because it describes the DEPLOYMENT posture, not
#: a per-request scope, and must hold for the off-request scheduler/recovery paths too.
_auth_configured: bool = False


def mark_auth_configured(configured: bool) -> None:
    """Record whether this process runs under a configured authenticator (startup).

    Called once when the API app is built (see ``himmy.api.app.create_app``) with
    ``app.state.authenticator is not None``. Flipping it to ``True`` engages the
    fail-closed-when-unscoped default at the tool chokepoint; the offline default
    (``False``) keeps the byte-unchanged pure-pass behaviour. Idempotent and safe to call
    repeatedly (e.g. several apps in one test process) — the LAST configured app wins,
    which for the single live server is the correct posture.
    """
    global _auth_configured
    _auth_configured = configured


def auth_is_configured() -> bool:
    """Whether an authenticator is configured process-wide (fail-closed posture on)."""
    return _auth_configured


def current_authorizer() -> ToolCapabilityAuthorizer | None:
    """The ambient authorizer for the current context, or ``None`` if none is set."""
    return _active_authorizer.get()


@contextlib.contextmanager
def use_tool_authorizer(
    authorizer: ToolCapabilityAuthorizer | None,
) -> Iterator[None]:
    """Bind ``authorizer`` as the ambient tool gate for the duration of the block.

    Sets the contextvar on entry and RESETS it on exit (even on exception), so the binding
    is strictly scoped — it leaks into ``asyncio.to_thread`` / ``create_task`` children
    (which copy the context, the propagation this design relies on) but never outlives the
    ``with`` block in the parent. Binding ``None`` is a legitimate, explicit "no ambient
    gate" scope (the offline path passes ``None`` and stays inert).
    """
    token = _active_authorizer.set(authorizer)
    try:
        yield
    finally:
        _active_authorizer.reset(token)


def resolve_effective_authorizer(
    explicit: ToolCapabilityAuthorizer | None,
) -> ToolCapabilityAuthorizer | None:
    """The authorizer the chokepoint must actually enforce, given the explicit arg.

    Combines the per-``ToolService`` ``explicit`` arg with the :data:`ambient
    <_active_authorizer>` one so NO construction site can run a tool un-gated when an
    authorizer is in scope:

    * **both present** → an :class:`_IntersectionAuthorizer` that permits a tool only if
      BOTH allow it (defense-in-depth: an attenuated sub-agent explicit gate can only
      narrow the ambient one, and a buggy wider explicit gate cannot amplify past it);
    * **only one present** → that one (the explicit arg WINS as the sole gate when no
      ambient is set, preserving today's explicit-threading behaviour, and vice-versa);
    * **neither present** → ``None`` offline (pure pass, byte-unchanged), but the
      :data:`DENY_ALL` fail-closed sentinel when :func:`auth_is_configured` — so a path
      that forgot BOTH the arg and the contextvar denies under a configured deployment
      instead of silently running un-gated.
    """
    ambient = _active_authorizer.get()
    if explicit is None and ambient is None:
        if _auth_configured:
            return DENY_ALL
        return None
    if explicit is None:
        return ambient
    if ambient is None:
        return explicit
    if explicit is ambient:
        return explicit
    return _IntersectionAuthorizer(explicit, ambient)


class _DenyAllAuthorizer:
    """The fail-closed sentinel: denies EVERY tool (enforce-with-no-grants).

    Returned by :func:`resolve_effective_authorizer` only when auth is configured AND no
    authorizer (explicit or ambient) is in scope — i.e. a path that forgot to thread the
    gate under a configured deployment. Quacks like a
    :class:`~himmy.services.tools.capability.ToolCapabilityAuthorizer` (``enforce``,
    ``authorize_definition``, ``is_authorized``, ``attenuate``) so the chokepoint and the
    sub-agent spawn sites treat it uniformly. ``attenuate`` returns itself — a deny-all
    cannot narrow to anything but deny-all.
    """

    enforce = True

    def is_authorized(self, name: str, read_only: bool | None) -> bool:  # noqa: ARG002
        return False

    def authorize_definition(self, definition: ToolDefinition) -> bool:  # noqa: ARG002
        return False

    def attenuate(self) -> _DenyAllAuthorizer:
        return self


#: Singleton fail-closed gate (see :class:`_DenyAllAuthorizer`).
DENY_ALL = _DenyAllAuthorizer()


class _IntersectionAuthorizer:
    """A gate that permits a tool only if BOTH wrapped authorizers permit it.

    The defense-in-depth combination of an explicit (per-``ToolService``) authorizer and
    the ambient (context) one. ``enforce`` is ``True`` if EITHER enforces (so a
    non-enforcing offline gate intersected with an enforcing ambient one still enforces —
    the strict side wins), and a tool is authorized only when BOTH say yes. Frozen-style:
    holds references, never mutates. ``attenuate`` intersects the attenuated children, so a
    sub-agent inheriting an intersection still cannot amplify past either parent.
    """

    __slots__ = ("_a", "_b")

    def __init__(
        self,
        a: ToolCapabilityAuthorizer,
        b: ToolCapabilityAuthorizer,
    ) -> None:
        self._a = a
        self._b = b

    @property
    def enforce(self) -> bool:
        return bool(getattr(self._a, "enforce", False)) or bool(
            getattr(self._b, "enforce", False)
        )

    def is_authorized(self, name: str, read_only: bool | None) -> bool:
        return self._a.is_authorized(name, read_only) and self._b.is_authorized(
            name, read_only
        )

    def authorize_definition(self, definition: ToolDefinition) -> bool:
        return self._a.authorize_definition(definition) and self._b.authorize_definition(
            definition
        )

    def attenuate(self) -> _IntersectionAuthorizer:
        return _IntersectionAuthorizer(self._a.attenuate(), self._b.attenuate())


__all__ = [
    "use_tool_authorizer",
    "current_authorizer",
    "resolve_effective_authorizer",
    "mark_auth_configured",
    "auth_is_configured",
    "DENY_ALL",
]
