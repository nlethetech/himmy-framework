"""Canonical truthy parsing for security-relevant environment flags.

WHY this module exists
======================
Boolean posture toggles (``HIMMY_MULTI_TENANT``, ``HIMMY_STUDIO_AUTH``,
``HIMMY_ALLOW_UNAUTHENTICATED``, the spec-sanitizer / fail-closed switches, …) were
historically each parsed *ad hoc* — one site accepted ``("1", "true", "yes")`` while
another accepted ``("1", "true", "yes", "on")``. That divergence is a real bypass
class: an operator who writes ``HIMMY_MULTI_TENANT=on`` would have the fail-closed
*detector* recognise it but a parallel reader silently treat it as *off*, half-honoring
a posture and failing **open** on the gap. (See the comment block in
``himmy/api/app.py`` warning that the posture detector and its downstream consumer must
share one vocabulary — this module IS that shared vocabulary.)

:func:`env_truthy` is the single accepted spelling of "is this flag on?". Every
security/posture flag routes through it so the accepted set can never diverge between
two readers again.

Accepted truthy tokens (case-insensitive, surrounding whitespace trimmed):
``1``, ``true``, ``yes``, ``on``, ``y``. Everything else — including empty/unset,
``0``, ``false``, ``no``, ``off``, ``maybe``, ``2`` — is **false**.

INVARIANT — offline/zero-config path is byte-unchanged: an unset flag returns the
caller's ``default`` (``False`` for every security flag), so a single-box deploy that
sets none of these behaves exactly as before.
"""

from __future__ import annotations

import os

#: The canonical truthy token set. Kept as a frozenset so callers may reuse it for
#: documentation / table tests without re-deriving the vocabulary. Lower-cased, trimmed
#: comparison is done in :func:`truthy`. Superset of the legacy ``("1","true","yes","on")``
#: vocabulary plus ``y`` (the common shorthand) — never narrower, so no previously-honored
#: flag value silently becomes false.
TRUTHY_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on", "y"})

#: The canonical FALSY token set, for explicit ON/OFF switches that default to ON (e.g.
#: ``HIMMY_STUDIO_AUTH``). For such a switch the safe reading is "stay on unless the
#: operator wrote a recognised *off* token" — an unset var OR an unrecognised value must
#: NOT silently disable the guard (a typo'd ``HIMMY_STUDIO_AUTH=of`` keeps auth ON).
#: Mirrors the historical ``("off", "0", "false", "no")`` spelling, plus ``n``/``off``
#: shorthands, so this is never narrower than the value sites it replaces.
FALSY_TOKENS: frozenset[str] = frozenset({"0", "false", "no", "off", "n"})


def truthy(value: str | None, *, default: bool = False) -> bool:
    """Whether a raw string ``value`` denotes a truthy flag.

    Case-insensitive and whitespace-trimmed. ``None`` (or, after trimming, the empty
    string) yields ``default`` so an *unset* flag falls back to the caller's intent
    (``False`` for fail-closed security toggles). Any non-empty token outside
    :data:`TRUTHY_TOKENS` is ``False`` — there is deliberately no "falsy token" list:
    anything we do not positively recognise as on is off, which is the safe default
    for a posture switch.
    """
    if value is None:
        return default
    token = value.strip().lower()
    if not token:
        return default
    return token in TRUTHY_TOKENS


def env_truthy(name: str, *, default: bool = False) -> bool:
    """Whether environment variable ``name`` is set to a truthy value.

    The one canonical reader for every security-relevant boolean env flag. Routes
    through :func:`truthy`, so ``HIMMY_MULTI_TENANT=on`` and ``HIMMY_MULTI_TENANT=yes``
    can never be honored by one call site and ignored by another. An unset variable
    returns ``default`` (``False`` for fail-closed posture flags), preserving the
    offline/zero-config no-op default byte-for-byte.
    """
    return truthy(os.environ.get(name), default=default)


def falsy(value: str | None) -> bool:
    """Whether a raw string ``value`` is an EXPLICIT falsy token.

    For ON/OFF switches that default to ON (so the absence of the flag must keep the
    guard active). Returns ``True`` only for a recognised off token (:data:`FALSY_TOKENS`);
    ``None``, empty, and any unrecognised token return ``False`` so the switch stays in
    its (safe, on) default state rather than being disabled by a typo. Case-insensitive
    and whitespace-trimmed.
    """
    if value is None:
        return False
    return value.strip().lower() in FALSY_TOKENS


def env_falsy(name: str) -> bool:
    """Whether environment variable ``name`` is set to an EXPLICIT falsy token.

    The canonical reader for default-ON kill-switches such as ``HIMMY_STUDIO_AUTH``:
    only a recognised off token disables the guard; unset or unrecognised keeps it on.
    """
    return falsy(os.environ.get(name))
