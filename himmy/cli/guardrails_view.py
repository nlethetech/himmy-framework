"""``himmy guardrails`` + the /guardrails REPL view + the live shield line.

Guardrails are the framework's safety layer (PII/Nepal-PII redaction, prompt-injection
blocking, grounding enforcement, DLP). They are most valuable when you can *see* them
work: for a trust interface, a guardrail firing must be surfaced, not buried. This module
turns that surfacing into three things, all reusing the desk-film palette from
:mod:`himmy.cli.ui`:

* :func:`cmd_guardrails` — the ``himmy guardrails`` subcommand: list every built-in
  guardrail (``pii``/``injection``/``nepal_pii``/``grounding``/``dlp``) with a one-line
  description, styled, with ``--json`` for scripts.
* :func:`render_guardrail_event` — formats a ``GUARDRAIL_APPLIED`` :class:`RunEvent` as a
  faint one-line shield (``🛡 guardrail pii redacted: email``) that :class:`LiveRunUI`
  emits the moment a guardrail fires.
* :func:`slash_guardrails` — the ``/guardrails`` REPL slash: which guardrails the current
  agent has active (``spec.guardrails``) and how many fired this session (counted from the
  turn recorder's events).

Everything degrades cleanly on non-TTY/piped/CI streams (no color, no hangs).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from himmy.cli.ui import styles
from himmy.services.guardrails.builtins import BUILTIN_GUARDRAILS

#: Stable display order for the built-in guardrails (matches the prompt's listing).
_ORDER = ["pii", "injection", "nepal_pii", "grounding", "dlp"]


def _eprint(*args: Any) -> None:
    """Diagnostics → stderr, so stdout stays clean for scripts/pipes."""
    print(*args, file=sys.stderr)


def _describe(name: str) -> str:
    """A one-line description for a built-in guardrail (its class docstring's first line)."""
    cls = BUILTIN_GUARDRAILS.get(name)
    doc = (cls.__doc__ or "").strip() if cls is not None else ""
    first = doc.splitlines()[0].strip() if doc else ""
    return first or f"{name} guardrail"


def builtin_rows() -> list[dict[str, str]]:
    """``[{name, description}]`` for every built-in guardrail, in stable order.

    Any guardrail registered in :data:`BUILTIN_GUARDRAILS` but missing from
    :data:`_ORDER` is appended (alphabetically) so a newly-added built-in is never
    silently dropped from the listing.
    """
    names = list(_ORDER)
    names += sorted(n for n in BUILTIN_GUARDRAILS if n not in _ORDER)
    seen = [n for n in names if n in BUILTIN_GUARDRAILS]
    return [{"name": n, "description": _describe(n)} for n in seen]


def cmd_guardrails(args: argparse.Namespace) -> int:
    """``himmy guardrails`` — list the built-in guardrails with one-line descriptions."""
    rows = builtin_rows()
    if getattr(args, "json", False):
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0

    c = styles(sys.stdout)
    print(f"{c['bold']}{c['snow']}built-in guardrails{c['reset']}")
    width = max((len(r["name"]) for r in rows), default=0)
    for r in rows:
        print(
            f"  {c['gold']}🛡{c['reset']} "
            f"{c['bold']}{c['snow']}{r['name']:<{width}}{c['reset']}  "
            f"{c['dim']}{r['description']}{c['reset']}"
        )
    _eprint(
        f"\n{c['faint']}attach with `guardrails: [pii, injection]` in an agent spec; "
        f"firings show live as 🛡 lines and in /guardrails.{c['reset']}"
    )
    return 0


# --------------------------------------------------------------- live shield line


def _event_payload(event: Any) -> dict[str, Any]:
    """The payload dict of a RunEvent (or a plain dict), defensively."""
    payload = getattr(event, "payload", None)
    if payload is None and isinstance(event, dict):
        payload = event.get("payload")
    return payload if isinstance(payload, dict) else {}


def _guardrail_label(payload: dict[str, Any]) -> str:
    """The guardrail's name from the GUARDRAIL_APPLIED flags (``pii:email`` → ``pii``).

    The emitter records the verdict's ``flags`` (e.g. ``pii:email``, ``injection``,
    ``ungrounded``); the leading token before ``:`` is the guardrail family. Falls back
    to a generic label when nothing was flagged (a bare redaction).
    """
    flags = payload.get("flags") or []
    for flag in flags:
        token = str(flag).split(":", 1)[0].strip()
        if token:
            return token
    return "guardrail"


def _detail(payload: dict[str, Any]) -> str:
    """A short human detail: the matched label classes, else the reason, else the action."""
    flags = [str(f) for f in (payload.get("flags") or [])]
    labels = [f.split(":", 1)[1] if ":" in f else f for f in flags]
    # Drop the guardrail-family duplicate (a bare ``injection`` flag has no sub-label).
    labels = [lbl for lbl in labels if lbl]
    if labels:
        return ", ".join(dict.fromkeys(labels))
    reasons = [str(r) for r in (payload.get("reasons") or []) if r]
    if reasons:
        return reasons[0]
    if payload.get("blocked"):
        return "blocked"
    if payload.get("redacted"):
        return "redacted"
    return "applied"


def _action_word(payload: dict[str, Any]) -> str:
    """``blocked`` / ``redacted`` / ``flagged`` — what the guardrail did to the text."""
    if payload.get("blocked"):
        return "blocked"
    if payload.get("redacted"):
        return "redacted"
    return "flagged"


def render_guardrail_event(event: Any, styles: dict[str, str]) -> str:
    """Format a ``GUARDRAIL_APPLIED`` event as a one-line faint shield for live UI.

    Signature is ``(event, styles)`` — ``event`` is a
    :class:`~himmy.core.events.RunEvent` (its ``payload`` carries ``stage``/``blocked``/
    ``redacted``/``flags``/``reasons``, as emitted in ``single_agent._apply_guardrail``),
    and ``styles`` is the style map from :func:`himmy.cli.ui.styles` (pass plain styles
    for a no-color line). A block is shown crimson, a redaction/flag faint, e.g.::

        🛡 guardrail pii redacted: email
        🛡 guardrail injection blocked: possible prompt injection

    Returns a single line with no trailing newline; the caller decides where it goes.
    """
    c = styles
    reset = c.get("reset", "")
    payload = _event_payload(event)
    name = _guardrail_label(payload)
    action = _action_word(payload)
    detail = _detail(payload)
    stage = str(payload.get("stage") or "").strip()
    stage_part = f" {c.get('faint', '')}[{stage}]{reset}" if stage else ""
    # Blocks are louder (crimson); redactions/flags stay faint so they never shout.
    accent = c.get("crimson", "") if payload.get("blocked") else c.get("faint", "")
    return (
        f"{c.get('faint', '')}🛡 guardrail {reset}"
        f"{c.get('gold', '')}{name}{reset} "
        f"{accent}{action}{reset}"
        f"{c.get('faint', '')}: {detail}{reset}"
        f"{stage_part}"
    )


# --------------------------------------------------------------- /guardrails slash


def _spec_guardrails(spec: Any) -> list[str]:
    """The guardrail names active on a spec (``spec.guardrails``), defensively."""
    names = getattr(spec, "guardrails", None) or []
    out: list[str] = []
    for n in names:
        text = str(n).strip()
        if text:
            out.append(text)
    return out


def _count_fired(events: Any) -> int:
    """How many GUARDRAIL_APPLIED events are in ``events`` (a recorder's event list)."""
    count = 0
    for event in events or []:
        et = getattr(event, "event_type", None)
        value = getattr(et, "value", et)
        if value == "GUARDRAIL_APPLIED":
            count += 1
    return count


def slash_guardrails(repl: Any) -> None:
    """``/guardrails`` — show the agent's active guardrails + how many fired this session.

    ``repl`` is the :class:`~himmy.cli.repl.ChatRepl` (uses ``repl.spec.guardrails``,
    ``repl._c`` styles, ``repl._eprint``, and ``repl.recorder.events`` to count firings).
    Written to be callable directly with a fake REPL for testing.
    """
    c = getattr(repl, "_c", None) or styles(sys.stdout)
    emit = getattr(repl, "_eprint", None) or _eprint
    active = _spec_guardrails(getattr(repl, "spec", None))
    recorder = getattr(repl, "recorder", None)
    fired = _count_fired(getattr(recorder, "events", None))

    if not active:
        emit(
            f"{c['dim']}no guardrails active on this agent — add e.g. "
            f"`guardrails: [pii, injection]` to the spec.{c['reset']}"
        )
        return
    emit(
        f"{c['dim']}active guardrails: {c['reset']}"
        f"{c['gold']}{', '.join(active)}{c['reset']}"
    )
    if fired:
        emit(
            f"{c['faint']}🛡 {fired} guardrail firing(s) this turn — `/why` for "
            f"the full event trace.{c['reset']}"
        )
    else:
        emit(f"{c['faint']}no guardrails fired this turn.{c['reset']}")


__all__ = [
    "cmd_guardrails",
    "render_guardrail_event",
    "slash_guardrails",
    "builtin_rows",
]
