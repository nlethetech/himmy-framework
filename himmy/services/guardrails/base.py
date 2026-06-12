"""Guardrails kernel: the verdict, the guardrail contract, and a pipeline.

A :class:`Guardrail` inspects a piece of text (a prompt, a model reply, a tool argument)
and returns a :class:`GuardrailVerdict` — possibly *redacted* text, an ``allowed`` flag
(``False`` blocks), and human-readable reasons/flags. A :class:`GuardrailPipeline` chains
several: redactions accumulate as text flows through, and the pipeline blocks if any
guardrail blocks. Guardrails are synchronous and dependency-free (regex/string checks),
so they sit on the hot path without adding latency or I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class GuardrailTrigger:
    """One guardrail firing: which guardrail, what it matched, and how it acted.

    ``guardrail`` is the originating guardrail's name; ``label`` is the matched
    class/flag with the guardrail prefix stripped (e.g. flag ``pii:email`` →
    ``email``), falling back to the guardrail name when nothing more specific was
    flagged; ``action`` is ``"block"`` (denied), ``"redact"`` (text transformed),
    or ``"flag"`` (detected, text untouched). Labels are classes, never values, so
    triggers are safe to write to an audit log.
    """

    guardrail: str
    label: str
    action: str


@dataclass
class GuardrailVerdict:
    """The outcome of inspecting one piece of text."""

    allowed: bool = True
    text: str = ""
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    #: Per-guardrail attribution for every block/redaction/flag (who fired, on what).
    triggers: list[GuardrailTrigger] = field(default_factory=list)
    #: Set by :meth:`GuardrailPipeline.inspect` when a BLOCKING guardrail supplied its
    #: own safe replacement text (it returned ``allowed=False`` with ``text`` changed
    #: from its own input — e.g. GroundingGuardrail's tailored refusal). The enforcing
    #: caller honors that replacement; when this is False on a block it means the
    #: blocking guardrail left the offending text in place (DLP ``…:block`` / blocklist /
    #: injection all return their input unchanged), so the caller MUST substitute a safe
    #: placeholder rather than let ``text`` through. Meaningless when ``allowed`` is True.
    block_replacement: bool = False


@runtime_checkable
class Guardrail(Protocol):
    """Inspects text and returns a (possibly redacted) :class:`GuardrailVerdict`."""

    name: str

    def inspect(self, text: str, *, context: dict[str, Any]) -> GuardrailVerdict: ...


class GuardrailPipeline:
    """Runs a sequence of guardrails; redactions chain, any block blocks the whole."""

    def __init__(self, guardrails: list[Guardrail]) -> None:
        """Wire the ordered guardrails the pipeline applies."""
        self._guardrails = list(guardrails)

    @property
    def names(self) -> list[str]:
        """The names of the guardrails in this pipeline."""
        return [g.name for g in self._guardrails]

    def suppresses_output_content(self) -> bool:
        """True when any guardrail in this pipeline WITHHOLDS blocked output content.

        A guardrail declares this (via an optional ``suppresses_output_content``
        method) when a block drops the offending text rather than redacting it in
        place — e.g. a DLP ``…:block`` policy, a blocklist, or a blocking injection
        guard. A streaming caller uses this to decide it must BUFFER the full answer
        and guard it before emitting any token (an already-streamed secret/banned
        phrase cannot be recalled). Guards without the method (pure redactors,
        grounding) contribute False, so a redact-only / grounding-only output
        pipeline stays on the cheaper guard-after-streaming path.
        """
        for guardrail in self._guardrails:
            probe = getattr(guardrail, "suppresses_output_content", None)
            if callable(probe) and probe():
                return True
        return False

    def inspect(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailVerdict:
        """Run every guardrail over ``text``; thread redactions, accumulate flags.

        Every guardrail that fires (blocks, redacts, or merely flags) is attributed
        in the aggregate verdict's ``triggers``, so a block/redaction always records
        WHICH guardrail caused it and on what matched class — guardrails that supply
        their own :class:`GuardrailTrigger` list are passed through verbatim, all
        others get triggers synthesized from their name, flags, and effect.
        """
        ctx = context or {}
        current = text
        allowed = True
        block_replacement = False
        reasons: list[str] = []
        flags: list[str] = []
        triggers: list[GuardrailTrigger] = []
        for guardrail in self._guardrails:
            verdict = guardrail.inspect(current, context=ctx)
            changed = verdict.text != current
            if not verdict.allowed:
                allowed = False
                # A blocking guardrail that rewrote ITS OWN input supplied a safe
                # replacement (Grounding's refusal); one that returned its input
                # unchanged (DLP …:block / blocklist / injection) did not — even if
                # an UPSTREAM redactor already changed ``current``. Tracking this per
                # blocking guardrail (not by comparing the aggregate text to the
                # original) is what closes the mixed redact+block fail-open: the
                # enforcing caller must substitute a placeholder when no blocking
                # guardrail supplied its own replacement.
                block_replacement = changed
            current = verdict.text
            reasons.extend(verdict.reasons)
            flags.extend(verdict.flags)
            triggers.extend(
                verdict.triggers
                or _synthesize_triggers(guardrail.name, verdict, changed=changed)
            )
        return GuardrailVerdict(
            allowed=allowed,
            text=current,
            reasons=reasons,
            flags=flags,
            triggers=triggers,
            block_replacement=block_replacement,
        )


def _synthesize_triggers(
    name: str, verdict: GuardrailVerdict, *, changed: bool
) -> list[GuardrailTrigger]:
    """Attribute one guardrail's verdict: a trigger per flag (or one bare trigger).

    Quiet verdicts (allowed, text untouched, nothing flagged) yield no triggers.
    The label is the flag with the ``"<name>:"`` prefix stripped (``pii:email`` →
    ``email``); a firing guardrail that raised no flags is labelled by its name.
    """
    fired = not verdict.allowed or changed or verdict.flags or verdict.reasons
    if not fired:
        return []
    action = "block" if not verdict.allowed else ("redact" if changed else "flag")
    labels = [flag.removeprefix(f"{name}:") for flag in verdict.flags] or [name]
    return [
        GuardrailTrigger(guardrail=name, label=label, action=action) for label in labels
    ]


__all__ = ["GuardrailTrigger", "GuardrailVerdict", "Guardrail", "GuardrailPipeline"]
