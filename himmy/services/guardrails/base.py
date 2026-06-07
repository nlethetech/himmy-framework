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


@dataclass
class GuardrailVerdict:
    """The outcome of inspecting one piece of text."""

    allowed: bool = True
    text: str = ""
    reasons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


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

    def inspect(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailVerdict:
        """Run every guardrail over ``text``; thread redactions, accumulate flags."""
        ctx = context or {}
        current = text
        allowed = True
        reasons: list[str] = []
        flags: list[str] = []
        for guardrail in self._guardrails:
            verdict = guardrail.inspect(current, context=ctx)
            current = verdict.text
            reasons.extend(verdict.reasons)
            flags.extend(verdict.flags)
            if not verdict.allowed:
                allowed = False
        return GuardrailVerdict(
            allowed=allowed, text=current, reasons=reasons, flags=flags
        )


__all__ = ["GuardrailVerdict", "Guardrail", "GuardrailPipeline"]
