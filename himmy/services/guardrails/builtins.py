"""Built-in guardrails: PII redaction, prompt-injection, and a configurable blocklist.

These are deliberately conservative regex/string checks (no model calls): they reduce the
obvious risks — leaking personal data, an injected "ignore previous instructions", a
banned phrase — without pretending to be a complete safety system. Compose them in a
:class:`~himmy.services.guardrails.base.GuardrailPipeline`.
"""

from __future__ import annotations

import re

from himmy.services.guardrails.base import (
    Guardrail,
    GuardrailPipeline,
    GuardrailVerdict,
)

# --------------------------------------------------------------------------- PII

_PII_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("email", "[REDACTED-EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("ssn", "[REDACTED-SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        "card",
        "[REDACTED-CARD]",
        re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    ),
    (
        "phone",
        "[REDACTED-PHONE]",
        re.compile(r"\b\+?\d[\d().\-\s]{7,}\d\b"),
    ),
    (
        "key",
        "[REDACTED-KEY]",
        re.compile(r"\b(?:sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16})\b"),
    ),
]


class PIIGuardrail:
    """Redacts emails, phone numbers, cards, SSNs, and obvious API keys."""

    name = "pii"

    def inspect(self, text: str, *, context: dict) -> GuardrailVerdict:
        """Replace any detected PII with typed placeholders (never blocks)."""
        redacted = text
        flags: list[str] = []
        # Keys/cards/ssn first (most specific), then phone/email.
        for label, placeholder, pattern in _PII_PATTERNS:
            new, count = pattern.subn(placeholder, redacted)
            if count:
                redacted = new
                flags.append(f"pii:{label}")
        return GuardrailVerdict(allowed=True, text=redacted, flags=flags)


# --------------------------------------------------------------------- injection

_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"ignore\s+(all\s+)?(the\s+)?previous\s+instructions", re.I),
    re.compile(r"disregard\s+(the\s+)?(above|previous|prior)", re.I),
    re.compile(r"ignore\s+the\s+above", re.I),
    re.compile(r"reveal\s+your\s+(system\s+)?(prompt|instructions)", re.I),
    re.compile(r"you\s+are\s+now\s+", re.I),
    re.compile(r"(print|show|repeat)\s+your\s+(system\s+)?prompt", re.I),
]


class InjectionGuardrail:
    """Flags (and by default blocks) common prompt-injection phrasings."""

    name = "injection"

    def __init__(self, *, block: bool = True) -> None:
        """``block`` (default True) makes a detection deny; otherwise it only flags."""
        self._block = block

    def inspect(self, text: str, *, context: dict) -> GuardrailVerdict:
        """Detect injection patterns; deny when ``block`` is set."""
        hits = [p.pattern for p in _INJECTION_PATTERNS if p.search(text)]
        if not hits:
            return GuardrailVerdict(allowed=True, text=text)
        return GuardrailVerdict(
            allowed=not self._block,
            text=text,
            reasons=["possible prompt injection"],
            flags=["injection"],
        )


class BlocklistGuardrail:
    """Blocks text matching any configured (case-insensitive) substring/pattern."""

    name = "blocklist"

    def __init__(self, patterns: list[str], *, name: str = "blocklist") -> None:
        """Compile the blocklist patterns (treated as regex, case-insensitive)."""
        self.name = name
        self._patterns = [re.compile(p, re.I) for p in patterns]

    def inspect(self, text: str, *, context: dict) -> GuardrailVerdict:
        """Deny when any blocklist pattern matches."""
        for pattern in self._patterns:
            if pattern.search(text):
                return GuardrailVerdict(
                    allowed=False,
                    text=text,
                    reasons=[f"blocked term: {pattern.pattern}"],
                    flags=[self.name],
                )
        return GuardrailVerdict(allowed=True, text=text)


#: Built-in guardrails resolvable by name (for specs/CLI).
BUILTIN_GUARDRAILS: dict[str, type[Guardrail]] = {
    "pii": PIIGuardrail,
    "injection": InjectionGuardrail,
}


def build_guardrail_pipeline(names: list[str]) -> GuardrailPipeline:
    """Build a pipeline from built-in guardrail names (``pii``/``injection``)."""
    guards: list[Guardrail] = []
    for name in names:
        factory = BUILTIN_GUARDRAILS.get(name)
        if factory is None:
            from himmy.core import HimmyError

            raise HimmyError(
                f"unknown guardrail {name!r}; known: {', '.join(BUILTIN_GUARDRAILS)}"
            )
        guards.append(factory())
    return GuardrailPipeline(guards)


__all__ = [
    "PIIGuardrail",
    "InjectionGuardrail",
    "BlocklistGuardrail",
    "BUILTIN_GUARDRAILS",
    "build_guardrail_pipeline",
]
