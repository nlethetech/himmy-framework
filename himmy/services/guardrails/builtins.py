"""Built-in guardrails: PII redaction, prompt-injection, and a configurable blocklist.

These are deliberately conservative regex/string checks (no model calls): they reduce the
obvious risks — leaking personal data, an injected "ignore previous instructions", a
banned phrase — without pretending to be a complete safety system. Compose them in a
:class:`~himmy.services.guardrails.base.GuardrailPipeline`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from himmy.services.guardrails.base import (
    Guardrail,
    GuardrailPipeline,
    GuardrailVerdict,
)

# --------------------------------------------------------------------------- PII


@dataclass(frozen=True)
class PIIRule:
    """One redaction rule: a label, placeholder, regex, and optional validator.

    ``validator`` (when set) receives the matched text and must return True for the
    match to be redacted — this is how regex false positives are cut (e.g. a 16-digit
    run is only treated as a card if it passes the Luhn checksum).
    """

    label: str
    placeholder: str
    pattern: re.Pattern[str]
    validator: Callable[[str], bool] | None = None


def _luhn_ok(text: str) -> bool:
    """True when ``text``'s digits pass the Luhn checksum (real card numbers do)."""
    digits = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for i, n in enumerate(digits):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _ipv4_ok(text: str) -> bool:
    """True when ``text`` is a syntactically valid IPv4 (each octet 0–255)."""
    parts = text.split(".")
    return len(parts) == 4 and all(p.isdigit() and int(p) <= 255 for p in parts)


def _redact(text: str, rules: list[PIIRule]) -> tuple[str, list[str]]:
    """Apply each rule (validated) in order; return redacted text + per-label flags."""
    redacted = text
    flags: list[str] = []
    for rule in rules:
        hit = False

        def _replace(match: re.Match[str], r: PIIRule = rule) -> str:
            nonlocal hit
            if r.validator is not None and not r.validator(match.group(0)):
                return match.group(0)
            hit = True
            return r.placeholder

        redacted = rule.pattern.sub(_replace, redacted)
        if hit:
            flags.append(f"pii:{rule.label}")
    return redacted, flags


# Ordered most-specific → loosest, so structured matches (keys/cards/IPs) redact
# before the broad phone rule could swallow their digits.
_PII_RULES: list[PIIRule] = [
    PIIRule(
        "api_key",
        "[REDACTED-KEY]",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[posru]_[A-Za-z0-9]{20,}"
            r"|AIza[0-9A-Za-z_-]{35}|xox[baprs]-[0-9A-Za-z-]{10,}"
            r"|AKIA[0-9A-Z]{16})\b"
        ),
    ),
    PIIRule(
        "jwt",
        "[REDACTED-JWT]",
        re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"),
    ),
    PIIRule(
        "url_credentials",
        "[REDACTED-URL]",
        re.compile(r"\bhttps?://[^\s/:@]+:[^\s/:@]+@\S+"),
    ),
    PIIRule("email", "[REDACTED-EMAIL]", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    PIIRule(
        "iban",
        "[REDACTED-IBAN]",
        re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    ),
    PIIRule(
        "card",
        "[REDACTED-CARD]",
        re.compile(r"\b(?:\d[ -]?){13,19}\b"),
        _luhn_ok,
    ),
    PIIRule("ssn", "[REDACTED-SSN]", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    PIIRule(
        "ipv4", "[REDACTED-IP]", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), _ipv4_ok
    ),
    PIIRule(
        "mac",
        "[REDACTED-MAC]",
        re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    ),
    PIIRule(
        "phone",
        "[REDACTED-PHONE]",
        re.compile(r"(?<![\d.])\+?\d[\d().\-\s]{7,}\d(?![\d.])"),
    ),
]


class PIIGuardrail:
    """Redacts API keys, JWTs, emails, IBANs, Luhn-valid cards, SSNs, IPs, MACs, phones."""

    name = "pii"

    def __init__(self, *, rules: list[PIIRule] | None = None) -> None:
        """Use the default rule set, or a custom list of :class:`PIIRule`s."""
        self._rules = rules if rules is not None else _PII_RULES

    def inspect(self, text: str, *, context: dict) -> GuardrailVerdict:
        """Replace any detected PII with typed placeholders (never blocks)."""
        redacted, flags = _redact(text, self._rules)
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


_NEPAL_PII_RULES: list[PIIRule] = [
    # +977 international form (with or without separators).
    PIIRule("np_phone", "[REDACTED-NP-PHONE]", re.compile(r"\+?977[-\s]?\d{8,10}")),
    # Domestic 10-digit mobile (98/97/96/9810… prefixes).
    PIIRule(
        "np_mobile", "[REDACTED-NP-PHONE]", re.compile(r"(?<!\d)9[678]\d{8}(?!\d)")
    ),
    # Citizenship certificate numbers (commonly ##-##-##-##### style).
    PIIRule(
        "citizenship",
        "[REDACTED-CITIZENSHIP]",
        re.compile(r"\b\d{2}-\d{2}-\d{2}-\d{4,6}\b"),
    ),
    # PAN (Permanent Account Number) — 9 digits, REQUIRED to be PAN-labelled so a bare
    # 9-digit number (an amount, an id) is never wrongly redacted.
    PIIRule("pan", "[REDACTED-PAN]", re.compile(r"\bPAN[:\s#]*\d{9}\b", re.IGNORECASE)),
]


class NepalPIIGuardrail:
    """Redacts Nepal-specific PII: +977 / domestic mobiles, citizenship numbers, PAN."""

    name = "nepal_pii"

    def inspect(self, text: str, *, context: dict) -> GuardrailVerdict:
        """Replace Nepali PII with typed placeholders (never blocks)."""
        redacted, flags = _redact(text, _NEPAL_PII_RULES)
        return GuardrailVerdict(allowed=True, text=redacted, flags=flags)


#: Built-in guardrails resolvable by name (for specs/CLI). ``dlp`` is appended below
#: (after ``_PII_RULES`` is defined) to avoid a circular import.
BUILTIN_GUARDRAILS: dict[str, type[Guardrail]] = {
    "pii": PIIGuardrail,
    "injection": InjectionGuardrail,
    "nepal_pii": NepalPIIGuardrail,
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


# Registered here, after _PII_RULES exists, so dlp.py can import the rule set without
# a circular import. DlpGuardrail() with no args defaults to redact-all (PII-like).
from himmy.services.guardrails.dlp import DlpGuardrail as _DlpGuardrail  # noqa: E402

BUILTIN_GUARDRAILS["dlp"] = _DlpGuardrail


__all__ = [
    "PIIRule",
    "PIIGuardrail",
    "InjectionGuardrail",
    "BlocklistGuardrail",
    "NepalPIIGuardrail",
    "BUILTIN_GUARDRAILS",
    "build_guardrail_pipeline",
]
