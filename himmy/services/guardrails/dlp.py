"""DLP as a compliance control: classify sensitive data → a policy action (WS4.1).

Where :class:`~himmy.services.guardrails.builtins.PIIGuardrail` always *redacts*, the
DLP guardrail applies a **per-class policy action** — ``allow`` / ``redact`` /
``tokenize`` / ``block`` — so a deployment can, say, tokenize emails (reversibly),
redact phones, and hard-**block** credit-card numbers. Detections are counted and
reported (never the values) so they can be written to the security audit log
(`audit_sink`). An optional Microsoft Presidio backend (extra ``dlp``) adds ML-based
detection beyond the regex rules.

``tokenize`` uses a :class:`TokenVault` (a reversible token↔value map) so a downstream
workflow can round-trip the original via :meth:`TokenVault.detokenize` while the model
only ever sees the token.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from himmy.services.guardrails.base import GuardrailVerdict
from himmy.services.guardrails.builtins import _PII_RULES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from himmy.services.guardrails.builtins import PIIRule


class DlpAction(str, Enum):
    """What to do with a detected sensitive-data class."""

    ALLOW = "allow"
    REDACT = "redact"
    TOKENIZE = "tokenize"
    BLOCK = "block"


@dataclass(frozen=True)
class DlpPolicy:
    """Maps a data class (e.g. ``email``, ``card``) to a :class:`DlpAction`."""

    actions: dict[str, DlpAction] = field(default_factory=dict)
    default: DlpAction = DlpAction.REDACT

    def action_for(self, label: str) -> DlpAction:
        """The action for ``label`` (or the policy default)."""
        return self.actions.get(label, self.default)

    @classmethod
    def parse(cls, spec: str, *, default: DlpAction = DlpAction.REDACT) -> DlpPolicy:
        """Parse ``"card:block,email:tokenize,*:redact"`` into a policy."""
        actions: dict[str, DlpAction] = {}
        resolved_default = default
        for pair in spec.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            label, _, action = pair.partition(":")
            label, action = label.strip(), action.strip().lower()
            try:
                parsed = DlpAction(action)
            except ValueError:
                continue
            if label in ("*", "default"):
                resolved_default = parsed
            else:
                actions[label] = parsed
        return cls(actions=actions, default=resolved_default)


class TokenVault:
    """A reversible token↔value map for ``tokenize`` actions (in-memory by default)."""

    def __init__(self) -> None:
        self._by_value: dict[str, str] = {}
        self._by_token: dict[str, str] = {}

    def tokenize(self, label: str, value: str) -> str:
        """Return a stable token for ``value`` (same value → same token)."""
        token = self._by_value.get(value)
        if token is None:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
            token = f"[{label.upper()}:{digest}]"
            self._by_value[value] = token
            self._by_token[token] = value
        return token

    def detokenize(self, text: str) -> str:
        """Restore original values for any tokens present in ``text``."""
        for token, value in self._by_token.items():
            text = text.replace(token, value)
        return text


class DlpGuardrail:
    """Apply a :class:`DlpPolicy` over PII classes (regex + optional Presidio)."""

    name = "dlp"

    def __init__(
        self,
        *,
        policy: DlpPolicy | None = None,
        rules: list[PIIRule] | None = None,
        vault: TokenVault | None = None,
        audit_sink: Callable[[dict[str, int]], None] | None = None,
        analyzer: Any = None,
    ) -> None:
        """Configure the policy, rule set, token vault, audit sink, and ML backend."""
        self._policy = policy or DlpPolicy()
        self._rules = rules if rules is not None else _PII_RULES
        self._vault = vault or TokenVault()
        self._audit_sink = audit_sink
        self._analyzer = analyzer

    def inspect(self, text: str, *, context: dict[str, Any]) -> GuardrailVerdict:
        """Classify sensitive data and apply the per-class action."""
        result = text
        counts: dict[str, int] = {}
        blocked: list[str] = []

        for rule in self._rules:
            action = self._policy.action_for(rule.label)
            if action is DlpAction.ALLOW:
                continue
            matches = [
                m
                for m in rule.pattern.finditer(result)
                if rule.validator is None or rule.validator(m.group(0))
            ]
            if not matches:
                continue
            counts[rule.label] = counts.get(rule.label, 0) + len(matches)
            if action is DlpAction.BLOCK:
                blocked.append(rule.label)
                continue
            result = self._apply(rule, result, action)

        if self._analyzer is not None:
            result, blocked = self._apply_presidio(result, counts, blocked)

        flags = [f"dlp:{label}" for label in counts]
        if self._audit_sink is not None and counts:
            self._audit_sink(dict(counts))
        if blocked:
            return GuardrailVerdict(
                allowed=False,
                text=text,
                reasons=[
                    f"DLP blocked sensitive data: {', '.join(sorted(set(blocked)))}"
                ],
                flags=flags,
            )
        return GuardrailVerdict(allowed=True, text=result, flags=flags)

    def _apply(self, rule: PIIRule, text: str, action: DlpAction) -> str:
        """Redact or tokenize every validated match of ``rule`` in ``text``."""

        def _replace(match: re.Match[str]) -> str:
            value = match.group(0)
            if rule.validator is not None and not rule.validator(value):
                return value
            if action is DlpAction.TOKENIZE:
                return self._vault.tokenize(rule.label, value)
            return rule.placeholder

        return rule.pattern.sub(_replace, text)

    def _apply_presidio(
        self, text: str, counts: dict[str, int], blocked: list[str]
    ) -> tuple[str, list[str]]:
        """Apply the policy to entities found by the Presidio analyzer (ML backend)."""
        entities = self._analyzer.analyze(text)  # list of (label, start, end)
        # Mutate from the end so earlier offsets stay valid.
        for label, start, end in sorted(entities, key=lambda e: e[1], reverse=True):
            action = self._policy.action_for(label)
            if action is DlpAction.ALLOW:
                continue
            counts[label] = counts.get(label, 0) + 1
            if action is DlpAction.BLOCK:
                blocked.append(label)
                continue
            value = text[start:end]
            replacement = (
                self._vault.tokenize(label, value)
                if action is DlpAction.TOKENIZE
                else f"[REDACTED-{label.upper()}]"
            )
            text = text[:start] + replacement + text[end:]
        return text, blocked


class PresidioAnalyzerAdapter:
    """Thin adapter over Microsoft Presidio (lazy import; analyzer injectable)."""

    #: Presidio entity type → our data-class label.
    DEFAULT_MAP = {
        "EMAIL_ADDRESS": "email",
        "CREDIT_CARD": "card",
        "PHONE_NUMBER": "phone",
        "US_SSN": "ssn",
        "IP_ADDRESS": "ipv4",
        "PERSON": "person",
        "LOCATION": "location",
        "IBAN_CODE": "iban",
    }

    def __init__(
        self,
        *,
        analyzer: Any = None,
        language: str = "en",
        entity_map: dict[str, str] | None = None,
    ) -> None:
        self._analyzer = analyzer
        self._language = language
        self._map = entity_map or dict(self.DEFAULT_MAP)

    def _ensure(self) -> Any:
        if self._analyzer is None:
            from presidio_analyzer import (
                AnalyzerEngine,  # type: ignore[import-not-found]
            )

            self._analyzer = AnalyzerEngine()
        return self._analyzer

    def analyze(self, text: str) -> list[tuple[str, int, int]]:
        """Return ``(class_label, start, end)`` for each recognized entity."""
        results = self._ensure().analyze(text=text, language=self._language)
        spans: list[tuple[str, int, int]] = []
        for r in results:
            label = self._map.get(r.entity_type)
            if label is not None:
                spans.append((label, r.start, r.end))
        return spans


def build_dlp_guardrail(
    *, audit_sink: Callable[[dict[str, int]], None] | None = None
) -> DlpGuardrail:
    """Build a DLP guardrail from env (``HIMMY_DLP_POLICY`` / ``HIMMY_DLP_BACKEND``)."""
    import os

    spec = os.environ.get("HIMMY_DLP_POLICY", "")
    default = DlpAction(os.environ.get("HIMMY_DLP_DEFAULT", "redact"))
    policy = (
        DlpPolicy.parse(spec, default=default) if spec else DlpPolicy(default=default)
    )
    analyzer = (
        PresidioAnalyzerAdapter()
        if os.environ.get("HIMMY_DLP_BACKEND", "").lower() == "presidio"
        else None
    )
    return DlpGuardrail(policy=policy, audit_sink=audit_sink, analyzer=analyzer)


__all__ = [
    "DlpAction",
    "DlpPolicy",
    "TokenVault",
    "DlpGuardrail",
    "PresidioAnalyzerAdapter",
    "build_dlp_guardrail",
]
