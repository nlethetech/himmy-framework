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
only ever sees the token. Storage is pluggable via :class:`TokenVaultBackend`:
the default :class:`InMemoryTokenVaultBackend` is process-local and bounded (LRU),
while :class:`SqliteTokenVaultBackend` keeps tokens reversible across restarts.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol

from himmy.core.sqlite_util import connect_hardened
from himmy.services.guardrails.base import GuardrailVerdict
from himmy.services.guardrails.builtins import _MAX_PII_SCAN_LEN, _PII_RULES

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from himmy.services.guardrails.builtins import PIIRule

logger = logging.getLogger("himmy.guardrails")


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


#: Default capacity of the in-memory token vault backend. Beyond this many unique
#: values the least-recently-used token is evicted (and becomes irreversible).
DEFAULT_VAULT_CAPACITY = 100_000

#: Tokens produced by :meth:`TokenVault.tokenize` — ``[LABEL:12-hex-digest]``.
_TOKEN_RE = re.compile(r"\[[^\[\]]+:[0-9a-f]{12}\]")


class TokenVaultBackend(Protocol):
    """Storage seam for :class:`TokenVault` so durable backends can be plugged in.

    Implementations should treat ``get_token``/``get_value`` as a *use* of the
    mapping (an LRU touch) when they enforce a capacity.
    """

    def get_token(self, value: str) -> str | None:
        """The token previously stored for ``value``, or ``None``."""
        ...  # pragma: no cover - protocol

    def get_value(self, token: str) -> str | None:
        """The original value behind ``token``, or ``None`` (e.g. evicted)."""
        ...  # pragma: no cover - protocol

    def store(self, value: str, token: str) -> None:
        """Persist the ``value`` ↔ ``token`` pair."""
        ...  # pragma: no cover - protocol


class InMemoryTokenVaultBackend:
    """Process-local LRU-bounded storage (the default :class:`TokenVault` backend).

    Capacity defaults to :data:`DEFAULT_VAULT_CAPACITY`; pass ``max_entries=None``
    for unbounded. **Evicted tokens become irreversible** — ``detokenize`` leaves
    them in place — so a one-time warning is logged on the first eviction.
    """

    def __init__(self, *, max_entries: int | None = DEFAULT_VAULT_CAPACITY) -> None:
        self._tokens_by_value: OrderedDict[str, str] = OrderedDict()
        self._values_by_token: dict[str, str] = {}
        self._max_entries = max_entries
        self._evict_warned = False

    def get_token(self, value: str) -> str | None:
        """The token for ``value`` (LRU touch), or ``None``."""
        token = self._tokens_by_value.get(value)
        if token is not None:
            self._tokens_by_value.move_to_end(value)
        return token

    def get_value(self, token: str) -> str | None:
        """The value behind ``token`` (LRU touch), or ``None`` if evicted/unknown."""
        value = self._values_by_token.get(token)
        if value is not None:
            self._tokens_by_value.move_to_end(value)
        return value

    def store(self, value: str, token: str) -> None:
        """Store the pair, evicting the least-recently-used entry at capacity."""
        self._tokens_by_value[value] = token
        self._tokens_by_value.move_to_end(value)
        self._values_by_token[token] = value
        if self._max_entries is None:
            return
        while len(self._tokens_by_value) > self._max_entries:
            _, evicted_token = self._tokens_by_value.popitem(last=False)
            self._values_by_token.pop(evicted_token, None)
            if not self._evict_warned:
                self._evict_warned = True
                logger.warning(
                    "TokenVault at capacity (%d): evicting least-recently-used "
                    "tokens; evicted tokens can no longer be detokenized. "
                    "Raise max_entries or use SqliteTokenVaultBackend.",
                    self._max_entries,
                )


class SqliteTokenVaultBackend:
    """SQLite-file-backed storage: tokens stay reversible across process restarts.

    A reference durable backend (stdlib only). Multiple vault instances — even in
    separate processes — can share one ``path`` and resolve each other's tokens.
    ``max_entries`` (default unbounded — it's disk, not memory) enables the same
    LRU eviction semantics as :class:`InMemoryTokenVaultBackend`. Note the vault
    stores **plaintext sensitive values**; protect the file accordingly.
    """

    def __init__(self, path: str, *, max_entries: int | None = None) -> None:
        self._conn: sqlite3.Connection = connect_hardened(path)
        self._max_entries = max_entries
        self._evict_warned = False
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS token_vault ("
            " value TEXT PRIMARY KEY,"
            " token TEXT NOT NULL UNIQUE,"
            " seq INTEGER NOT NULL)"
        )
        self._conn.commit()

    def get_token(self, value: str) -> str | None:
        """The token for ``value`` (LRU touch), or ``None``."""
        row = self._conn.execute(
            "SELECT token FROM token_vault WHERE value = ?", (value,)
        ).fetchone()
        if row is None:
            return None
        self._touch(value)
        return str(row[0])

    def get_value(self, token: str) -> str | None:
        """The value behind ``token`` (LRU touch), or ``None`` if evicted/unknown."""
        row = self._conn.execute(
            "SELECT value FROM token_vault WHERE token = ?", (token,)
        ).fetchone()
        if row is None:
            return None
        value = str(row[0])
        self._touch(value)
        return value

    def store(self, value: str, token: str) -> None:
        """Store the pair, evicting the least-recently-used rows at capacity."""
        self._conn.execute(
            "INSERT OR REPLACE INTO token_vault (value, token, seq) VALUES"
            " (?, ?, (SELECT COALESCE(MAX(seq), 0) + 1 FROM token_vault))",
            (value, token),
        )
        self._evict_if_needed()
        self._conn.commit()

    def _touch(self, value: str) -> None:
        """Mark ``value`` as most-recently-used."""
        self._conn.execute(
            "UPDATE token_vault SET seq ="
            " (SELECT COALESCE(MAX(seq), 0) + 1 FROM token_vault)"
            " WHERE value = ?",
            (value,),
        )
        self._conn.commit()

    def _evict_if_needed(self) -> None:
        """Delete the lowest-``seq`` (least-recently-used) rows beyond capacity."""
        if self._max_entries is None:
            return
        row = self._conn.execute("SELECT COUNT(*) FROM token_vault").fetchone()
        overflow = int(row[0]) - self._max_entries
        if overflow <= 0:
            return
        self._conn.execute(
            "DELETE FROM token_vault WHERE value IN"
            " (SELECT value FROM token_vault ORDER BY seq ASC LIMIT ?)",
            (overflow,),
        )
        if not self._evict_warned:
            self._evict_warned = True
            logger.warning(
                "TokenVault at capacity (%d): evicting least-recently-used "
                "tokens; evicted tokens can no longer be detokenized.",
                self._max_entries,
            )


class TokenVault:
    """A reversible token↔value map for ``tokenize`` actions.

    Storage is delegated to a :class:`TokenVaultBackend`. The default is a
    bounded :class:`InMemoryTokenVaultBackend` (process-local, LRU-evicting —
    evicted tokens become irreversible); pass a :class:`SqliteTokenVaultBackend`
    to keep tokens reversible across restarts.
    """

    def __init__(self, *, backend: TokenVaultBackend | None = None) -> None:
        self._backend: TokenVaultBackend = backend or InMemoryTokenVaultBackend()

    def tokenize(self, label: str, value: str) -> str:
        """Return a stable token for ``value`` (same value → same token)."""
        token = self._backend.get_token(value)
        if token is None:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
            token = f"[{label.upper()}:{digest}]"
            self._backend.store(value, token)
        return token

    def detokenize(self, text: str) -> str:
        """Restore original values for any live (non-evicted) tokens in ``text``."""

        def _restore(match: re.Match[str]) -> str:
            value = self._backend.get_value(match.group(0))
            return value if value is not None else match.group(0)

        return _TOKEN_RE.sub(_restore, text)


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
        # Cap the regex scan the same way PIIGuardrail does (see
        # builtins._MAX_PII_SCAN_LEN): Python's ``re`` holds the GIL, so a multi-MiB
        # blob stalls the event loop ~900 ms. Only the head is scanned; the tail is
        # re-appended verbatim (redact/tokenize) or triggers a fail-closed block when
        # the policy can BLOCK — an unscanned tail could carry the very class to withhold.
        tail = ""
        if len(text) > _MAX_PII_SCAN_LEN:
            text_head, tail = text[:_MAX_PII_SCAN_LEN], text[_MAX_PII_SCAN_LEN:]
        else:
            text_head = text
        result = text_head
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

        if tail and self.suppresses_output_content():
            # A :block policy cannot be verified past the scan cap; rather than pass
            # the unscanned tail through (fail-open past the cap), withhold the whole
            # input. Redact/tokenize policies instead re-append the tail below.
            blocked.append("unscanned-tail")
        elif tail:
            result = result + tail

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

    def suppresses_output_content(self) -> bool:
        """True when this DLP guard can BLOCK matched content (a ``…:block`` action).

        A block means the offending text is withheld, never redacted-in-place — so a
        streaming caller must NOT emit tokens before this guard runs (already-streamed
        secrets cannot be recalled). The default action counts: ``*:block`` blocks
        every class. Redact/tokenize-only policies (no block) return False, so they
        stay on the cheaper guard-after-streaming path.
        """
        if self._policy.default is DlpAction.BLOCK:
            return True
        return any(action is DlpAction.BLOCK for action in self._policy.actions.values())

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
    """Build a DLP guardrail from env (``HIMMY_DLP_POLICY`` / ``HIMMY_DLP_BACKEND``).

    ``HIMMY_DLP_VAULT_PATH`` (optional) opts the token vault into the durable
    :class:`SqliteTokenVaultBackend` at that path; default stays in-memory.
    """
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
    vault_path = os.environ.get("HIMMY_DLP_VAULT_PATH", "")
    vault = (
        TokenVault(backend=SqliteTokenVaultBackend(vault_path)) if vault_path else None
    )
    return DlpGuardrail(
        policy=policy, vault=vault, audit_sink=audit_sink, analyzer=analyzer
    )


__all__ = [
    "DEFAULT_VAULT_CAPACITY",
    "DlpAction",
    "DlpPolicy",
    "TokenVault",
    "TokenVaultBackend",
    "InMemoryTokenVaultBackend",
    "SqliteTokenVaultBackend",
    "DlpGuardrail",
    "PresidioAnalyzerAdapter",
    "build_dlp_guardrail",
]
