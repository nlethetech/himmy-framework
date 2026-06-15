"""WS4.1 — DLP as policy: classify → allow/redact/tokenize/block, audited."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from himmy.services.guardrails import build_guardrail_pipeline
from himmy.services.guardrails.dlp import (
    DlpAction,
    DlpGuardrail,
    DlpPolicy,
    InMemoryTokenVaultBackend,
    PresidioAnalyzerAdapter,
    SqliteTokenVaultBackend,
    TokenVault,
)


def _inspect(guard: DlpGuardrail, text: str):
    return guard.inspect(text, context={})


# --------------------------------------------------------------------- policy
def test_policy_parse_and_default() -> None:
    pol = DlpPolicy.parse("card:block,email:tokenize,*:redact")
    assert pol.action_for("card") is DlpAction.BLOCK
    assert pol.action_for("email") is DlpAction.TOKENIZE
    assert pol.action_for("phone") is DlpAction.REDACT  # default


def test_token_vault_roundtrip() -> None:
    vault = TokenVault()
    t1 = vault.tokenize("email", "a@b.com")
    t2 = vault.tokenize("email", "a@b.com")
    assert t1 == t2  # stable
    assert vault.tokenize("email", "c@d.com") != t1
    assert vault.detokenize(f"contact {t1} now") == "contact a@b.com now"


# ------------------------------------------------------------- vault backends
def test_sqlite_backend_tokens_stable_across_vault_instances(tmp_path: Path) -> None:
    db = str(tmp_path / "vault.db")
    vault1 = TokenVault(backend=SqliteTokenVaultBackend(db))
    token = vault1.tokenize("email", "a@b.com")

    # A fresh vault + fresh backend on the same file (≈ process restart).
    vault2 = TokenVault(backend=SqliteTokenVaultBackend(db))
    assert vault2.tokenize("email", "a@b.com") == token  # same token survives
    assert vault2.detokenize(f"contact {token} now") == "contact a@b.com now"


def test_in_memory_backend_lru_eviction_at_capacity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    vault = TokenVault(backend=InMemoryTokenVaultBackend(max_entries=2))
    vault.tokenize("email", "a@b.com")
    t_b = vault.tokenize("email", "b@b.com")
    vault.tokenize("email", "a@b.com")  # touch a → b becomes least-recently-used

    with caplog.at_level(logging.WARNING, logger="himmy.guardrails"):
        t_c = vault.tokenize("email", "c@b.com")  # evicts b
        vault.tokenize("email", "d@b.com")  # evicts a — but warns only once

    # Evicted token is irreversible (left in place); live tokens still round-trip.
    assert vault.detokenize(t_b) == t_b
    assert vault.detokenize(f"{t_c} and {vault.tokenize('email', 'd@b.com')}") == (
        "c@b.com and d@b.com"
    )
    evict_warnings = [r for r in caplog.records if "evicting" in r.getMessage()]
    assert len(evict_warnings) == 1  # one-time warning


def test_sqlite_backend_lru_eviction_at_capacity(tmp_path: Path) -> None:
    db = str(tmp_path / "vault.db")
    vault = TokenVault(backend=SqliteTokenVaultBackend(db, max_entries=2))
    t_a = vault.tokenize("email", "a@b.com")
    t_b = vault.tokenize("email", "b@b.com")
    vault.tokenize("email", "a@b.com")  # touch a → b is LRU
    t_c = vault.tokenize("email", "c@b.com")  # evicts b

    assert vault.detokenize(t_b) == t_b  # evicted → irreversible
    assert vault.detokenize(t_a) == "a@b.com"
    assert vault.detokenize(t_c) == "c@b.com"


def test_unknown_token_left_untouched() -> None:
    vault = TokenVault()
    assert vault.detokenize("see [EMAIL:0123456789ab]") == "see [EMAIL:0123456789ab]"


# ------------------------------------------------------------------- actions
def test_redact_action() -> None:
    guard = DlpGuardrail(policy=DlpPolicy(actions={"email": DlpAction.REDACT}))
    v = _inspect(guard, "mail me at a@b.com")
    assert v.allowed is True
    assert "a@b.com" not in v.text
    assert "dlp:email" in v.flags


def test_tokenize_action_is_reversible() -> None:
    vault = TokenVault()
    guard = DlpGuardrail(
        policy=DlpPolicy(actions={"email": DlpAction.TOKENIZE}), vault=vault
    )
    v = _inspect(guard, "mail me at a@b.com")
    assert "a@b.com" not in v.text
    assert vault.detokenize(v.text) == "mail me at a@b.com"  # restored downstream


def test_block_action_denies() -> None:
    # 4111111111111111 is a Luhn-valid test card.
    guard = DlpGuardrail(policy=DlpPolicy(actions={"card": DlpAction.BLOCK}))
    v = _inspect(guard, "card 4111 1111 1111 1111")
    assert v.allowed is False
    assert any("card" in r for r in v.reasons)


def test_allow_action_leaves_text() -> None:
    guard = DlpGuardrail(
        policy=DlpPolicy(actions={"email": DlpAction.ALLOW}, default=DlpAction.ALLOW)
    )
    v = _inspect(guard, "mail me at a@b.com")
    assert v.text == "mail me at a@b.com"
    assert v.flags == []


def test_audit_sink_receives_counts_never_values() -> None:
    seen: list[dict] = []
    guard = DlpGuardrail(
        policy=DlpPolicy(default=DlpAction.REDACT), audit_sink=seen.append
    )
    _inspect(guard, "emails a@b.com and c@d.com")
    assert seen and seen[0].get("email") == 2  # counts, not the addresses


# ------------------------------------------------------------------ presidio
def test_presidio_backend_detects_and_acts() -> None:
    class _FakeAnalyzer:
        def analyze(self, text: str) -> list[tuple[str, int, int]]:
            i = text.find("Jane Doe")
            return [("person", i, i + len("Jane Doe"))] if i >= 0 else []

    guard = DlpGuardrail(
        policy=DlpPolicy(actions={"person": DlpAction.REDACT}),
        analyzer=_FakeAnalyzer(),
    )
    v = _inspect(guard, "patient Jane Doe arrived")
    assert "Jane Doe" not in v.text
    assert "dlp:person" in v.flags


def test_presidio_adapter_maps_entity_types() -> None:
    class _R:
        def __init__(self, t, s, e):
            self.entity_type, self.start, self.end = t, s, e

    class _Engine:
        def analyze(self, *, text: str, language: str) -> list[_R]:
            return [_R("EMAIL_ADDRESS", 0, 5), _R("UNKNOWN_TYPE", 6, 9)]

    adapter = PresidioAnalyzerAdapter(analyzer=_Engine())
    spans = adapter.analyze("hello world")
    assert spans == [("email", 0, 5)]  # mapped; unknown type dropped


# -------------------------------------------------------------------- wiring
def test_dlp_registered_as_named_guardrail() -> None:
    pipe = build_guardrail_pipeline(["dlp"])  # default policy = redact
    v = pipe.inspect("reach me at a@b.com")
    assert "a@b.com" not in v.text


def test_named_dlp_guardrail_honors_block_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The spec/CLI path must resolve ``dlp`` via ``build_dlp_guardrail`` so
    # HIMMY_DLP_* policy config takes effect — a *:block policy BLOCKS, it does
    # not merely redact (the bare DlpGuardrail() default would redact-all).
    monkeypatch.setenv("HIMMY_DLP_POLICY", "*:block")
    pipe = build_guardrail_pipeline(["dlp"])
    v = pipe.inspect("reach me at a@b.com")
    assert v.allowed is False
    assert any("email" in r for r in v.reasons)
