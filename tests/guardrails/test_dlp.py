"""WS4.1 — DLP as policy: classify → allow/redact/tokenize/block, audited."""

from __future__ import annotations

from himmy.services.guardrails import build_guardrail_pipeline
from himmy.services.guardrails.dlp import (
    DlpAction,
    DlpGuardrail,
    DlpPolicy,
    PresidioAnalyzerAdapter,
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
