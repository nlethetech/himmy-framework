"""Property tests: guardrail pipeline robustness, redaction idempotence, DLP.

Hypothesis drives three guarantees the regex guardrails must hold under fuzz:
arbitrary unicode never crashes the pipeline, redaction is idempotent (guarding
already-redacted text changes nothing), and no seeded raw PII value survives in
the output. DLP additionally must preserve text on ``block`` and round-trip
``tokenize`` through the vault. PII seeds are embedded space-separated between
letters-only filler so each seed is reliably matchable; full-seed substring
absence is then a sound property even when adjacent seeds merge into one larger
match. Skips gracefully when ``hypothesis`` is not installed (dev-only).
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from himmy.services.guardrails.base import GuardrailPipeline  # noqa: E402
from himmy.services.guardrails.builtins import (  # noqa: E402
    GroundingGuardrail,
    InjectionGuardrail,
    NepalPIIGuardrail,
    PIIGuardrail,
    build_guardrail_pipeline,
)
from himmy.services.guardrails.dlp import (  # noqa: E402
    DlpAction,
    DlpGuardrail,
    DlpPolicy,
    TokenVault,
)

# Moderate example counts, no deadline, derandomized => fast and seed-stable in CI.
_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True, database=None)

#: Known-good raw PII values, each verified to redact when space-bounded.
_PII_SEEDS = (
    "alice@example.com",
    "4111111111111111",  # Luhn-valid card
    "123-45-6789",  # SSN
    "10.0.0.1",  # IPv4
    "sk-abcdefghijklmnop1234",  # API key
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",  # JWT
    "00:1A:2B:3C:4D:5E",  # MAC
    "+1 415 555 0132",  # phone
    "https://user:pass123@example.com/path",  # URL credentials
    "DE44500105175407324931",  # IBAN
)
_NEPAL_SEEDS = (
    "+9779812345678",  # +977 international form
    "9812345678",  # domestic mobile
    "12-34-56-78901",  # citizenship number
    "PAN: 123456789",  # labelled PAN
)

#: Letters-only filler words: can never match a PII rule (no digits/@/:/./-).
_FILLER_WORDS = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    min_size=1,
    max_size=8,
)


@st.composite
def _documents(
    draw: st.DrawFn, seed_pool: tuple[str, ...], *, min_seeds: int = 0
) -> tuple[str, list[str]]:
    """A space-joined document of PII seeds + filler; returns (text, seeds used)."""
    seeds = draw(st.lists(st.sampled_from(seed_pool), min_size=min_seeds, max_size=4))
    fillers = draw(st.lists(_FILLER_WORDS, max_size=5))
    fragments = draw(st.permutations(seeds + fillers))
    return " ".join(fragments), seeds


@_SETTINGS
@given(text=st.text(max_size=400))
def test_full_builtin_pipeline_is_total_on_arbitrary_unicode(text: str) -> None:
    """Every builtin guardrail chained: arbitrary unicode never crashes."""
    pipeline = build_guardrail_pipeline(
        ["pii", "injection", "nepal_pii", "grounding", "dlp"]
    )
    for stage in ("input", "output"):
        verdict = pipeline.inspect(text, context={"stage": stage})
        assert isinstance(verdict.text, str)
        assert isinstance(verdict.allowed, bool)


@_SETTINGS
@given(case=_documents(_PII_SEEDS), tail=st.text(max_size=120))
def test_redaction_is_idempotent(case: tuple[str, list[str]], tail: str) -> None:
    """Guarding already-redacted text changes nothing."""
    text = case[0] + " " + tail
    pipeline = GuardrailPipeline([PIIGuardrail(), NepalPIIGuardrail()])
    once = pipeline.inspect(text).text
    assert pipeline.inspect(once).text == once


@_SETTINGS
@given(case=_documents(_PII_SEEDS))
def test_no_seeded_pii_survives_redaction(case: tuple[str, list[str]]) -> None:
    """The redacted output never contains a seeded raw PII value."""
    text, seeds = case
    verdict = PIIGuardrail().inspect(text, context={})
    assert verdict.allowed  # PII guardrail redacts, never blocks
    for seed in seeds:
        assert seed not in verdict.text


@_SETTINGS
@given(case=_documents(_NEPAL_SEEDS))
def test_no_seeded_nepal_pii_survives_redaction(case: tuple[str, list[str]]) -> None:
    """Nepal-specific seeds (phones, citizenship, PAN) are always redacted."""
    text, seeds = case
    verdict = NepalPIIGuardrail().inspect(text, context={})
    assert verdict.allowed
    for seed in seeds:
        assert seed not in verdict.text


@_SETTINGS
@given(case=_documents(_PII_SEEDS))
def test_dlp_default_policy_strips_all_seeds(case: tuple[str, list[str]]) -> None:
    """The default DLP policy (redact everything) removes every seeded value."""
    text, seeds = case
    verdict = DlpGuardrail().inspect(text, context={})
    assert verdict.allowed
    for seed in seeds:
        assert seed not in verdict.text


@_SETTINGS
@given(case=_documents(_PII_SEEDS, min_seeds=1))
def test_dlp_block_policy_denies_without_rewriting(
    case: tuple[str, list[str]],
) -> None:
    """A blanket ``block`` policy denies and returns the ORIGINAL text untouched."""
    text, _ = case
    guard = DlpGuardrail(policy=DlpPolicy(default=DlpAction.BLOCK))
    verdict = guard.inspect(text, context={})
    assert not verdict.allowed
    assert verdict.text == text


@_SETTINGS
@given(
    value=st.text(min_size=1, max_size=40),
    label=st.sampled_from(("email", "card", "phone")),
)
def test_token_vault_round_trips_any_value(value: str, label: str) -> None:
    """tokenize → detokenize restores the value; same value → same token."""
    vault = TokenVault()
    token = vault.tokenize(label, value)
    assert vault.tokenize(label, value) == token
    assert vault.detokenize(f"a {token} b") == f"a {value} b"


@_SETTINGS
@given(
    emails=st.lists(
        st.builds(
            "{}@{}.com".format,
            st.text(alphabet="abcdefghij0123456789", min_size=1, max_size=8),
            st.text(alphabet="klmnopqrst", min_size=1, max_size=8),
        ),
        min_size=1,
        max_size=3,
    ),
    fillers=st.lists(_FILLER_WORDS, max_size=4),
    data=st.data(),
)
def test_dlp_tokenize_round_trips_whole_documents(
    emails: list[str], fillers: list[str], data: st.DataObject
) -> None:
    """``email:tokenize`` replaces every email; detokenize restores the document."""
    fragments = data.draw(st.permutations(emails + fillers))
    text = " ".join(fragments)
    vault = TokenVault()
    guard = DlpGuardrail(
        policy=DlpPolicy.parse("email:tokenize", default=DlpAction.ALLOW),
        vault=vault,
    )
    verdict = guard.inspect(text, context={})
    assert verdict.allowed
    for email in emails:
        assert email not in verdict.text
    assert vault.detokenize(verdict.text) == text


@_SETTINGS
@given(text=st.text(max_size=300))
def test_injection_and_grounding_never_rewrite_input(text: str) -> None:
    """Injection only blocks (never rewrites); grounding is an input-stage no-op."""
    injection = InjectionGuardrail().inspect(text, context={})
    assert injection.text == text
    grounding = GroundingGuardrail().inspect(text, context={"stage": "input"})
    assert grounding.allowed
    assert grounding.text == text
