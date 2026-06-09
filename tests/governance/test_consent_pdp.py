"""WS4.6 — the consent policy-decision-point (pure, zero-I/O)."""

from __future__ import annotations

import json

import pytest

from himmy.services.governance.consent import (
    ConsentPolicy,
    ConsentState,
    Effect,
    Purpose,
    build_consent_policy,
    consent_stable_id,
)


def test_ungoverned_allows_everything() -> None:
    policy = ConsentPolicy(governed=False)
    for purpose in Purpose:
        for state in ConsentState:
            decision = policy.decide("alice", purpose, state)
            assert decision.effect is Effect.ALLOW
            assert decision.allowed is True


def test_governed_unknown_defaults_deny_retain_train_ephemeral_infer() -> None:
    policy = ConsentPolicy(governed=True)
    assert (
        policy.decide("a", Purpose.RETAIN, ConsentState.UNKNOWN).effect is Effect.DENY
    )
    assert policy.decide("a", Purpose.TRAIN, ConsentState.UNKNOWN).effect is Effect.DENY
    infer = policy.decide("a", Purpose.INFER, ConsentState.UNKNOWN)
    assert infer.effect is Effect.EPHEMERAL
    assert infer.allowed is False  # ephemeral may be USED live but never persisted


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (ConsentState.GRANTED, Effect.ALLOW),
        (ConsentState.DENIED, Effect.DENY),
        (ConsentState.WITHDRAWN, Effect.DENY),
    ],
)
def test_governed_known_states(state: ConsentState, expected: Effect) -> None:
    policy = ConsentPolicy(governed=True)
    for purpose in (Purpose.RETAIN, Purpose.TRAIN, Purpose.INFER):
        assert policy.decide("a", purpose, state).effect is expected


def test_consent_stable_id_is_deterministic_and_purpose_scoped() -> None:
    assert consent_stable_id("alice", Purpose.RETAIN) == consent_stable_id(
        "alice", "retain"
    )
    assert consent_stable_id("alice", Purpose.RETAIN) != consent_stable_id(
        "alice", Purpose.TRAIN
    )
    assert consent_stable_id("alice", Purpose.RETAIN) != consent_stable_id(
        "bob", Purpose.RETAIN
    )


def test_build_consent_policy_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIMMY_CONSENT", raising=False)
    assert build_consent_policy().governed is False


def test_build_consent_policy_on_and_file_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    path = tmp_path / "consent.json"  # type: ignore[operator]
    path.write_text(json.dumps({"infer": "deny", "*": "deny"}))
    monkeypatch.setenv("HIMMY_CONSENT", "on")
    monkeypatch.setenv("HIMMY_CONSENT_FILE", str(path))
    policy = build_consent_policy()
    assert policy.governed is True
    # The file overrode INFER's default (EPHEMERAL) to DENY.
    assert policy.decide("a", Purpose.INFER, ConsentState.UNKNOWN).effect is Effect.DENY
