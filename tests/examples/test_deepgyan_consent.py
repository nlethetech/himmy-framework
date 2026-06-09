"""End-to-end test for the DeepGyan consent worked example (WS4.6 A4).

Runs ``examples/deepgyan_consent/main.py`` through its public helpers and asserts the four
guarantees the scenario is built to prove:

* **persisted-vs-not** — a RETAIN-consented teacher (A) is persisted *encrypted* while a
  non-consenting teacher (B) leaves zero subject-bearing bytes on storage AND the spine;
* **training-set composition** — the consent-filtered export is empty until A grants TRAIN,
  then contains only A (never B);
* **audit-bundle verification** — the signed bundle verifies clean and flags a tampered
  cited record;
* **post-withdrawal crypto-shred** — withdrawal writes a tombstone, flips the ledger to
  WITHDRAWN, makes a fresh decision DENY, and leaves A's ciphertext unrecoverable.

A final check re-runs the *same* pipeline zero-config to prove the layer is purely opt-in
(the offline path stays byte-identical).

The example mutates ``os.environ`` (it is a runnable script, not a library), so the
``_isolated_consent_env`` fixture snapshots and restores the consent env around every test
to keep the suite hermetic.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

# ``cryptography`` is required: the example asserts field-level ciphertext on the spine.
pytest.importorskip("cryptography")

from examples.deepgyan_consent.main import (  # noqa: E402
    TEACHER_A_LESSON,
    TEACHER_B_LESSON,
    run_deepgyan_scenario,
    teacher_a_persisted_but_train_suppressed,
    teacher_b_is_not_persisted,
    withdraw_and_verify_erasure,
    zero_config_persists_verbatim,
)
from himmy.entities.records import EntityQuery  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_consent_env() -> Iterator[None]:
    """Snapshot + restore the consent env so the runnable example cannot leak state."""
    saved = {k: os.environ.get(k) for k in ("HIMMY_CONSENT", "HIMMY_CONSENT_FILE")}
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.asyncio
async def test_non_consenting_teacher_is_not_persisted() -> None:
    """teacher_b never opts in → NOTHING of theirs reaches storage or the spine."""
    result = await run_deepgyan_scenario()
    container = result.container

    assert teacher_b_is_not_persisted(container)

    # Belt-and-suspenders: no subject-bearing record for teacher_b on the spine, and the
    # transcript kinds for them are absent entirely (skip-on-deny, not merely blanked).
    spine = container.entity_registry
    assert spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_b"})) == []
    assert spine.list_by_kind("message") != []  # A's messages exist…
    assert all(
        r.metadata.get("subject_id") != "teacher_b"
        for r in spine.list_by_kind("message")
    )
    # The verbatim lesson text never appears anywhere on the spine for B.
    blob = "".join(
        str(r.payload)
        for kind in ("run_event", "message", "chat_thread")
        for r in spine.list_by_kind(kind)
    )
    assert TEACHER_B_LESSON not in blob


@pytest.mark.asyncio
async def test_consenting_teacher_persisted_but_train_suppressed() -> None:
    """teacher_a is served + persisted encrypted, but TRAIN capture is suppressed."""
    result = await run_deepgyan_scenario()
    container = result.container

    assert await teacher_a_persisted_but_train_suppressed(container)

    # A's transcript reached the spine (RETAIN granted) tagged with their subject…
    spine = container.entity_registry
    tagged = spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_a"}))
    assert tagged
    assert any(r.kind == "run_event" for r in tagged)

    # …but no run event carries a raw-I/O blob or the verbatim prompt (TRAIN denied).
    a_events = [r for r in tagged if r.kind == "run_event"]
    assert a_events
    assert all("io" not in (r.payload or {}) for r in a_events)
    assert all("rendered_prompt" not in (r.payload or {}) for r in a_events)
    assert all(TEACHER_A_LESSON not in str(r.payload) for r in a_events)

    # The persisted user message is encrypted (never cleartext) on the spine.
    from himmy.services.storage.encryption import ENC_PREFIX

    user_msgs = [
        r
        for r in spine.list_by_kind("message")
        if r.metadata.get("subject_id") == "teacher_a"
        and r.payload.get("role") == "user"
    ]
    assert user_msgs
    assert all(m.payload["content"].startswith(ENC_PREFIX) for m in user_msgs)
    assert all(TEACHER_A_LESSON not in m.payload["content"] for m in user_msgs)


@pytest.mark.asyncio
async def test_training_set_composition() -> None:
    """The consent-filtered export is empty until A grants TRAIN, then only A."""
    result = await run_deepgyan_scenario()

    # Before A grants TRAIN: provably empty, even though A's run WAS retained.
    assert result.training_corpus_before_train_grant == []

    # After A grants TRAIN: exactly A's run(s); teacher_b never appears (no run persisted).
    after = result.training_corpus_after_train_grant
    assert after, "A's run should appear once TRAIN is granted"
    subjects = {r.subject_id for r in after}
    assert subjects == {"teacher_a"}
    assert "teacher_b" not in subjects


@pytest.mark.asyncio
async def test_signed_audit_bundle_verifies_and_detects_tampering() -> None:
    """The signed bundle verifies clean and flags a tampered cited record."""
    result = await run_deepgyan_scenario()
    assert result.bundle_verified
    assert result.bundle_detects_tampering


@pytest.mark.asyncio
async def test_withdrawal_crypto_shreds_and_denies() -> None:
    """Withdrawal → tombstone + ledger WITHDRAWN + fresh DENY + unrecoverable ciphertext."""
    result = await run_deepgyan_scenario()
    erasure = await withdraw_and_verify_erasure(result.container)

    assert erasure["message_was_encrypted"]
    assert erasure["tombstone_present"]
    assert erasure["ledger_withdrawn"]
    assert erasure["fresh_decision_deny"]
    assert erasure["ciphertext_unrecoverable"]


@pytest.mark.asyncio
async def test_zero_config_is_opt_in_and_persists_verbatim() -> None:
    """With consent OFF the pipeline is bare and persists a non-consenting prompt verbatim."""
    assert await zero_config_persists_verbatim()
