"""WS4.6 — ConsentAwareRegistry: the spine gate (skip-on-deny + encrypt-on-allow)."""

from __future__ import annotations

import pytest

from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.services.audit.log import SecurityAuditLog
from himmy.services.governance.consent import ConsentPolicy, Purpose
from himmy.services.governance.consent_ledger import ConsentLedger
from himmy.services.governance.consent_registry import ConsentAwareRegistry

_GATED = {"run_event"}


def _subject_of(record: EntityRecord) -> str | None:
    value = record.payload.get("subject_id")
    return str(value) if value else None


def _event(
    subject_id: str | None, *, stable: str, text: str = "secret"
) -> EntityRecord:
    payload: dict[str, object] = {"text": text}
    if subject_id is not None:
        payload["subject_id"] = subject_id
    return EntityRecord.create(
        stable_id=stable, version=1, kind="run_event", payload=payload
    )


def _wrap(
    inner: EntityRegistry, **kw: object
) -> tuple[ConsentAwareRegistry, ConsentLedger, SecurityAuditLog]:
    audit = SecurityAuditLog(inner)
    ledger = ConsentLedger(inner, policy=ConsentPolicy(governed=True))
    kw.setdefault("gated_kinds", _GATED)
    reg = ConsentAwareRegistry(
        inner,
        decider=ledger.decision,
        subject_extractor=_subject_of,
        audit=audit,
        **kw,  # type: ignore[arg-type]
    )
    return reg, ledger, audit


def test_consented_subject_event_reaches_spine() -> None:
    inner = EntityRegistry()
    reg, ledger, _ = _wrap(inner)
    ledger.grant("alice", Purpose.RETAIN)
    rec = _event("alice", stable="ev1")
    reg.register(rec)
    assert inner.get(rec.record_id) is not None


def test_unconsented_subject_never_reaches_spine() -> None:
    inner = EntityRegistry()
    reg, _, audit = _wrap(inner)
    rec = _event("bob", stable="ev2")  # no consent → governed DENY
    reg.register(rec)
    assert inner.get(rec.record_id) is None
    assert inner.list_by_kind("run_event") == []
    assert audit.recent(event_type="consent_denied_persist")


def test_unresolved_subject_fails_closed() -> None:
    inner = EntityRegistry()
    reg, _, audit = _wrap(inner)
    rec = _event(None, stable="ev3")  # subject-bearing kind, no subject in payload
    reg.register(rec)
    assert inner.get(rec.record_id) is None
    assert audit.recent(event_type="consent_denied_persist")


def test_infrastructure_kind_passes_through_untouched() -> None:
    inner = EntityRegistry()
    reg, _, _ = _wrap(inner)
    persona = EntityRecord.create(
        stable_id="p1", version=1, kind="persona", payload={"name": "tutor"}
    )
    reg.register(persona)
    stored = inner.get(persona.record_id)
    assert stored is not None and stored.payload == {
        "name": "tutor"
    }  # not gated, not encrypted


def test_allow_encrypts_subject_fields_under_shreddable_key() -> None:
    pytest.importorskip("cryptography")
    from himmy.services.governance.retention import SubjectKeyVault
    from himmy.services.storage.encryption import ENC_PREFIX

    inner = EntityRegistry()
    vault = SubjectKeyVault()
    reg, ledger, _ = _wrap(
        inner, key_vault=vault, encrypted_fields_for=lambda _kind: ("text",)
    )
    ledger.grant("carol", Purpose.RETAIN)
    rec = _event("carol", stable="ev4", text="carol's prompt")
    reg.register(rec)

    stored = inner.get(rec.record_id)
    assert stored is not None
    # The content-addressed id is unchanged; the payload field is now ciphertext.
    assert stored.record_id == rec.record_id
    assert stored.payload["text"].startswith(ENC_PREFIX)
    assert "carol's prompt" not in stored.payload["text"]

    # Crypto-shred: destroying the subject key makes the ciphertext unrecoverable. A later
    # re-onboarding of the same subject mints a FRESH key (so re-consent doesn't crash), but
    # that new key cannot recover the pre-erasure ciphertext — the shred is preserved.
    token = stored.payload["text"]
    assert token  # ciphertext exists pre-shred
    vault.destroy("carol")
    with pytest.raises(Exception):  # noqa: B017,PT011 - re-minted key never decrypts old token
        vault.encryptor_for("carol").decrypt(token, aad=b"carol")


def test_chat_thread_nested_message_content_is_encrypted_on_spine() -> None:
    """The richest record kind (chat_thread) carries no plaintext conversation body.

    Regression for the right-to-erasure hole: ``chat_thread`` maps to the
    ``MESSAGES_CONTENT_FIELD`` marker, so the registry must walk ``messages[*].content``
    through the subject cipher before the immutable spine sees it — otherwise crypto-shred
    could never render the conversation unrecoverable.
    """
    pytest.importorskip("cryptography")
    from himmy.services.governance.consent_registry import MESSAGES_CONTENT_FIELD
    from himmy.services.governance.retention import SubjectKeyVault
    from himmy.services.storage.encryption import ENC_PREFIX

    inner = EntityRegistry()
    vault = SubjectKeyVault()
    reg, ledger, _ = _wrap(
        inner,
        gated_kinds={"chat_thread"},
        key_vault=vault,
        encrypted_fields_for=lambda kind: (
            (MESSAGES_CONTENT_FIELD,) if kind == "chat_thread" else ()
        ),
    )
    ledger.grant("dave", Purpose.RETAIN)
    rec = EntityRecord.create(
        stable_id="th1",
        version=1,
        kind="chat_thread",
        payload={
            "thread_id": "th1",
            "subject_id": "dave",
            "messages": [
                {"role": "user", "content": "dave's bank PIN is 4242"},
                {"role": "assistant", "content": "noted"},
            ],
        },
    )
    reg.register(rec)

    stored = inner.get(rec.record_id)
    assert stored is not None
    # No plaintext conversation body anywhere in the spine record.
    import json

    blob = json.dumps(stored.payload)
    assert "4242" not in blob and "noted" not in blob
    assert stored.payload["messages"][0]["content"].startswith(ENC_PREFIX)

    # Crypto-shred the subject: the nested ciphertext is now permanently unreadable. A later
    # re-onboarding mints a FRESH key (so re-consent doesn't crash), but that new key cannot
    # recover the pre-erasure ciphertext — the shred is preserved.
    token = stored.payload["messages"][0]["content"]
    assert token  # ciphertext exists pre-shred
    vault.destroy("dave")
    with pytest.raises(Exception):  # noqa: B017,PT011 - re-minted key never decrypts old token
        vault.encryptor_for("dave").decrypt(token, aad=b"dave")
