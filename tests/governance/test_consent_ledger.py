"""WS4.6 — the consent ledger (immutable version chain + withdraw⇒erase)."""

from __future__ import annotations

from himmy.entities.registry import EntityRegistry
from himmy.services.governance.consent import (
    CONSENT_KIND,
    ConsentPolicy,
    ConsentState,
    Effect,
    Purpose,
    consent_stable_id,
)
from himmy.services.governance.consent_ledger import ConsentLedger
from himmy.services.governance.retention import (
    ERASURE_KIND,
    RetentionService,
    SubjectKeyVault,
)


def _governed_ledger() -> tuple[ConsentLedger, EntityRegistry, SubjectKeyVault]:
    registry = EntityRegistry()
    vault = SubjectKeyVault()
    ledger = ConsentLedger(
        registry,
        policy=ConsentPolicy(governed=True),
        retention_service=RetentionService(registry, key_vault=vault),
    )
    return ledger, registry, vault


def test_version_chain_grant_withdraw_regrant() -> None:
    ledger, registry, _ = _governed_ledger()
    ledger.grant("alice", Purpose.RETAIN)
    ledger.withdraw("alice", Purpose.RETAIN)
    ledger.grant("alice", Purpose.RETAIN)

    history = ledger.history("alice", Purpose.RETAIN)
    assert [r.state for r in history] == [
        ConsentState.GRANTED,
        ConsentState.WITHDRAWN,
        ConsentState.GRANTED,
    ]
    # Immutable, content-addressed chain on the spine.
    versions = registry.get_history(consent_stable_id("alice", Purpose.RETAIN))
    assert [r.version for r in versions] == [1, 2, 3]


def test_decision_reflects_latest_state() -> None:
    ledger, _, _ = _governed_ledger()
    # No record yet → UNKNOWN → governed default.
    assert ledger.decision("bob", Purpose.RETAIN).effect is Effect.DENY
    ledger.grant("bob", Purpose.RETAIN)
    assert ledger.decision("bob", Purpose.RETAIN).effect is Effect.ALLOW
    ledger.deny("bob", Purpose.RETAIN)
    assert ledger.decision("bob", Purpose.RETAIN).effect is Effect.DENY


def test_withdraw_crypto_shreds_and_denies() -> None:
    ledger, registry, vault = _governed_ledger()
    ledger.grant("carol", Purpose.RETAIN)
    ledger.grant("carol", Purpose.TRAIN)
    vault.key_for("carol")  # subject has a live key (data was encrypted under it)

    ledger.withdraw("carol")  # all purposes

    # Latest state for every purpose is WITHDRAWN → DENY.
    assert ledger.decision("carol", Purpose.RETAIN).effect is Effect.DENY
    assert ledger.decision("carol", Purpose.TRAIN).effect is Effect.DENY
    # Crypto-shred + signed tombstone.
    assert vault.has("carol") is False
    tombstones = registry.list_by_kind(ERASURE_KIND)
    assert any(t.payload["subject_id"] == "carol" for t in tombstones)


def test_query_lists_consents_by_subject_metadata() -> None:
    ledger, registry, _ = _governed_ledger()
    ledger.grant("dave", Purpose.RETAIN)
    ledger.grant("dave", Purpose.INFER)
    records = registry.list_by_kind(CONSENT_KIND)
    subjects = {r.metadata["subject_id"] for r in records}
    assert subjects == {"dave"}
    purposes = {r.metadata["purpose"] for r in records}
    assert purposes == {"retain", "infer"}
