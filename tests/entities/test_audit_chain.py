"""Opt-in hash-chain audit mode: order-sensitive, pinpoints breaks.

The default Merkle bundle sorts its leaves, so a reordered trail produces the
same root — these tests prove the chain mode closes exactly that gap (reorder
and mid-history deletion are detected and localised) without touching the
Merkle default.
"""

from __future__ import annotations

from himmy.entities.integrity import (
    CHAIN_GENESIS_HASH,
    export_audit_bundle,
    export_audit_chain,
    verify_audit_bundle,
    verify_audit_chain,
)
from himmy.entities.records import EntityRecord

SECRET = "chain-test-secret"  # noqa: S105 - test fixture, not a real credential


def _trail(n: int = 4) -> list[EntityRecord]:
    return [
        EntityRecord.create(
            stable_id=f"evt-{i}", version=1, kind="security_event", payload={"seq": i}
        )
        for i in range(n)
    ]


def test_intact_chain_verifies() -> None:
    records = _trail()
    bundle = export_audit_chain(records, secret=SECRET)
    assert bundle.entries[0].prev_hash == CHAIN_GENESIS_HASH
    assert bundle.tip_hash == bundle.entries[-1].chain_hash
    result = verify_audit_chain(bundle, records, secret=SECRET)
    assert result.ok and result.signature_valid and result.chain_intact
    assert result.first_break_index is None
    assert not result.tampered_record_ids
    assert not result.reordered_record_ids


def test_empty_chain_verifies() -> None:
    bundle = export_audit_chain([], secret=SECRET)
    assert bundle.tip_hash == CHAIN_GENESIS_HASH
    assert verify_audit_chain(bundle, [], secret=SECRET).ok


def test_reordering_detected_by_chain_but_not_merkle() -> None:
    records = _trail()
    chain = export_audit_chain(records, secret=SECRET)
    merkle = export_audit_bundle(records, [], secret=SECRET)

    swapped = [records[0], records[2], records[1], records[3]]

    # The order-independent Merkle default is blind to the reorder...
    assert verify_audit_bundle(merkle, swapped, [], secret=SECRET).ok

    # ...the chain is not, and it pinpoints the first divergent position.
    result = verify_audit_chain(chain, swapped, secret=SECRET)
    assert not result.ok
    assert result.signature_valid  # the bundle itself is untouched
    assert not result.chain_intact
    assert result.first_break_index == 1
    assert set(result.reordered_record_ids) == {
        records[1].record_id,
        records[2].record_id,
    }
    assert not result.missing_record_ids and not result.added_record_ids


def test_mid_history_deletion_detected_and_localised() -> None:
    records = _trail()
    bundle = export_audit_chain(records, secret=SECRET)
    truncated = [records[0]] + records[2:]  # drop the second event
    result = verify_audit_chain(bundle, truncated, secret=SECRET)
    assert not result.ok and not result.chain_intact
    assert result.first_break_index == 1
    assert records[1].record_id in result.missing_record_ids
    # the survivors shifted position, so they surface as reordered, not missing
    assert records[2].record_id in result.reordered_record_ids


def test_in_place_tamper_detected_with_index() -> None:
    records = _trail()
    bundle = export_audit_chain(records, secret=SECRET)
    tampered = list(records)
    tampered[2] = tampered[2].model_copy(update={"payload": {"seq": 999}})
    result = verify_audit_chain(bundle, tampered, secret=SECRET)
    assert not result.ok and not result.chain_intact
    assert result.first_break_index == 2
    assert result.tampered_record_ids == [records[2].record_id]


def test_unsigned_append_detected_as_added() -> None:
    records = _trail()
    bundle = export_audit_chain(records, secret=SECRET)
    extra = EntityRecord.create(
        stable_id="evt-extra", version=1, kind="security_event", payload={"seq": 99}
    )
    result = verify_audit_chain(bundle, [*records, extra], secret=SECRET)
    assert not result.ok and not result.chain_intact
    assert result.first_break_index == len(records)
    assert result.added_record_ids == [extra.record_id]


def test_wrong_secret_fails_signature() -> None:
    records = _trail()
    bundle = export_audit_chain(records, secret=SECRET)
    result = verify_audit_chain(bundle, records, secret="not-the-secret")
    assert not result.signature_valid and not result.ok
    assert result.chain_intact  # the data itself is fine; only trust is broken
