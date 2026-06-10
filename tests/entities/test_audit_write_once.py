"""Write-once guard: audit-kind records in the in-memory registry are aliasing-proof.

``EntityRecord`` is frozen, but its ``payload``/``metadata`` dicts are mutable —
without the guard, any holder of a stored reference could rewrite audit history
through those dicts. The registry now deep-copies audit-kind records on every
write and read, so these tests mutate every reference a caller can obtain and
assert the stored history never moves.
"""

from __future__ import annotations

from himmy.entities.records import EntityQuery, EntityRecord
from himmy.entities.registry import AUDIT_RECORD_KINDS, EntityRegistry


def _event(payload: dict[str, object] | None = None) -> EntityRecord:
    return EntityRecord.create(
        stable_id="evt-1",
        version=1,
        kind="security_event",
        payload=payload or {"outcome": "deny"},
        metadata={"actor": "alice"},
    )


def test_mutating_a_read_copy_does_not_rewrite_history() -> None:
    registry = EntityRegistry()
    registry.register(_event())

    leaked = registry.get(_event().record_id)
    assert leaked is not None
    leaked.payload["outcome"] = "allow"  # attempted history rewrite via alias
    leaked.metadata["actor"] = "mallory"

    stored = registry.get(_event().record_id)
    assert stored is not None
    assert stored.payload == {"outcome": "deny"}
    assert stored.metadata == {"actor": "alice"}


def test_mutating_the_callers_original_after_register_is_inert() -> None:
    registry = EntityRegistry()
    record = _event()
    returned = registry.register(record)

    record.payload["outcome"] = "allow"  # caller still holds the original dicts
    returned.payload["outcome"] = "allow"  # ...and the returned record

    stored = registry.get(record.record_id)
    assert stored is not None
    assert stored.payload == {"outcome": "deny"}


def test_all_read_paths_return_shielded_audit_copies() -> None:
    registry = EntityRegistry()
    record = registry.register(_event())

    for fetched in (
        registry.get_latest(record.stable_id),
        registry.get_history(record.stable_id)[0],
        registry.list_by_kind("security_event")[0],
        registry.query(EntityQuery(kind="security_event"))[0],
    ):
        assert fetched is not None
        fetched.payload["outcome"] = "allow"

    stored = registry.get(record.record_id)
    assert stored is not None
    assert stored.payload == {"outcome": "deny"}


def test_audit_record_reads_compare_equal_to_what_was_registered() -> None:
    registry = EntityRegistry()
    record = _event()
    registry.register(record)
    assert registry.get(record.record_id) == record  # value semantics preserved


def test_non_audit_kinds_keep_zero_copy_identity() -> None:
    registry = EntityRegistry()
    record = EntityRecord.create(
        stable_id="p-1", version=1, kind="persona", payload={"name": "x"}
    )
    stored = registry.register(record)
    assert stored is record
    assert registry.get(record.record_id) is record


def test_custom_audit_kinds_extend_the_guard() -> None:
    registry = EntityRegistry(audit_kinds=AUDIT_RECORD_KINDS | {"ledger_entry"})
    record = EntityRecord.create(
        stable_id="l-1", version=1, kind="ledger_entry", payload={"amount": 10}
    )
    registry.register(record)
    record.payload["amount"] = 999
    stored = registry.get(record.record_id)
    assert stored is not None
    assert stored.payload == {"amount": 10}
