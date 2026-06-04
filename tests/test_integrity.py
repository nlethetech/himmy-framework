"""Tests for tamper-evident audit: content hashes + signed bundle verification."""

from __future__ import annotations

from himmy.entities import (
    EntityRegistry,
    content_hash,
    export_audit_bundle,
    verify_audit_bundle,
)
from himmy.entities.records import EntityLink, EntityRecord

SECRET = "audit-secret-key"


def _graph() -> tuple[EntityRecord, EntityRecord, EntityLink]:
    reg = EntityRegistry()
    a = reg.register(
        EntityRecord.create(
            stable_id="A", version=1, kind="chat_thread", payload={"q": "buy ACME"}
        )
    )
    b = reg.register(
        EntityRecord.create(
            stable_id="B", version=1, kind="persona", payload={"name": "analyst"}
        )
    )
    link = reg.link(
        from_record_id=a.record_id, to_record_id=b.record_id, relation="uses_persona"
    )
    return a, b, link


def test_content_hash_is_canonical_and_content_sensitive() -> None:
    """Hash is stable across key order but changes with content."""
    r1 = EntityRecord.create(
        stable_id="A", version=1, kind="k", payload={"a": 1, "b": 2}
    )
    r2 = EntityRecord.create(
        stable_id="A", version=1, kind="k", payload={"b": 2, "a": 1}
    )
    r3 = EntityRecord.create(
        stable_id="A", version=1, kind="k", payload={"a": 1, "b": 3}
    )
    assert content_hash(r1) == content_hash(r2)
    assert content_hash(r1) != content_hash(r3)


def test_clean_graph_verifies() -> None:
    """An unmodified graph verifies with a valid signature."""
    a, b, link = _graph()
    bundle = export_audit_bundle([a, b], [link], secret=SECRET)
    result = verify_audit_bundle(bundle, [a, b], [link], secret=SECRET)
    assert result.ok is True
    assert result.signature_valid is True


def test_detects_in_place_record_tampering() -> None:
    """A record edited in place (same id, new payload) is flagged tampered.

    This is the exact gap content-addressing on (kind, stable_id, version) alone
    left open: the record_id is unchanged, but the content is not.
    """
    a, b, link = _graph()
    bundle = export_audit_bundle([a, b], [link], secret=SECRET)
    tampered_a = EntityRecord.create(
        stable_id="A", version=1, kind="chat_thread", payload={"q": "buy WORTHLESSCO"}
    )
    assert tampered_a.record_id == a.record_id  # identity unchanged...
    result = verify_audit_bundle(bundle, [tampered_a, b], [link], secret=SECRET)
    assert result.ok is False
    assert a.record_id in result.tampered_record_ids  # ...but content tamper caught


def test_detects_missing_and_added_records() -> None:
    """Dropping or inserting a record is reported precisely."""
    a, b, link = _graph()
    bundle = export_audit_bundle([a, b], [link], secret=SECRET)

    dropped = verify_audit_bundle(bundle, [a], [link], secret=SECRET)
    assert b.record_id in dropped.missing_record_ids and dropped.ok is False

    extra = EntityRecord.create(stable_id="C", version=1, kind="prompt", payload={})
    inserted = verify_audit_bundle(bundle, [a, b, extra], [link], secret=SECRET)
    assert extra.record_id in inserted.added_record_ids and inserted.ok is False


def test_detects_link_tampering() -> None:
    """Rewriting an edge's relation is flagged."""
    a, b, link = _graph()
    bundle = export_audit_bundle([a, b], [link], secret=SECRET)
    forged = EntityLink(
        link_id=link.link_id,
        from_record_id=link.from_record_id,
        to_record_id=link.to_record_id,
        relation="HACKED",
    )
    result = verify_audit_bundle(bundle, [a, b], [forged], secret=SECRET)
    assert link.link_id in result.tampered_link_ids and result.ok is False


def test_wrong_secret_fails_signature() -> None:
    """A bundle verified with the wrong secret fails on the signature."""
    a, b, link = _graph()
    bundle = export_audit_bundle([a, b], [link], secret=SECRET)
    result = verify_audit_bundle(bundle, [a, b], [link], secret="not-the-secret")
    assert result.signature_valid is False
    assert result.ok is False
