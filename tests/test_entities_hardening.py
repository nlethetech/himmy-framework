"""Expanded hardening tests for the entities kernel (SE-8/9/10/12).

These lock in the production behaviour the kernel audit hardened, complementing
``tests/test_entities.py``:

- SE-9  EntityRecord is ``frozen`` — mutation of *any* field raises and the
        deterministic ``record_id`` can never drift out of sync with its triple.
- SE-10 register() raises on a true content-address collision (same
        ``(kind, stable_id, version)`` with different payload *or* metadata) and
        never silently divergences in-memory state from what the caller wrote.
- SE-8  ``metadata_contains`` implements Postgres ``@>`` JSONB-containment
        semantics exactly (nested dicts, list-membership, scalar equality, type
        mismatches), and ``EntityQuery`` supports kind-less / stable_id-only /
        metadata-only / no-predicate queries equivalently.
- SE-12 mutable model defaults are per-instance (``default_factory``), not shared
        class-level literals.

All tests are plain ``def test_*`` functions; no event loop is needed because the
in-memory registry surface is synchronous.
"""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from opensims.core.errors import OpenSimsError
from opensims.entities.records import (
    EntityLink,
    EntityQuery,
    EntityRecord,
    metadata_contains,
    record_id_for,
    stable_id_for,
)
from opensims.entities.registry import EntityRegistry


# --------------------------------------------------------------- local fixtures
@pytest.fixture()
def reg() -> EntityRegistry:
    """A fresh in-memory registry (area fixture, defined locally per the brief)."""
    return EntityRegistry()


def _sid(prefix: str) -> str:
    """A unique, namespaced stable id for isolation between tests."""
    return stable_id_for(f"{prefix}-{uuid.uuid4()}", namespace="persona")


# --------------------------------------------------------- SE-9: frozen records
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", 99),
        ("kind", "tampered"),
        ("stable_id", "other"),
        ("payload", {"x": 1}),
        ("metadata", {"y": 2}),
        ("record_id", "forged"),
        ("created_at", "2000-01-01T00:00:00Z"),
    ],
)
def test_entity_record_every_field_is_frozen(field: str, value: object) -> None:
    """Mutating ANY field on a frozen EntityRecord raises (SE-9)."""
    rec = EntityRecord.create(
        stable_id=_sid("frz"), version=1, kind="persona", payload={"v": 1}
    )
    with pytest.raises(ValidationError):
        setattr(rec, field, value)


def test_entity_record_id_cannot_drift_from_its_triple() -> None:
    """record_id is content-addressed and immutable, so it never goes stale (SE-9)."""
    sid = _sid("drift")
    rec = EntityRecord(stable_id=sid, kind="persona", version=3)
    expected = record_id_for(stable_id=sid, version=3, kind="persona")
    assert rec.record_id == expected
    # A failed mutation does not partially apply or change the derived id.
    with pytest.raises(ValidationError):
        rec.version = 4  # type: ignore[misc]
    assert rec.version == 3
    assert rec.record_id == expected


def test_ctor_and_create_paths_yield_identical_ids() -> None:
    """The auto-fill ctor path and the explicit create() path agree (SE-9)."""
    sid = _sid("eq")
    via_ctor = EntityRecord(stable_id=sid, kind="persona", version=2)
    via_create = EntityRecord.create(stable_id=sid, version=2, kind="persona")
    assert via_ctor.record_id == via_create.record_id
    assert via_ctor.record_id == record_id_for(stable_id=sid, version=2, kind="persona")


# -------------------------------------------- SE-10: content-address collisions
def test_register_collision_on_payload_divergence_raises(reg: EntityRegistry) -> None:
    """Same triple, different payload raises and the original is preserved (SE-10)."""
    sid = _sid("payload")
    original = EntityRecord.create(
        stable_id=sid, version=1, kind="persona", payload={"v": "original"}
    )
    reg.register(original)
    tampered = EntityRecord.create(
        stable_id=sid, version=1, kind="persona", payload={"v": "TAMPERED"}
    )
    with pytest.raises(OpenSimsError):
        reg.register(tampered)
    stored = reg.get(original.record_id)
    assert stored is not None and stored.payload == {"v": "original"}


def test_register_collision_on_metadata_divergence_raises(
    reg: EntityRegistry,
) -> None:
    """Same triple + same payload but DIFFERENT metadata also raises (SE-10)."""
    sid = _sid("meta")
    reg.register(
        EntityRecord.create(
            stable_id=sid,
            version=1,
            kind="persona",
            payload={"v": 1},
            metadata={"team": "alpha"},
        )
    )
    with pytest.raises(OpenSimsError):
        reg.register(
            EntityRecord.create(
                stable_id=sid,
                version=1,
                kind="persona",
                payload={"v": 1},
                metadata={"team": "beta"},
            )
        )


def test_register_identical_content_is_idempotent_after_collision(
    reg: EntityRegistry,
) -> None:
    """A collision does not poison later idempotent re-registration (SE-10)."""
    sid = _sid("idem")
    original = EntityRecord.create(
        stable_id=sid, version=1, kind="persona", payload={"v": "x"}
    )
    reg.register(original)
    with pytest.raises(OpenSimsError):
        reg.register(
            EntityRecord.create(
                stable_id=sid, version=1, kind="persona", payload={"v": "y"}
            )
        )
    # Byte-identical content still returns the stored instance, not a duplicate.
    again = reg.register(original)
    assert again is original
    assert len(reg.list_by_kind("persona")) == 1


def test_new_version_after_collision_still_increments(reg: EntityRegistry) -> None:
    """new_version() is the documented escape hatch for evolving content (SE-10)."""
    sid = _sid("evolve")
    reg.register(
        EntityRecord.create(stable_id=sid, version=1, kind="persona", payload={"v": 1})
    )
    v2 = reg.new_version(stable_id=sid, kind="persona", payload={"v": 2})
    assert v2.version == 2
    assert [r.version for r in reg.get_history(sid)] == [1, 2]


# ---------------------------------- SE-8: JSONB-containment metadata + queries
def test_metadata_contains_nested_dict() -> None:
    """A nested dict needle is contained when each key matches by containment."""
    haystack = {"score": {"raw": 0.9, "norm": 0.5}, "team": "alpha"}
    assert metadata_contains(haystack, {"score": {"raw": 0.9}}) is True
    assert metadata_contains(haystack, {"score": {"raw": 0.1}}) is False


def test_metadata_contains_list_membership() -> None:
    """A list needle matches when every element is contained by some element."""
    assert metadata_contains({"tags": ["q1", "q2", "q3"]}, {"tags": ["q1", "q3"]})
    assert not metadata_contains({"tags": ["q1"]}, {"tags": ["q1", "q9"]})


def test_metadata_contains_scalar_and_type_mismatches() -> None:
    """Scalar equality + structural type mismatches behave like PG ``@>``."""
    assert metadata_contains({"a": 1}, {"a": 1})
    assert not metadata_contains({"a": 1}, {"a": 2})
    assert not metadata_contains({"a": 1}, {"b": 1})  # missing key
    assert not metadata_contains({"a": 1}, {"a": [1]})  # list vs scalar
    assert not metadata_contains({"a": 1}, {"a": {"x": 1}})  # dict vs scalar


def test_metadata_contains_empty_needle_is_universal() -> None:
    """An empty filter contains everything (vacuous truth, like ``{} @> X``)."""
    assert metadata_contains({"anything": 1}, {})
    assert metadata_contains({}, {})


def test_query_no_predicates_returns_all(reg: EntityRegistry) -> None:
    """A query with no kind/stable_id/metadata returns every record (SE-8)."""
    reg.register(EntityRecord.create(stable_id=_sid("a"), version=1, kind="k1"))
    reg.register(EntityRecord.create(stable_id=_sid("b"), version=1, kind="k2"))
    assert len(reg.query(EntityQuery())) == 2


def test_query_metadata_only_uses_containment(reg: EntityRegistry) -> None:
    """A kind-less metadata-only query matches by containment across kinds (SE-8)."""
    sid_a = _sid("ma")
    reg.register(
        EntityRecord.create(
            stable_id=sid_a,
            version=1,
            kind="persona",
            metadata={"team": "alpha", "tags": ["q1", "q2"]},
        )
    )
    reg.register(
        EntityRecord.create(
            stable_id=_sid("mb"),
            version=1,
            kind="agent",
            metadata={"team": "alpha"},
        )
    )
    reg.register(
        EntityRecord.create(
            stable_id=_sid("mc"), version=1, kind="agent", metadata={"team": "beta"}
        )
    )
    alpha = reg.query(EntityQuery(metadata_filters={"team": "alpha"}))
    assert len(alpha) == 2  # crosses kinds, no kind required
    nested_list = reg.query(EntityQuery(metadata_filters={"tags": ["q1"]}))
    assert len(nested_list) == 1 and nested_list[0].stable_id == sid_a


def test_query_stable_id_only(reg: EntityRegistry) -> None:
    """A stable_id-only query (no kind) returns just that artefact's versions (SE-8)."""
    sid = _sid("so")
    reg.new_version(stable_id=sid, kind="persona", payload={"v": 1})
    reg.new_version(stable_id=sid, kind="persona", payload={"v": 2})
    reg.register(EntityRecord.create(stable_id=_sid("other"), version=1, kind="agent"))
    got = reg.query(EntityQuery(stable_id=sid))
    assert {r.version for r in got} == {1, 2}
    assert all(r.stable_id == sid for r in got)


def test_query_combines_kind_stable_id_and_metadata(reg: EntityRegistry) -> None:
    """All three predicates AND together (SE-8)."""
    sid = _sid("combo")
    reg.register(
        EntityRecord.create(
            stable_id=sid, version=1, kind="persona", metadata={"team": "alpha"}
        )
    )
    reg.register(
        EntityRecord.create(
            stable_id=_sid("combo2"),
            version=1,
            kind="persona",
            metadata={"team": "alpha"},
        )
    )
    got = reg.query(
        EntityQuery(kind="persona", stable_id=sid, metadata_filters={"team": "alpha"})
    )
    assert len(got) == 1 and got[0].stable_id == sid
    # A contradictory metadata filter yields nothing.
    assert (
        reg.query(EntityQuery(stable_id=sid, metadata_filters={"team": "beta"})) == []
    )


# ---------------------------------------------- SE-12: per-instance defaults
def test_entity_record_mutable_defaults_are_per_instance() -> None:
    """payload/metadata defaults are independent objects, not shared (SE-12)."""
    a = EntityRecord(stable_id="a", kind="k")
    b = EntityRecord(stable_id="b", kind="k")
    assert a.payload is not b.payload
    assert a.metadata is not b.metadata
    assert a.payload == {} and a.metadata == {}


def test_entity_query_and_link_mutable_defaults_are_per_instance() -> None:
    """EntityQuery.metadata_filters and EntityLink.metadata use default_factory (SE-12)."""
    q1 = EntityQuery()
    q2 = EntityQuery()
    assert q1.metadata_filters is not q2.metadata_filters
    link1 = EntityLink(from_record_id="a", to_record_id="b", relation="r")
    link2 = EntityLink(from_record_id="c", to_record_id="d", relation="r")
    assert link1.metadata is not link2.metadata
    # link_id is also a per-instance factory value.
    assert link1.link_id != link2.link_id
