"""Tests for the entities kernel: deterministic ids, versioning, lineage, queries."""

from __future__ import annotations

import uuid

import pytest

from himmy.core.errors import HimmyError
from himmy.entities.records import (
    EntityQuery,
    EntityRecord,
    record_id_for,
    stable_id_for,
)
from himmy.entities.registry import EntityRegistry


def test_stable_id_for_is_deterministic_and_namespaced() -> None:
    """Same (namespace, value) -> same id; different namespaces -> different ids."""
    a = stable_id_for("acme", namespace="persona")
    b = stable_id_for("acme", namespace="persona")
    c = stable_id_for("acme", namespace="agent")
    assert a == b
    assert a != c


def test_stable_id_for_passes_through_existing_uuid() -> None:
    """A value that is already a UUID round-trips unchanged."""
    existing = str(uuid.uuid4())
    assert stable_id_for(existing, namespace="persona") == existing


def test_stable_id_for_uses_fallback_key_when_value_falsy() -> None:
    """An empty value falls back to ``fallback_key``."""
    derived = stable_id_for("", namespace="persona", fallback_key="Analyst")
    expected = stable_id_for("Analyst", namespace="persona")
    assert derived == expected


def test_record_id_for_is_deterministic() -> None:
    """record_id is a pure function of (kind, stable_id, version)."""
    sid = stable_id_for("acme", namespace="persona")
    rid1 = record_id_for(stable_id=sid, version=1, kind="persona")
    rid2 = record_id_for(stable_id=sid, version=1, kind="persona")
    rid3 = record_id_for(stable_id=sid, version=2, kind="persona")
    assert rid1 == rid2
    assert rid1 != rid3


def test_entity_record_autofills_record_id() -> None:
    """Constructing an EntityRecord without a record_id computes it deterministically."""
    sid = stable_id_for("acme", namespace="persona")
    rec = EntityRecord(stable_id=sid, kind="persona", version=1)
    assert rec.record_id == record_id_for(stable_id=sid, version=1, kind="persona")


def test_register_is_idempotent_on_record_id() -> None:
    """Re-registering an identical record returns the original (no duplication)."""
    reg = EntityRegistry()
    rec = EntityRecord.create(
        stable_id=stable_id_for("acme", namespace="persona"),
        version=1,
        kind="persona",
        payload={"name": "ACME"},
    )
    first = reg.register(rec)
    second = reg.register(rec)
    assert first is second
    assert len(reg.list_by_kind("persona")) == 1


def test_new_version_increments_and_tracks_history() -> None:
    """new_version bumps the version and get_history returns ascending versions."""
    reg = EntityRegistry()
    sid = stable_id_for("acme", namespace="persona")
    v1 = reg.new_version(stable_id=sid, kind="persona", payload={"v": 1})
    v2 = reg.new_version(stable_id=sid, kind="persona", payload={"v": 2})
    assert v1.version == 1
    assert v2.version == 2
    latest = reg.get_latest(sid)
    assert latest is not None and latest.version == 2
    history = reg.get_history(sid)
    assert [r.version for r in history] == [1, 2]


def test_new_version_optimistic_concurrency_conflict() -> None:
    """A stale expected_version raises HimmyError."""
    reg = EntityRegistry()
    sid = stable_id_for("acme", namespace="persona")
    reg.new_version(stable_id=sid, kind="persona", payload={"v": 1})
    with pytest.raises(HimmyError):
        reg.new_version(
            stable_id=sid, kind="persona", payload={"v": 2}, expected_version=0
        )
    # The correct expected_version succeeds.
    ok = reg.new_version(
        stable_id=sid, kind="persona", payload={"v": 2}, expected_version=1
    )
    assert ok.version == 2


def test_link_and_links_from() -> None:
    """Links connect records and are retrievable by origin (lineage)."""
    reg = EntityRegistry()
    a = reg.new_version(
        stable_id=stable_id_for("a", namespace="chat_thread"),
        kind="chat_thread",
        payload={},
    )
    b = reg.new_version(
        stable_id=stable_id_for("b", namespace="persona"),
        kind="persona",
        payload={},
    )
    link = reg.link(
        from_record_id=a.record_id,
        to_record_id=b.record_id,
        relation="uses_persona",
    )
    assert link.relation == "uses_persona"
    out = reg.links_from(a.record_id)
    assert len(out) == 1
    assert out[0].to_record_id == b.record_id


def test_query_filters_by_kind_and_metadata() -> None:
    """EntityQuery filters by kind and metadata pairs."""
    reg = EntityRegistry()
    reg.register(
        EntityRecord.create(
            stable_id=stable_id_for("x", namespace="persona"),
            version=1,
            kind="persona",
            payload={},
            metadata={"team": "alpha"},
        )
    )
    reg.register(
        EntityRecord.create(
            stable_id=stable_id_for("y", namespace="persona"),
            version=1,
            kind="persona",
            payload={},
            metadata={"team": "beta"},
        )
    )
    reg.register(
        EntityRecord.create(
            stable_id=stable_id_for("z", namespace="agent"),
            version=1,
            kind="agent",
            payload={},
        )
    )
    personas = reg.query(EntityQuery(kind="persona"))
    assert len(personas) == 2
    alpha = reg.query(EntityQuery(kind="persona", metadata_filters={"team": "alpha"}))
    assert len(alpha) == 1


def test_entity_record_is_frozen_and_id_stays_consistent() -> None:
    """EntityRecord is immutable; record_id always matches its triple (SE-9)."""
    from pydantic import ValidationError

    sid = stable_id_for("acme", namespace="persona")
    rec = EntityRecord.create(stable_id=sid, version=1, kind="persona", payload={})
    with pytest.raises(ValidationError):
        rec.version = 99  # type: ignore[misc]
    assert rec.record_id == record_id_for(stable_id=sid, version=1, kind="persona")


def test_register_raises_on_content_address_collision() -> None:
    """Same (kind, stable_id, version) with different payload raises (SE-10)."""
    reg = EntityRegistry()
    sid = stable_id_for("acme", namespace="persona")
    original = EntityRecord.create(
        stable_id=sid, version=1, kind="persona", payload={"v": "original"}
    )
    reg.register(original)

    tampered = EntityRecord.create(
        stable_id=sid, version=1, kind="persona", payload={"v": "TAMPERED"}
    )
    with pytest.raises(HimmyError):
        reg.register(tampered)
    # The stored content is unchanged; the tamper did not silently win.
    assert reg.get(original.record_id).payload == {"v": "original"}
    # Re-registering identical content is still idempotent.
    assert reg.register(original) is original


def test_query_metadata_containment_and_partial_queries() -> None:
    """Containment metadata filters + stable_id-only / metadata-only queries (SE-8)."""
    reg = EntityRegistry()
    sid_a = stable_id_for("a", namespace="persona")
    reg.register(
        EntityRecord.create(
            stable_id=sid_a,
            version=1,
            kind="persona",
            payload={},
            metadata={"team": "alpha", "tags": ["q1", "q2"]},
        )
    )
    reg.register(
        EntityRecord.create(
            stable_id=stable_id_for("b", namespace="agent"),
            version=1,
            kind="agent",
            payload={},
            metadata={"team": "alpha"},
        )
    )

    # Metadata-only query (no kind) via containment, including a nested list match.
    by_meta = reg.query(EntityQuery(metadata_filters={"team": "alpha"}))
    assert len(by_meta) == 2
    list_contained = reg.query(EntityQuery(metadata_filters={"tags": ["q1"]}))
    assert len(list_contained) == 1 and list_contained[0].stable_id == sid_a

    # stable_id-only query (no kind).
    by_sid = reg.query(EntityQuery(stable_id=sid_a))
    assert len(by_sid) == 1 and by_sid[0].stable_id == sid_a


def test_domain_to_record_projections() -> None:
    """Domain types project to records with the documented kinds + stable namespaces."""
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
    from himmy.agents.personas.persona import Persona

    persona = Persona(name="Analyst")
    task = Task(title="t", prompt="p")
    thread = ChatThread(version=3)
    msg = Message(role=MessageRole.USER, content="hi")

    assert persona.to_record().kind == "persona"
    assert task.to_record().kind == "prompt"
    assert msg.to_record().kind == "message"
    # ChatThread.to_record uses self.version when version arg is None.
    assert thread.to_record().version == 3
    assert thread.to_record(version=5).version == 5
