"""Tests for the durable SQLite registry: CRUD, trace, durability, audit."""

from __future__ import annotations

import json

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.entities import (
    EntityQuery,
    EntityRecord,
    SqliteEntityRegistry,
    export_audit_bundle,
    verify_audit_bundle,
)
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _rec(reg: SqliteEntityRegistry, kind: str, sid: str, **payload) -> EntityRecord:
    return reg.register(
        EntityRecord.create(
            stable_id=sid, version=1, kind=kind, payload=payload or {"n": sid}
        )
    )


def test_crud_query_and_violation() -> None:
    """Basic register/get/query, idempotency, and content-address violation."""
    reg = SqliteEntityRegistry()
    a = _rec(reg, "chat_thread", "T")
    assert reg.get(a.record_id).stable_id == "T"
    assert reg.get_latest("T").record_id == a.record_id
    assert [r.record_id for r in reg.list_by_kind("chat_thread")] == [a.record_id]

    reg.register(
        EntityRecord.create(
            stable_id="M", version=1, kind="doc", metadata={"client": "acme"}
        )
    )
    hits = reg.query(EntityQuery(kind="doc", metadata_filters={"client": "acme"}))
    assert len(hits) == 1

    # Re-registering identical content is a no-op; differing content is a violation.
    reg.register(
        EntityRecord.create(
            stable_id="T", version=1, kind="chat_thread", payload={"n": "T"}
        )
    )
    with pytest.raises(HimmyError):
        reg.register(
            EntityRecord.create(
                stable_id="T", version=1, kind="chat_thread", payload={"n": "X"}
            )
        )
    reg.close()


def test_new_version_history_and_concurrency() -> None:
    """new_version increments + records history; expected_version is enforced."""
    reg = SqliteEntityRegistry()
    reg.new_version(stable_id="S", kind="k", payload={"v": 1})
    reg.new_version(stable_id="S", kind="k", payload={"v": 2})
    assert [r.version for r in reg.get_history("S")] == [1, 2]
    assert reg.get_latest("S").version == 2
    with pytest.raises(HimmyError):
        reg.new_version(stable_id="S", kind="k", payload={"v": 3}, expected_version=1)
    reg.close()


def test_links_and_trace_match_in_memory_semantics() -> None:
    """links_from/links_to/trace behave like the in-memory registry."""
    reg = SqliteEntityRegistry()
    t = _rec(reg, "chat_thread", "T")
    p = _rec(reg, "persona", "P")
    q = _rec(reg, "prompt", "Q")
    reg.link(
        from_record_id=t.record_id, to_record_id=p.record_id, relation="uses_persona"
    )
    reg.link(from_record_id=t.record_id, to_record_id=q.record_id, relation="in_thread")

    assert {link.relation for link in reg.links_from(t.record_id)} == {
        "uses_persona",
        "in_thread",
    }
    assert [link.relation for link in reg.links_to(p.record_id)] == ["uses_persona"]
    graph = reg.trace(t.record_id)
    assert {r.kind for r in graph.nodes.values()} == {
        "chat_thread",
        "persona",
        "prompt",
    }
    assert graph.relations() == {"uses_persona", "in_thread"}
    reg.close()


def test_durable_across_reopen(tmp_path) -> None:
    """Records + links survive closing and reopening the same database file."""
    path = str(tmp_path / "audit.db")
    reg = SqliteEntityRegistry(path)
    t = _rec(reg, "chat_thread", "T")
    p = _rec(reg, "persona", "P")
    reg.link(
        from_record_id=t.record_id, to_record_id=p.record_id, relation="uses_persona"
    )
    reg.close()

    reopened = SqliteEntityRegistry(path)
    assert reopened.get_latest("T") is not None
    graph = reopened.trace(t.record_id)
    assert "persona" in {r.kind for r in graph.nodes.values()}
    assert "uses_persona" in graph.relations()
    reopened.close()


def test_durable_store_plus_tamper_evident_audit(tmp_path) -> None:
    """A signed bundle catches a direct edit to the SQLite file."""
    path = str(tmp_path / "audit.db")
    secret = "nepal-civic-audit-key"
    reg = SqliteEntityRegistry(path)
    thread = _rec(reg, "chat_thread", "T", prompt="recommend: buy ACME")
    persona = _rec(reg, "persona", "P")
    reg.link(
        from_record_id=thread.record_id,
        to_record_id=persona.record_id,
        relation="uses_persona",
    )

    bundle = export_audit_bundle(reg.all_records(), reg.all_links(), secret=secret)
    clean = verify_audit_bundle(
        bundle, reg.all_records(), reg.all_links(), secret=secret
    )
    assert clean.ok is True

    # Someone edits the row directly in the database, bypassing the registry API.
    reg._conn.execute(
        "UPDATE entity_records SET payload = ? WHERE record_id = ?",
        (json.dumps({"prompt": "recommend: buy WORTHLESSCO"}), thread.record_id),
    )
    reg._conn.commit()

    tampered = verify_audit_bundle(
        bundle, reg.all_records(), reg.all_links(), secret=secret
    )
    assert tampered.ok is False
    assert thread.record_id in tampered.tampered_record_ids
    reg.close()


def test_real_run_lineage_survives_restart(tmp_path) -> None:
    """A real run's lineage, captured into SQLite, is intact after a 'restart'."""
    path = str(tmp_path / "run-audit.db")
    storage = StorageService()
    registry = SqliteEntityRegistry(path)
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        context_service=ContextService(
            storage_service=storage, entity_registry=registry
        ),
        entity_registry=registry,
    )
    run_async(
        runtime.run_task(
            Persona(name="analyst", description="x"),
            Task(title="t", prompt="Summarize ACME."),
        )
    )
    registry.close()

    # "Restart": reopen the same file and confirm the run's lineage is queryable.
    reopened = SqliteEntityRegistry(path)
    threads = reopened.list_by_kind("chat_thread")
    assert threads
    graph = reopened.trace(threads[0].record_id)
    kinds = {r.kind for r in graph.nodes.values()}
    assert "persona" in kinds
    assert "uses_persona" in graph.relations()
    reopened.close()
