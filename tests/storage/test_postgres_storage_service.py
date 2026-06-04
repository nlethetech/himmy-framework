"""Docker-gated contract tests for PostgresStorageService (SE-1/4/5/6/11).

These tests are SKIPPED unless both:

- ``OPENSIMS_TEST_POSTGRES_DSN`` is set (e.g.
  ``postgresql://opensims:opensims@localhost:5433/opensims`` — see
  ``docker/docker-compose.yml``), and
- the ``asyncpg`` extra is importable.

They run the SAME assertions the in-memory ``StorageService`` satisfies against a
real database, plus the Postgres-specific guarantees (JSONB round-trip, the
DB-enforced idempotency upsert, the migration runner, and the ``ai_call_log`` view
payload-key contract). Each test uses a unique workspace/id prefix so the suite is
re-runnable without truncation.
"""

from __future__ import annotations

import os
import uuid

import pytest

from opensims.agents.base_agent.thread import ChatThread, Message, MessageRole
from opensims.core.events import EventType, RunEvent
from opensims.services.storage import (
    MemoryObject,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)
from opensims.services.storage.postgres import PostgresStorageService
from tests.conftest import run_async

_DSN = os.environ.get("OPENSIMS_TEST_POSTGRES_DSN")

try:  # pragma: no cover - import probe
    import asyncpg  # type: ignore  # noqa: F401

    _HAVE_ASYNCPG = True
except ImportError:  # pragma: no cover
    _HAVE_ASYNCPG = False

pytestmark = pytest.mark.skipif(
    not _DSN or not _HAVE_ASYNCPG,
    reason="requires OPENSIMS_TEST_POSTGRES_DSN + the [postgres] extra",
)


async def _fresh_storage() -> PostgresStorageService:
    """Connect, run migrations (idempotent), and return a live storage service."""
    storage = await PostgresStorageService.connect(_DSN)
    await storage.migrate()
    return storage


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


def test_migrate_is_idempotent_and_tracked() -> None:
    """migrate() applies once; a second call applies nothing (SE-5)."""

    async def scenario() -> None:
        storage = await PostgresStorageService.connect(_DSN)
        try:
            await storage.migrate()
            second = await storage.migrate()
            assert second == []  # nothing left to apply
        finally:
            await storage.close()

    run_async(scenario())


def test_thread_round_trip() -> None:
    """A thread persists and reloads with its message list intact (SE-1)."""

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            thread = ChatThread(thread_id=_uid("t"))
            thread.append_message(Message(role=MessageRole.USER, content="hi"))
            await storage.save_thread(thread)
            loaded = await storage.load_thread(thread.thread_id)
            assert loaded is not None
            assert loaded.thread_id == thread.thread_id
            assert loaded.last_message is not None
            assert loaded.last_message.content == "hi"
        finally:
            await storage.close()

    run_async(scenario())


def test_run_jsonb_round_trip_and_idempotency() -> None:
    """Nested-dict metadata/output round-trips; idempotency upsert is race-free (SE-1/2/4)."""

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            ws = _uid("w")
            key = _uid("k")
            run = RunRecord(
                workspace_id=ws,
                subject_id="s1",
                idempotency_key=key,
                status=RunStatus.QUEUED,
                metadata={"nested": {"a": [1, 2, {"b": "c"}]}},
                output_structured={"recommendations": [{"title": "X"}]},
            )
            stored, created = await storage.save_run_if_absent_by_idempotency(run)
            assert created is True

            # JSONB round-trips as native Python objects (codec contract, SE-4).
            fetched = await storage.get_run(run.run_id)
            assert fetched is not None
            assert fetched.metadata == {"nested": {"a": [1, 2, {"b": "c"}]}}
            assert fetched.output_structured == {"recommendations": [{"title": "X"}]}

            # A second create with the same key returns the existing run (SE-2/4).
            dup = RunRecord(workspace_id=ws, subject_id="s1", idempotency_key=key)
            again, created2 = await storage.save_run_if_absent_by_idempotency(dup)
            assert created2 is False
            assert again.run_id == run.run_id

            # load_run_by_idempotency resolves the same run.
            by_key = await storage.load_run_by_idempotency(ws, key)
            assert by_key is not None and by_key.run_id == run.run_id
        finally:
            await storage.close()

    run_async(scenario())


def test_idempotency_race_collapses_to_one_run() -> None:
    """Concurrent same-key inserts produce exactly one created run (SE-2)."""
    import asyncio

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            ws = _uid("w")
            key = _uid("k")
            runs = [
                RunRecord(workspace_id=ws, subject_id="s", idempotency_key=key)
                for _ in range(10)
            ]
            results = await asyncio.gather(
                *(storage.save_run_if_absent_by_idempotency(r) for r in runs)
            )
            created = sum(1 for _, c in results if c)
            assert created == 1
            run_ids = {stored.run_id for stored, _ in results}
            assert len(run_ids) == 1
        finally:
            await storage.close()

    run_async(scenario())


def test_save_run_refreshes_updated_at() -> None:
    """save_run stamps a fresh updated_at at the DB boundary (SE-11)."""

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            run = RunRecord(workspace_id=_uid("w"), subject_id="s", updated_at="STALE")
            await storage.save_run(run)
            fetched = await storage.get_run(run.run_id)
            assert fetched is not None and fetched.updated_at != "STALE"
        finally:
            await storage.close()

    run_async(scenario())


def test_recommendation_crud_round_trip() -> None:
    """Recommendations persist, list (multi-predicate), and update (SE-1)."""

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            ws = _uid("w")
            item = RecommendationItem(
                run_id=_uid("r"),
                workspace_id=ws,
                subject_id="s1",
                kind="trade",
                title="Buy ACME",
                evidence_refs=["e1", "e2"],
                metadata={"score": {"raw": 0.9}},
            )
            await storage.save_recommendation(item)
            listed = await storage.list_recommendations(
                workspace_id=ws, kind="trade", status=RecommendationStatus.PROPOSED
            )
            assert len(listed) == 1
            assert listed[0].evidence_refs == ["e1", "e2"]
            assert listed[0].metadata == {"score": {"raw": 0.9}}

            updated = await storage.update_recommendation(
                item.recommendation_id,
                status=RecommendationStatus.ACCEPTED,
                notes="ok",
            )
            assert updated is not None
            assert updated.status == RecommendationStatus.ACCEPTED
            assert updated.notes == "ok"
        finally:
            await storage.close()

    run_async(scenario())


def test_events_round_trip_and_ai_call_log_contract() -> None:
    """A REQUESTED/SUCCEEDED pair populates ai_call_log non-null columns (SE-6).

    Locks the payload-key contract between the runtime/inference layer and the view:
    prompt_template, rendered_prompt, model_name (coalesced from model_path),
    input_tokens, cost_usd, and status must be non-null for the request_id.
    """

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            request_id = _uid("req")
            trace_id = _uid("tr")
            await storage.append_event(
                RunEvent(
                    event_type=EventType.INFERENCE_REQUESTED,
                    request_id=request_id,
                    trace_id=trace_id,
                    payload={
                        "prompt_template": "default_prompt",
                        "prompt_version": "v1",
                        "rendered_prompt": "Summarize ACME.",
                        "retrieval_ctx": ["k1", "k2"],
                        "snapshot_id": "snap-1",
                    },
                )
            )
            await storage.append_event(
                RunEvent(
                    event_type=EventType.INFERENCE_SUCCEEDED,
                    request_id=request_id,
                    trace_id=trace_id,
                    latency_ms=12.5,
                    cost=0.0021,
                    payload={
                        "input_tokens": 100,
                        "output_tokens": 42,
                        "model_path": "stub:default",
                        "provider_name": "stub",
                    },
                )
            )

            events = await storage.list_events(trace_id=trace_id)
            assert len(events) == 2

            pool = storage._require_pool()  # type: ignore[attr-defined]
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM ai_call_log WHERE request_id = $1", request_id
                )
            assert row is not None
            assert row["prompt_template"] == "default_prompt"
            assert row["prompt_version"] == "v1"
            assert row["rendered_prompt"] == "Summarize ACME."
            assert row["snapshot_id"] == "snap-1"
            assert row["model_name"] == "stub:default"  # coalesced from model_path
            assert row["input_tokens"] == 100
            assert row["output_tokens"] == 42
            assert row["cost_usd"] == pytest.approx(0.0021)
            assert row["status"] == "success"
        finally:
            await storage.close()

    run_async(scenario())


def test_memory_round_trip() -> None:
    """A memory object persists and reloads with its payload intact (SE-1)."""

    async def scenario() -> None:
        storage = await _fresh_storage()
        try:
            sid = _uid("s")
            mem = MemoryObject(subject_id=sid, payload={"note": {"x": 1}})
            await storage.save_memory(mem)
            got = await storage.get_memory(mem.memory_id)
            assert got is not None and got.payload == {"note": {"x": 1}}
            assert len(await storage.list_memory(subject_id=sid)) == 1
        finally:
            await storage.close()

    run_async(scenario())


def test_async_context_manager_closes_pool() -> None:
    """`async with` closes the pool on exit (SE-5)."""

    async def scenario() -> None:
        async with await PostgresStorageService.connect(_DSN) as storage:
            await storage.migrate()
            await storage.get_run("nope")
        # After exit the pool is closed; a fresh op must re-require a pool.
        assert storage._pool._closed  # type: ignore[attr-defined]

    run_async(scenario())
