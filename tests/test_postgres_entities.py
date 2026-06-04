"""Docker-gated contract tests for PostgresEntityRegistry (SE-3/4/8).

SKIPPED unless ``HIMMY_TEST_POSTGRES_DSN`` is set and ``asyncpg`` is importable.
Covers the Postgres-specific guarantees the in-memory registry already satisfies:
JSONB round-trip of nested payload/metadata, DB-enforced optimistic concurrency in
``new_version`` (a lost version raises rather than silently succeeding), the
content-address collision guard in ``register``, and indexed JSONB-containment
``query`` (including stable_id-only / metadata-only / lineage queries).
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from himmy.core.errors import HimmyError
from himmy.entities.postgres import PostgresEntityRegistry
from himmy.entities.records import EntityQuery, EntityRecord, stable_id_for
from tests.conftest import run_async

_DSN = os.environ.get("HIMMY_TEST_POSTGRES_DSN")

try:  # pragma: no cover - import probe
    import asyncpg  # type: ignore  # noqa: F401

    _HAVE_ASYNCPG = True
except ImportError:  # pragma: no cover
    _HAVE_ASYNCPG = False

pytestmark = pytest.mark.skipif(
    not _DSN or not _HAVE_ASYNCPG,
    reason="requires HIMMY_TEST_POSTGRES_DSN + the [postgres] extra",
)


async def _fresh_registry() -> PostgresEntityRegistry:
    reg = await PostgresEntityRegistry.connect(_DSN)
    await reg.create_schema()
    return reg


def _sid(prefix: str) -> str:
    return stable_id_for(f"{prefix}-{uuid.uuid4()}", namespace="persona")


def test_register_and_get_jsonb_round_trip() -> None:
    """Nested payload/metadata round-trip as native dicts via the codec (SE-4)."""

    async def scenario() -> None:
        reg = await _fresh_registry()
        try:
            sid = _sid("rt")
            rec = EntityRecord.create(
                stable_id=sid,
                version=1,
                kind="persona",
                payload={"nested": {"a": [1, {"b": "c"}]}},
                metadata={"team": "alpha"},
            )
            await reg.register(rec)
            got = await reg.get(rec.record_id)
            assert got is not None
            assert got.payload == {"nested": {"a": [1, {"b": "c"}]}}
            assert got.metadata == {"team": "alpha"}
        finally:
            await reg.close()

    run_async(scenario())


def test_register_idempotent_and_collision_guard() -> None:
    """Identical content is idempotent; different payload raises (SE-3/10)."""

    async def scenario() -> None:
        reg = await _fresh_registry()
        try:
            sid = _sid("col")
            original = EntityRecord.create(
                stable_id=sid, version=1, kind="persona", payload={"v": "original"}
            )
            await reg.register(original)
            # Re-register identical content -> no error, returns equivalent record.
            again = await reg.register(original)
            assert again.payload == {"v": "original"}

            tampered = EntityRecord.create(
                stable_id=sid, version=1, kind="persona", payload={"v": "TAMPERED"}
            )
            with pytest.raises(HimmyError):
                await reg.register(tampered)
            stored = await reg.get(original.record_id)
            assert stored is not None and stored.payload == {"v": "original"}
        finally:
            await reg.close()

    run_async(scenario())


def test_new_version_concurrency_does_not_lose_versions() -> None:
    """Concurrent new_version calls serialise; none silently lost (SE-3)."""

    async def scenario() -> None:
        reg = await _fresh_registry()
        try:
            sid = _sid("conc")
            # Fire many concurrent new_version calls on the same stable_id.
            results = await asyncio.gather(
                *(
                    reg.new_version(stable_id=sid, kind="persona", payload={"i": i})
                    for i in range(8)
                ),
                return_exceptions=True,
            )
            ok = [r for r in results if isinstance(r, EntityRecord)]
            # The advisory lock + FOR UPDATE serialise writers, so every call
            # gets a distinct, contiguous version (no DO NOTHING swallowing).
            versions = sorted(r.version for r in ok)
            assert versions == list(range(1, len(ok) + 1))
            history = await reg.get_history(sid)
            assert [h.version for h in history] == versions
        finally:
            await reg.close()

    run_async(scenario())


def test_new_version_expected_version_conflict() -> None:
    """A stale expected_version raises HimmyError (SE-3 parity with in-memory)."""

    async def scenario() -> None:
        reg = await _fresh_registry()
        try:
            sid = _sid("exp")
            await reg.new_version(stable_id=sid, kind="persona", payload={"v": 1})
            with pytest.raises(HimmyError):
                await reg.new_version(
                    stable_id=sid, kind="persona", payload={"v": 2}, expected_version=0
                )
            ok = await reg.new_version(
                stable_id=sid, kind="persona", payload={"v": 2}, expected_version=1
            )
            assert ok.version == 2
        finally:
            await reg.close()

    run_async(scenario())


def test_query_jsonb_containment_and_partial_queries() -> None:
    """query() pushes containment into SQL + supports partial queries (SE-8)."""

    async def scenario() -> None:
        reg = await _fresh_registry()
        try:
            sid_a = _sid("qa")
            await reg.register(
                EntityRecord.create(
                    stable_id=sid_a,
                    version=1,
                    kind="persona",
                    payload={},
                    metadata={"team": "alpha", "tags": ["q1", "q2"]},
                )
            )
            await reg.register(
                EntityRecord.create(
                    stable_id=_sid("qb"),
                    version=1,
                    kind="agent",
                    payload={},
                    metadata={"team": "beta"},
                )
            )

            # Metadata-only query (no kind), containment incl. nested list.
            alpha = await reg.query(EntityQuery(metadata_filters={"team": "alpha"}))
            assert len(alpha) == 1 and alpha[0].stable_id == sid_a
            tagged = await reg.query(EntityQuery(metadata_filters={"tags": ["q1"]}))
            assert any(r.stable_id == sid_a for r in tagged)

            # stable_id-only query (no kind).
            by_sid = await reg.query(EntityQuery(stable_id=sid_a))
            assert len(by_sid) == 1 and by_sid[0].stable_id == sid_a
        finally:
            await reg.close()

    run_async(scenario())


def test_link_and_links_from_round_trip() -> None:
    """Links persist and resolve by origin (lineage); no Postgres coverage gap (SE-7)."""

    async def scenario() -> None:
        reg = await _fresh_registry()
        try:
            a = await reg.new_version(
                stable_id=_sid("la"), kind="chat_thread", payload={}
            )
            b = await reg.new_version(stable_id=_sid("lb"), kind="persona", payload={})
            link = await reg.link(
                from_record_id=a.record_id,
                to_record_id=b.record_id,
                relation="uses_persona",
                metadata={"weight": 1},
            )
            assert link.relation == "uses_persona"
            out = await reg.links_from(a.record_id)
            assert len(out) == 1
            assert out[0].to_record_id == b.record_id
            assert out[0].metadata == {"weight": 1}
        finally:
            await reg.close()

    run_async(scenario())
