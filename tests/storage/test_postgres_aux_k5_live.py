"""Live-PG-gated contract tests for the K5 Postgres aux mirrors.

SKIPPED unless ``HIMMY_TEST_POSTGRES_DSN`` is set AND ``asyncpg`` is importable (the same
gate as the other live aux tests). They run the K5 mirrors — the Studio CRUD sidecars
(calendar / cookbook / notes / tasks / notify) plus the long-term memory store — against a
REAL database through their SYNCHRONOUS facades, asserting the authoritative SQL semantics
the offline lane cannot: real round-trips, the indexed filters (memory ``subject_id`` /
``active_only`` / ``tier``; calendar month), ordering, tenant isolation, and the notify
hydrate/persist/ring-prune cycle.

Each test binds a UNIQUE tenant (a uuid) so the suite is re-runnable without truncation and
two tenants are provably isolated.
"""

from __future__ import annotations

import os
import uuid

import pytest

from himmy.services.storage.aux_store_factory import (
    aux_loop,
    reset_aux_store_factory,
)

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


@pytest.fixture(autouse=True)
def _schema() -> object:  # pragma: no cover - only runs with a live DB
    from himmy.services.storage.postgres import PostgresStorageService

    reset_aux_store_factory()

    async def _migrate() -> None:
        storage = await PostgresStorageService.connect(_DSN)
        try:
            await storage.migrate()
        finally:
            await storage.close()

    aux_loop().run(_migrate())
    yield
    reset_aux_store_factory()


def _tenant() -> str:  # pragma: no cover - only runs with a live DB
    return f"t-{uuid.uuid4()}"


def test_calendar_roundtrip_month_filter_and_isolation() -> None:  # pragma: no cover
    from himmy.api.studio_calendar import CalendarEvent
    from himmy.services.storage.postgres_aux import PostgresCalendarStore

    ta, tb = _tenant(), _tenant()
    a = PostgresCalendarStore(tenant=ta, dsn=_DSN)
    b = PostgresCalendarStore(tenant=tb, dsn=_DSN)
    a.add(CalendarEvent(date="2026-06-13", time="09:00", title="standup"))
    a.add(CalendarEvent(date="2026-06-13", time=None, title="all-day"))
    a.add(CalendarEvent(date="2026-07-01", title="july"))

    june = a.list(month="2026-06")
    assert [e.title for e in june] == ["standup", "all-day"]  # all-day sorts last
    assert len(a.list()) == 3
    assert b.list() == []  # tenant isolation

    target = june[0]
    assert a.delete(target.id) is True
    assert len(a.list(month="2026-06")) == 1


def test_cookbook_roundtrip() -> None:  # pragma: no cover
    from himmy.api.studio_cookbook import Recipe
    from himmy.services.storage.postgres_aux import PostgresCookbookStore

    store = PostgresCookbookStore(tenant=_tenant(), dsn=_DSN)
    r = Recipe(name="summarize", agent_path="a.yaml", prompt="do it", notes="n")
    store.upsert(r)
    listed = store.list()
    assert [x.name for x in listed] == ["summarize"]
    r.notes = "updated"
    store.upsert(r)
    assert store.list()[0].notes == "updated"
    assert store.delete(r.id) is True
    assert store.list() == []


def test_notes_roundtrip_and_find_by_title() -> None:  # pragma: no cover
    from himmy.api.studio_notes import Note
    from himmy.services.storage.postgres_aux import PostgresNotesStore

    store = PostgresNotesStore(tenant=_tenant(), dsn=_DSN)
    n = Note(title="todo", body="b1")
    store.upsert(n)
    assert store.get(n.id) is not None
    assert store.find_by_title("todo").id == n.id
    assert store.find_by_title("missing") is None
    n.body = "b2"
    store.upsert(n)
    assert store.get(n.id).body == "b2"
    assert store.delete(n.id) is True
    assert store.get(n.id) is None


def test_tasks_roundtrip_and_complete_by_title() -> None:  # pragma: no cover
    from himmy.services.storage.postgres_aux import PostgresTasksStore

    store = PostgresTasksStore(tenant=_tenant(), dsn=_DSN)
    t = store.add("ship it", due="2026-06-20")
    assert [x.title for x in store.list()] == ["ship it"]
    assert store.complete_by_title("ship it") is True
    assert store.list()[0].done is True
    # Already-done -> the conditional UPDATE matches nothing.
    assert store.complete_by_title("ship it") is False
    assert store.set_done(t.id, False) is True
    assert store.delete(t.id) is True
    assert store.list() == []


def test_memory_roundtrip_filters_and_links() -> None:  # pragma: no cover
    from himmy.services.memory.store import (
        CONTRADICTS,
        MemoryLink,
        MemoryRecord,
    )
    from himmy.services.storage.postgres_aux import PostgresMemoryStore

    store = PostgresMemoryStore(tenant=_tenant(), dsn=_DSN)
    a = MemoryRecord(subject_id="alice", text="likes tea", tier="core")
    b = MemoryRecord(
        subject_id="alice", text="lived in Pune", valid_to="2026-01-01T00:00:00+00:00"
    )
    c = MemoryRecord(subject_id="bob", text="likes coffee")
    store.save(a)
    store.save(b)
    store.save(c)

    assert {r.text for r in store.list("alice")} == {"likes tea", "lived in Pune"}
    # active_only drops the invalidated fact (valid_to set).
    assert {r.text for r in store.list("alice", active_only=True)} == {"likes tea"}
    # tier filter.
    assert {r.text for r in store.list("alice", tier="core")} == {"likes tea"}
    assert store.get(a.memory_id).text == "likes tea"

    link = MemoryLink(
        from_memory_id=a.memory_id, to_memory_id=b.memory_id, relation=CONTRADICTS
    )
    store.save_link(link)
    assert [x.link_id for x in store.links_from(a.memory_id)] == [link.link_id]
    assert [x.relation for x in store.links_to(b.memory_id)] == [CONTRADICTS]

    assert store.delete(a.memory_id) is True
    assert store.get(a.memory_id) is None


def test_notify_hydrate_persist_and_ring_prune() -> None:  # pragma: no cover
    from himmy.services.storage.postgres_aux import PostgresNotifyStore

    tenant = _tenant()
    store = PostgresNotifyStore(tenant=tenant, dsn=_DSN)
    assert store.max_id() == 0
    items, forward = store.hydrate(500)
    assert items == [] and forward is False

    store.set_setting("forward_telegram", "1")
    item = {
        "id": 1,
        "kind": "mission",
        "title": "done",
        "body": "ok",
        "link": "/m",
        "created_at": "2026-06-13T00:00:00+00:00",
        "read": False,
    }
    store.persist_item(item, 500)
    rehydrated, forward2 = store.hydrate(500)
    assert [r["id"] for r in rehydrated] == [1]
    assert forward2 is True
    assert store.max_id() == 1

    store.mark_read([1])
    assert store.hydrate(500)[0][0]["read"] is True
