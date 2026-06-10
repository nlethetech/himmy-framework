"""Registry concurrency + graph traversal: cross-backend audit-spine coverage.

The entity registry is the audit spine of the framework, so both backends (the
in-memory ``EntityRegistry`` and the durable ``SqliteEntityRegistry``) must keep
the version chain intact under concurrent writers, keep registration idempotent
on content, derive stable UUID5 ids, and answer link traversals correctly.

The concurrency tests here are regression tests for the read-latest -> insert
race in ``new_version`` (and the check-then-store race in ``register``): without
the registry lock, two threads both read version N and one update is silently
lost or surfaces as a spurious content-address violation. The tests shrink the
GIL switch interval so thread interleavings actually happen.
"""

from __future__ import annotations

import sys
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from himmy.core.errors import HimmyError
from himmy.entities import (
    EntityRecord,
    EntityRegistry,
    LineageDirection,
    SqliteEntityRegistry,
    export_audit_chain,
    record_id_for,
    stable_id_for,
    verify_audit_chain,
)

Registry = EntityRegistry | SqliteEntityRegistry


@pytest.fixture(autouse=True)
def _tight_gil_switch() -> Iterator[None]:
    """Shrink the GIL switch interval so racy interleavings are actually hit."""
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


@pytest.fixture(params=["memory", "sqlite"])
def registry(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Registry]:
    """Yield each registry backend so every test runs against both."""
    if request.param == "memory":
        yield EntityRegistry()
        return
    with SqliteEntityRegistry(str(tmp_path / "registry.db")) as reg:
        yield reg


def _hammer_new_version(
    reg: Registry,
    *,
    threads: int,
    calls_per_thread: int,
    stable_id: str = "doc",
) -> tuple[list[EntityRecord], list[BaseException]]:
    """Fire new_version from many threads at once; collect records + errors."""
    barrier = threading.Barrier(threads)
    records: list[EntityRecord] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        barrier.wait()
        for i in range(calls_per_thread):
            try:
                rec = reg.new_version(
                    stable_id=stable_id, kind="doc", payload={"writer": idx, "i": i}
                )
            except BaseException as exc:  # noqa: BLE001 - the test asserts on these
                with lock:
                    errors.append(exc)
            else:
                with lock:
                    records.append(rec)

    pool = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    return records, errors


# ------------------------------------------------------------------ concurrency
def test_concurrent_new_version_yields_contiguous_chain(registry: Registry) -> None:
    """N concurrent new_version calls -> exactly N versions, no gaps, no errors."""
    threads, calls = 8, 10
    records, errors = _hammer_new_version(
        registry, threads=threads, calls_per_thread=calls
    )
    assert errors == []
    total = threads * calls
    assert len(records) == total
    assert sorted(r.version for r in records) == list(range(1, total + 1))
    history = registry.get_history("doc")
    assert [r.version for r in history] == list(range(1, total + 1))
    latest = registry.get_latest("doc")
    assert latest is not None
    assert latest.version == total


def test_version_chain_integrity_after_concurrent_writers(
    registry: Registry,
) -> None:
    """After a concurrent hammer the chain is still deterministic + verifiable.

    Every stored record_id must equal the UUID5 derived from its
    (kind, stable_id, version) triple, and the ordered history must round-trip
    through the signed audit chain (export -> verify) without a break.
    """
    _hammer_new_version(registry, threads=6, calls_per_thread=5)
    history = registry.get_history("doc")
    assert len(history) == 30
    for rec in history:
        assert rec.record_id == record_id_for(
            stable_id=rec.stable_id, version=rec.version, kind=rec.kind
        )
    bundle = export_audit_chain(history, secret="s3cret")
    result = verify_audit_chain(bundle, registry.get_history("doc"), secret="s3cret")
    assert result.ok
    assert result.chain_intact
    assert result.first_break_index is None


def test_concurrent_register_identical_record_is_idempotent(
    registry: Registry,
) -> None:
    """Many threads registering the SAME record -> one stored version, no dupes."""
    record = EntityRecord.create(
        stable_id="shared", version=1, kind="doc", payload={"x": 1}
    )
    n = 12
    barrier = threading.Barrier(n)
    returned: list[EntityRecord] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            rec = registry.register(record)
        except BaseException as exc:  # noqa: BLE001 - the test asserts on these
            with lock:
                errors.append(exc)
        else:
            with lock:
                returned.append(rec)

    pool = [threading.Thread(target=worker) for _ in range(n)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    assert errors == []
    assert len(returned) == n
    assert {r.record_id for r in returned} == {record.record_id}
    history = registry.get_history("shared")
    assert len(history) == 1
    assert history[0].payload == {"x": 1}


def test_concurrent_conflicting_register_single_winner(registry: Registry) -> None:
    """Racing DIFFERENT content at one identity -> exactly one winner, rest raise."""
    n = 8
    barrier = threading.Barrier(n)
    winners: list[EntityRecord] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        record = EntityRecord.create(
            stable_id="contested", version=1, kind="doc", payload={"writer": idx}
        )
        barrier.wait()
        try:
            rec = registry.register(record)
        except BaseException as exc:  # noqa: BLE001 - the test asserts on these
            with lock:
                errors.append(exc)
        else:
            with lock:
                winners.append(rec)

    pool = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    assert len(winners) == 1
    assert len(errors) == n - 1
    assert all(isinstance(exc, HimmyError) for exc in errors)
    history = registry.get_history("contested")
    assert len(history) == 1
    assert history[0].payload == winners[0].payload


def test_concurrent_link_creation_and_fanout(registry: Registry) -> None:
    """Concurrent link() calls all land; links_from/links_to stay consistent."""
    hub = registry.register(
        EntityRecord.create(stable_id="hub", version=1, kind="node", payload={})
    )
    n = 10
    spokes = [
        registry.register(
            EntityRecord.create(stable_id=f"spoke-{i}", version=1, kind="node")
        )
        for i in range(n)
    ]
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []

    def worker(i: int) -> None:
        barrier.wait()
        try:
            registry.link(
                from_record_id=hub.record_id,
                to_record_id=spokes[i].record_id,
                relation="feeds",
            )
        except BaseException as exc:  # noqa: BLE001 - the test asserts on these
            errors.append(exc)

    pool = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    assert errors == []
    out = registry.links_from(hub.record_id)
    assert len(out) == n
    assert len({link.link_id for link in out}) == n
    assert {link.to_record_id for link in out} == {s.record_id for s in spokes}
    for spoke in spokes:
        assert [link.from_record_id for link in registry.links_to(spoke.record_id)] == [
            hub.record_id
        ]
    graph = registry.trace(hub.record_id, max_depth=1)
    assert graph.node_count == n + 1
    assert graph.edge_count == n
    assert not graph.truncated


# ---------------------------------------------------------- content addressing
def test_register_identical_content_is_idempotent(registry: Registry) -> None:
    """Re-registering byte-identical content returns the stored record."""
    first = registry.register(
        EntityRecord.create(stable_id="idem", version=1, kind="doc", payload={"a": 1})
    )
    again = registry.register(
        EntityRecord.create(stable_id="idem", version=1, kind="doc", payload={"a": 1})
    )
    assert again.record_id == first.record_id
    assert again.payload == {"a": 1}
    assert len(registry.get_history("idem")) == 1


def test_register_raises_on_content_address_violation(registry: Registry) -> None:
    """Same (kind, stable_id, version) with different content must raise."""
    registry.register(
        EntityRecord.create(stable_id="ca", version=1, kind="doc", payload={"a": 1})
    )
    with pytest.raises(HimmyError, match="[Cc]ontent-address violation"):
        registry.register(
            EntityRecord.create(stable_id="ca", version=1, kind="doc", payload={"a": 2})
        )
    history = registry.get_history("ca")
    assert len(history) == 1
    assert history[0].payload == {"a": 1}


def test_new_version_optimistic_concurrency_conflict(registry: Registry) -> None:
    """expected_version mismatches the live latest -> HimmyError, no write."""
    registry.new_version(stable_id="occ", kind="doc", payload={"v": 1})
    with pytest.raises(HimmyError, match="Optimistic concurrency conflict"):
        registry.new_version(
            stable_id="occ", kind="doc", payload={"v": 2}, expected_version=5
        )
    assert len(registry.get_history("occ")) == 1


# ------------------------------------------------------------- id determinism
def test_record_id_uuid5_is_pinned_and_deterministic() -> None:
    """record_id derivation is UUID5-stable across calls, processes, machines."""
    rid = record_id_for(stable_id="doc-1", version=1, kind="doc")
    assert rid == record_id_for(stable_id="doc-1", version=1, kind="doc")
    # Pinned literals: changing the namespace or key format is a breaking change.
    assert rid == "313c83fe-17b7-5a6c-9bf1-c5462dc64146"
    assert (
        record_id_for(stable_id="doc-1", version=2, kind="doc")
        == "b5403178-9d79-5623-b182-3c647fded6a8"
    )
    record = EntityRecord.create(stable_id="doc-1", version=1, kind="doc")
    assert record.record_id == rid
    # The implicit-id path (model_post_init) must agree with the explicit one.
    assert EntityRecord(stable_id="doc-1", version=1, kind="doc").record_id == rid


def test_stable_id_for_round_trips_uuids_and_pins_uuid5() -> None:
    """Existing UUIDs pass through unchanged; semantic keys derive pinned UUID5s."""
    existing = "313c83fe-17b7-5a6c-9bf1-c5462dc64146"
    assert stable_id_for(existing, namespace="thread") == existing
    derived = stable_id_for("alpha", namespace="thread")
    assert derived == "ec811be2-ac2d-5a1c-bdb1-e952ae75b9fc"
    assert stable_id_for("", namespace="thread", fallback_key="alpha") == derived


# ------------------------------------------------------------- graph traversal
def _node(registry: Registry, name: str) -> EntityRecord:
    return registry.register(
        EntityRecord.create(stable_id=name, version=1, kind="node")
    )


def test_links_from_links_to_and_neighbors(registry: Registry) -> None:
    """Forward/reverse link queries and direction/relation-filtered neighbors."""
    a, b, c, d = (_node(registry, n) for n in ("a", "b", "c", "d"))
    registry.link(
        from_record_id=a.record_id, to_record_id=b.record_id, relation="built_from"
    )
    registry.link(
        from_record_id=a.record_id, to_record_id=c.record_id, relation="uses_persona"
    )
    registry.link(
        from_record_id=d.record_id, to_record_id=a.record_id, relation="observed_in_run"
    )
    self_loop = registry.link(
        from_record_id=a.record_id, to_record_id=a.record_id, relation="self"
    )

    assert {link.to_record_id for link in registry.links_from(a.record_id)} == {
        b.record_id,
        c.record_id,
        a.record_id,
    }
    assert {link.from_record_id for link in registry.links_to(a.record_id)} == {
        d.record_id,
        a.record_id,
    }
    out = registry.neighbors(a.record_id, direction=LineageDirection.OUT)
    assert len(out) == 3
    incoming = registry.neighbors(a.record_id, direction=LineageDirection.IN)
    assert len(incoming) == 2
    # BOTH de-duplicates the self-loop: 4 distinct links, the loop only once.
    both = registry.neighbors(a.record_id, direction=LineageDirection.BOTH)
    assert len(both) == 4
    assert sum(1 for link in both if link.link_id == self_loop.link_id) == 1
    filtered = registry.neighbors(
        a.record_id, direction=LineageDirection.BOTH, relation="built_from"
    )
    assert [link.to_record_id for link in filtered] == [b.record_id]


def test_trace_depth_budget_direction_and_truncation(registry: Registry) -> None:
    """BFS respects max_depth (flagging truncation), direction, and relations."""
    chain = [_node(registry, f"n{i}") for i in range(4)]
    for prev, nxt in zip(chain, chain[1:], strict=False):
        registry.link(
            from_record_id=prev.record_id,
            to_record_id=nxt.record_id,
            relation="derived",
        )

    partial = registry.trace(
        chain[0].record_id, max_depth=2, direction=LineageDirection.OUT
    )
    assert partial.record_ids() == {r.record_id for r in chain[:3]}
    assert partial.truncated

    full = registry.trace(
        chain[0].record_id, max_depth=3, direction=LineageDirection.OUT
    )
    assert full.record_ids() == {r.record_id for r in chain}
    assert not full.truncated
    assert full.relations() == {"derived"}

    # Reverse walk: from the tail, IN edges reach back to the head.
    reverse = registry.trace(
        chain[-1].record_id, max_depth=10, direction=LineageDirection.IN
    )
    assert reverse.record_ids() == {r.record_id for r in chain}

    # A relations filter that matches nothing leaves only the root.
    none = registry.trace(chain[0].record_id, relations=["unrelated"])
    assert none.record_ids() == {chain[0].record_id}
    assert none.edge_count == 0


# ------------------------------------------------------------------ durability
def test_sqlite_chain_survives_reopen_after_concurrent_writers(
    tmp_path: Path,
) -> None:
    """The version chain written under contention is intact after a reopen."""
    path = str(tmp_path / "durable.db")
    with SqliteEntityRegistry(path) as reg:
        _hammer_new_version(reg, threads=6, calls_per_thread=5)
    with SqliteEntityRegistry(path) as reopened:
        history = reopened.get_history("doc")
        assert [r.version for r in history] == list(range(1, 31))
        for rec in history:
            assert rec.record_id == record_id_for(
                stable_id=rec.stable_id, version=rec.version, kind=rec.kind
            )
        latest = reopened.get_latest("doc")
        assert latest is not None
        assert latest.version == 30
