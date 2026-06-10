"""SQLite registry concurrency: hardened connection + serialised new_version.

Regression tests for the read-latest -> insert race: two concurrent writers used
to both read version N and one update was silently lost (or surfaced as a raw
content-address violation). new_version() now runs under a write lock inside a
``BEGIN IMMEDIATE`` transaction, so N concurrent calls yield exactly N versions.
"""

from __future__ import annotations

import threading
from pathlib import Path

from himmy.entities import EntityRecord, SqliteEntityRegistry


def _hammer_new_version(
    registries: list[SqliteEntityRegistry],
    *,
    threads: int,
    calls_per_thread: int,
) -> tuple[list[EntityRecord], list[BaseException]]:
    """Fire new_version from many threads at once; collect records + errors."""
    barrier = threading.Barrier(threads)
    records: list[EntityRecord] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def worker(idx: int) -> None:
        reg = registries[idx % len(registries)]
        barrier.wait()
        for i in range(calls_per_thread):
            try:
                rec = reg.new_version(
                    stable_id="doc",
                    kind="doc",
                    payload={"writer": idx, "i": i},
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


def test_concurrent_new_version_one_connection_no_lost_updates(
    tmp_path: Path,
) -> None:
    """N threads sharing one registry -> exactly N versions, none lost."""
    with SqliteEntityRegistry(str(tmp_path / "reg.db")) as reg:
        threads, calls = 8, 5
        records, errors = _hammer_new_version(
            [reg], threads=threads, calls_per_thread=calls
        )
        assert errors == []
        total = threads * calls
        assert len(records) == total
        versions = sorted(r.version for r in records)
        assert versions == list(range(1, total + 1))  # strictly increasing, no gaps
        assert [r.version for r in reg.get_history("doc")] == versions
        assert reg.get_latest("doc").version == total  # type: ignore[union-attr]


def test_concurrent_new_version_across_connections(tmp_path: Path) -> None:
    """Two connections to the same file serialise via BEGIN IMMEDIATE + busy wait."""
    path = str(tmp_path / "reg.db")
    with SqliteEntityRegistry(path) as a, SqliteEntityRegistry(path) as b:
        threads, calls = 6, 4
        records, errors = _hammer_new_version(
            [a, b], threads=threads, calls_per_thread=calls
        )
        assert errors == []
        total = threads * calls
        assert len(records) == total
        assert sorted(r.version for r in records) == list(range(1, total + 1))
        assert len(a.get_history("doc")) == total


def test_file_connection_is_hardened(tmp_path: Path) -> None:
    """File-backed registries get WAL journaling and a non-zero busy timeout."""
    with SqliteEntityRegistry(str(tmp_path / "reg.db")) as reg:
        conn = reg._conn  # noqa: SLF001 - asserting on connection tuning
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] >= 1000


def test_register_and_links_still_work_under_threads(tmp_path: Path) -> None:
    """register() + link() stay correct when called from concurrent threads."""
    with SqliteEntityRegistry(str(tmp_path / "reg.db")) as reg:
        n = 12
        barrier = threading.Barrier(n)
        errors: list[BaseException] = []

        def worker(i: int) -> None:
            barrier.wait()
            try:
                rec = reg.register(
                    EntityRecord.create(
                        stable_id=f"s{i}", version=1, kind="doc", payload={"i": i}
                    )
                )
                reg.link(
                    from_record_id=rec.record_id,
                    to_record_id=rec.record_id,
                    relation="self",
                )
            except BaseException as exc:  # noqa: BLE001 - the test asserts on these
                errors.append(exc)

        pool = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        assert errors == []
        assert len(reg.list_by_kind("doc")) == n
        assert len(reg.all_links()) == n
