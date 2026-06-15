"""Shared SQLite connection hardening for the local stores.

The Studio stores are touched from the request loop and from worker threads
(``asyncio.to_thread`` persistence), so a plain connection can hit
``database is locked``. :func:`connect_hardened` opens a connection in WAL mode
with a busy timeout so concurrent readers/writers coordinate instead of failing.
"""

from __future__ import annotations

import sqlite3


def connect_hardened(
    path: str, *, busy_timeout_ms: int = 5000, synchronous: str = "NORMAL"
) -> sqlite3.Connection:
    """Open a SQLite connection tuned for concurrent local use.

    - ``check_same_thread=False`` so it can be shared across the loop + worker
      threads (callers serialize their own writes or rely on the busy timeout).
    - WAL journaling so a writer doesn't block readers.
    - ``busy_timeout`` so a contended write waits instead of raising immediately.
    - ``synchronous`` — ``NORMAL`` (default) is durable enough for a local cache and
      far faster than ``FULL``. Stores whose correctness depends on a commit surviving
      a power-cut (the audit spine, an erasure tombstone) pass ``synchronous="FULL"``
      so the WAL is fsynced on every commit, not just at checkpoint.
    """
    conn = sqlite3.connect(
        path, check_same_thread=False, timeout=busy_timeout_ms / 1000
    )
    try:
        if path != ":memory:":  # WAL is meaningless (and noisy) for in-memory DBs
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        conn.execute(f"PRAGMA synchronous={synchronous}")
    except sqlite3.Error:  # pragma: no cover - pragmas are best-effort
        pass
    return conn


__all__ = ["connect_hardened"]
