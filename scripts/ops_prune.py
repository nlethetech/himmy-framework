#!/usr/bin/env python3
"""Prune (compact) a himmy deployment's durable SQLite stores.

himmy never deletes durable history automatically: every agent turn appends run
events to the run store, every human-in-the-loop pause writes a full checkpoint
snapshot, and every REPL turn re-upserts the whole session thread. On a long-lived
Studio server those stores grow without bound — `.himmy/storage.db` and the
approvals/sessions sidecars get steadily slower and larger. This is the operator
entry point the per-store ``prune_*`` retention methods are meant to be driven from
(run it from cron / a scheduled job — it is NOT invoked by the server itself).

Each store keeps LIVE/unresolved work regardless of age:

* run events  — ``prune_events`` drops events older than the cutoff (and/or caps the
  stream to the most recent N); it never touches a row it cannot date.
* checkpoints — ``prune_resolved`` deletes only ``approved``/``rejected`` rows; an
  ``awaiting_approval``/``resolving`` checkpoint is live work and is always kept.
* graph checkpoints (``--graph-path``) — ``prune_terminal`` deletes only
  ``completed``/``failed`` rows; a ``running``/``interrupted`` graph is resumable and
  is always kept.
* sessions     — ``prune`` drops sessions not touched since the cutoff (and/or keeps
  only the most recent N).

The recommended starting policy for a long-lived server is ``--older-than-days 90``
(see docs/enterprise/deployment.md → Retention / pruning). Stores are pruned in place
through the same hardened, WAL-aware connection the server uses; run it while the
server is up (the write lock serializes against live writers) or, for a fully
quiescent compaction, after a `VACUUM`.

Usage:
    python scripts/ops_prune.py --older-than-days 90
    python scripts/ops_prune.py --older-than-days 90 --keep-last-events 1000000
    python scripts/ops_prune.py --older-than-days 30 --graph-path .himmy/graphs.db
    python scripts/ops_prune.py --keep-last-events 500000 --keep-last-sessions 200
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_DEFAULT_STORE_PATH = ".himmy/storage.db"
_DEFAULT_APPROVALS_PATH = ".himmy/approvals.db"
# As of T2.3 conversations live in the unified .himmy/conversations.db (the old
# .himmy/sessions.db is folded into it on first open); prune the unified store by default.
_DEFAULT_SESSIONS_PATH = ".himmy/conversations.db"


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _prune_events(path: Path, older_than_days: float | None, keep_last: int | None) -> int:
    """Prune the run-event stream in ``path`` (returns rows removed; 0 if absent)."""
    import asyncio

    from himmy.services.storage.sqlite import SqliteStorageService

    if not path.is_file():
        _eprint(f"  run store {path} not present — skipped")
        return 0
    store = SqliteStorageService(str(path))
    try:
        removed = asyncio.run(
            store.prune_events(older_than_days=older_than_days, keep_last=keep_last)
        )
    finally:
        asyncio.run(store.close())
    _eprint(f"  run events: pruned {removed} from {path}")
    return removed


def _prune_resolved(path: Path, older_than_days: float) -> int:
    """Prune resolved approval checkpoints in ``path`` (returns rows removed)."""
    from himmy.runtime.checkpoint import SqliteCheckpointStore

    if not path.is_file():
        _eprint(f"  approvals store {path} not present — skipped")
        return 0
    store = SqliteCheckpointStore(str(path))
    try:
        removed = store.prune_resolved(older_than_days=older_than_days)
    finally:
        store.close()
    _eprint(f"  resolved checkpoints: pruned {removed} from {path}")
    return removed


def _prune_graph(path: Path, older_than_days: float) -> int:
    """Prune terminal graph checkpoints in ``path`` (returns rows removed)."""
    from himmy.runtime.checkpoint import SqliteGraphCheckpointStore

    if not path.is_file():
        _eprint(f"  graph store {path} not present — skipped")
        return 0
    store = SqliteGraphCheckpointStore(str(path))
    try:
        removed = store.prune_terminal(older_than_days=older_than_days)
    finally:
        store.close()
    _eprint(f"  terminal graph checkpoints: pruned {removed} from {path}")
    return removed


def _prune_sessions(
    path: Path, older_than_days: float | None, keep_last: int | None
) -> int:
    """Prune stale sessions in ``path`` (returns rows removed; 0 if absent)."""
    from himmy.runtime.session import SqliteSessionStore

    if not path.is_file():
        _eprint(f"  session store {path} not present — skipped")
        return 0
    store = SqliteSessionStore(str(path))
    try:
        removed = store.prune(older_than_days=older_than_days, keep_last=keep_last)
    finally:
        store.close()
    _eprint(f"  sessions: pruned {removed} from {path}")
    return removed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--older-than-days",
        type=float,
        default=None,
        help="age cutoff in days for run events / sessions / resolved + terminal "
        "checkpoints (90 is the recommended starting policy)",
    )
    parser.add_argument(
        "--keep-last-events",
        type=int,
        default=None,
        help="additionally cap the run-event stream to the N most recent events",
    )
    parser.add_argument(
        "--keep-last-sessions",
        type=int,
        default=None,
        help="additionally cap the session store to the N most recently updated",
    )
    parser.add_argument(
        "--store-path",
        default=os.environ.get("HIMMY_STORE_PATH", _DEFAULT_STORE_PATH),
        help="run store path (default $HIMMY_STORE_PATH or .himmy/storage.db)",
    )
    parser.add_argument(
        "--approvals-path",
        default=_DEFAULT_APPROVALS_PATH,
        help="approval checkpoint store path (default .himmy/approvals.db)",
    )
    parser.add_argument(
        "--sessions-path",
        default=_DEFAULT_SESSIONS_PATH,
        help="conversation store path (default .himmy/conversations.db)",
    )
    parser.add_argument(
        "--graph-path",
        default=None,
        help="graph checkpoint store path to also prune (no default — opt in)",
    )
    args = parser.parse_args(argv)

    if args.older_than_days is None and args.keep_last_events is None and (
        args.keep_last_sessions is None
    ):
        parser.error(
            "give at least one bound: --older-than-days and/or --keep-last-events "
            "and/or --keep-last-sessions"
        )

    total = 0
    _eprint("pruning durable stores…")

    # Run events: prune if either an age bound or an events keep-last was given.
    if args.older_than_days is not None or args.keep_last_events is not None:
        total += _prune_events(
            Path(args.store_path), args.older_than_days, args.keep_last_events
        )

    # Resolved + terminal checkpoints are age-only retention.
    if args.older_than_days is not None:
        total += _prune_resolved(Path(args.approvals_path), args.older_than_days)
        if args.graph_path is not None:
            total += _prune_graph(Path(args.graph_path), args.older_than_days)

    # Sessions: prune if either an age bound or a sessions keep-last was given.
    if args.older_than_days is not None or args.keep_last_sessions is not None:
        total += _prune_sessions(
            Path(args.sessions_path), args.older_than_days, args.keep_last_sessions
        )

    _eprint(f"prune complete: {total} row(s) removed")
    print(total)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
