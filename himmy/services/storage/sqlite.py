"""Storage kernel: a durable, file-backed SQLite ``StorageService`` (offline).

A middle tier between the volatile in-memory
:class:`~himmy.services.storage.service.StorageService` and the full
:class:`~himmy.services.storage.postgres.PostgresStorageService`: durable across
process restarts (and power cuts) but just a stdlib ``sqlite3`` file — no server to
run. This is the default durable backend for server/multi-worker entrypoints when no
``HIMMY_DATABASE_URL`` is set (see :class:`himmy.services.storage.factory.StoreFactory`).

The data methods mirror the in-memory ``StorageService`` surface 1:1 and reconstruct
records by ``model_validate`` over a stored JSON ``payload`` column (exactly as the
Postgres backend does), plus indexed filter columns for the list queries. The blocking
``sqlite3`` work runs in :func:`asyncio.to_thread` so the async API never blocks the
loop; the connection is opened in WAL mode (via
:func:`himmy.core.sqlite_util.connect_hardened`) so two processes/workers sharing the
same file see each other's writes (durability) and a contended write waits instead of
raising. The parent directory is auto-created.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from himmy.core.events import RunEvent
from himmy.core.ids import iso_plus_seconds as _iso_plus_seconds
from himmy.core.ids import utc_now_iso
from himmy.core.sqlite_util import connect_hardened
from himmy.services.storage.at_rest import StorePayloadCipher, build_store_cipher
from himmy.services.storage.models import (
    ActionRecord,
    AgentDefRecord,
    AgentStateRecord,
    EnvironmentStateRecord,
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids storage <-> context cycle
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.services.context.models import ContextField, ContextSnapshot
    from himmy.services.evaluation.models import EvaluationRun
    from himmy.services.storage.trigger_dedup import DedupClaim

#: Idempotent schema for the full storage surface. Each concern table stores the full
#: record JSON in ``payload`` plus the indexed filter columns the list queries need —
#: a 1:1 mirror of the Postgres tables in :data:`himmy.services.storage.postgres.STORAGE_DDL`,
#: down to the ``runs`` partial-unique idempotency index.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id  TEXT PRIMARY KEY,
    payload    TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS run_events (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id   TEXT NOT NULL UNIQUE,
    thread_id  TEXT,
    trace_id   TEXT,
    payload    TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS run_events_thread_id_idx ON run_events (thread_id);
CREATE INDEX IF NOT EXISTS run_events_trace_id_idx ON run_events (trace_id);

CREATE TABLE IF NOT EXISTS context_fields (
    subject_id TEXT NOT NULL,
    key        TEXT NOT NULL,
    payload    TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL,
    PRIMARY KEY (subject_id, key)
);

CREATE TABLE IF NOT EXISTS context_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    subject_id  TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS context_snapshots_subject_id_idx
    ON context_snapshots (subject_id);

CREATE TABLE IF NOT EXISTS context_evidence (
    evidence_id TEXT PRIMARY KEY,
    subject_id  TEXT,
    snapshot_id TEXT,
    key         TEXT,
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS context_evidence_snapshot_id_idx
    ON context_evidence (snapshot_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    subject_id      TEXT NOT NULL,
    idempotency_key TEXT,
    status          TEXT NOT NULL,
    payload         TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_workspace_id_idx ON runs (workspace_id);
CREATE INDEX IF NOT EXISTS runs_subject_id_idx ON runs (subject_id);
CREATE INDEX IF NOT EXISTS runs_status_idx ON runs (status);
CREATE UNIQUE INDEX IF NOT EXISTS runs_idempotency_idx
    ON runs (workspace_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    workspace_id      TEXT NOT NULL,
    subject_id        TEXT NOT NULL,
    kind              TEXT NOT NULL,
    status            TEXT NOT NULL,
    payload           TEXT NOT NULL DEFAULT '{}',
    created_at        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS recommendations_workspace_id_idx
    ON recommendations (workspace_id);
CREATE INDEX IF NOT EXISTS recommendations_subject_id_idx
    ON recommendations (subject_id);
CREATE INDEX IF NOT EXISTS recommendations_run_id_idx ON recommendations (run_id);
CREATE INDEX IF NOT EXISTS recommendations_status_idx ON recommendations (status);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id     TEXT PRIMARY KEY,
    suite_id   TEXT,
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS evaluation_runs_suite_id_idx
    ON evaluation_runs (suite_id);

CREATE TABLE IF NOT EXISTS memory_objects (
    memory_id  TEXT PRIMARY KEY,
    subject_id TEXT,
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_objects_subject_id_idx
    ON memory_objects (subject_id);

CREATE TABLE IF NOT EXISTS episodic_memory_objects (
    episode_id TEXT PRIMARY KEY,
    subject_id TEXT,
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS episodic_memory_objects_subject_id_idx
    ON episodic_memory_objects (subject_id);

CREATE TABLE IF NOT EXISTS agent_states (
    state_id         TEXT PRIMARY KEY,
    environment_name TEXT,
    round            INTEGER,
    payload          TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS agent_states_env_round_idx
    ON agent_states (environment_name, round);

CREATE TABLE IF NOT EXISTS actions (
    action_id        TEXT PRIMARY KEY,
    environment_name TEXT,
    round            INTEGER,
    payload          TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS actions_env_round_idx
    ON actions (environment_name, round);

CREATE TABLE IF NOT EXISTS environment_states (
    environment_state_id TEXT PRIMARY KEY,
    environment_name     TEXT,
    round                INTEGER,
    payload              TEXT NOT NULL DEFAULT '{}',
    created_at           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS environment_states_env_round_idx
    ON environment_states (environment_name, round);
"""

#: Schema version of the base :data:`_SCHEMA` above (PRAGMA user_version after it is
#: applied). Bump this and append to :data:`_MIGRATIONS` whenever a table/column/index
#: is added so existing ``.himmy/storage.db`` files upgrade forward in lock-step rather
#: than silently diverging (``CREATE TABLE IF NOT EXISTS`` is a no-op on an existing db).
SQLITE_SCHEMA_VERSION = 5

#: Ordered forward migrations applied after the base DDL, mirroring the Postgres
#: :data:`himmy.services.storage.postgres.STORAGE_MIGRATIONS`. Each entry is
#: ``(version, [statements...])`` and is applied exactly once, gated by
#: ``PRAGMA user_version``: a database steps through every migration whose version
#: exceeds its current ``user_version`` and is then stamped at the highest version.
#: The convention is **frozen base + forward migrations** — :data:`_SCHEMA` is the
#: version-1 shape and is NEVER edited in place, so a fresh database (``user_version``
#: 0) lays down that frozen base and then runs *every* migration (version 2, 3, …)
#: just like a legacy file would, converging to the exact same schema by either path.
#: (Editing :data:`_SCHEMA` to add a column AND re-adding it via a migration would
#: double-apply the ALTER and break fresh installs.) Append new entries here — never
#: edit a shipped one — and bump :data:`SQLITE_SCHEMA_VERSION` to match the highest.
_MIGRATIONS: tuple[tuple[int, tuple[str, ...]], ...] = (
    # v2 (P0-B): promote event_type + tool_name to indexed columns on run_events
    # so learning miners can answer "which tools fail most" / "what gets blocked"
    # with an index seek instead of a full-table json_extract scan. New writes
    # populate the columns directly (append_event); existing rows are backfilled
    # from the stored JSON best-effort (top-level event_type is always plaintext;
    # the nested tool_name backfills only for unencrypted databases).
    (
        2,
        (
            "ALTER TABLE run_events ADD COLUMN event_type TEXT",
            "ALTER TABLE run_events ADD COLUMN tool_name TEXT",
            "CREATE INDEX IF NOT EXISTS run_events_event_type_idx "
            "ON run_events (event_type)",
            "CREATE INDEX IF NOT EXISTS run_events_tool_name_idx "
            "ON run_events (tool_name)",
            "UPDATE run_events SET event_type = "
            "json_extract(payload, '$.event_type') WHERE event_type IS NULL",
            "UPDATE run_events SET tool_name = "
            "json_extract(payload, '$.payload.tool_name') "
            "WHERE tool_name IS NULL AND json_valid(payload)",
        ),
    ),
    # v3 (T1.1): promote workspace_id to an indexed column on evaluation_runs so
    # GET /v1/evaluation/runs/{id} can be tenant-scoped (close the cross-tenant
    # IDOR) and lists can filter by workspace with an index seek. New writes
    # populate the column directly (save_evaluation_run); existing rows are
    # backfilled best-effort from the stored JSON payload (top-level
    # workspace_id; only for unencrypted databases where the column is plaintext).
    (
        3,
        (
            "ALTER TABLE evaluation_runs ADD COLUMN workspace_id TEXT",
            "CREATE INDEX IF NOT EXISTS evaluation_runs_workspace_id_idx "
            "ON evaluation_runs (workspace_id)",
            "UPDATE evaluation_runs SET workspace_id = "
            "json_extract(payload, '$.workspace_id') "
            "WHERE workspace_id IS NULL AND json_valid(payload)",
        ),
    ),
    # v4 (T2e): durable workspace-scoped stored agent definitions backing the
    # /v1/agents CRUD resource. ``agent_id`` is the PK; the full AgentDefRecord JSON
    # (incl. the sanitized AgentSpec) rides in ``payload``; ``workspace_id`` is an
    # indexed column for tenant-scoped reads and a partial-unique idempotency index
    # mirrors the runs table so a re-POSTed idempotency key returns the prior row.
    (
        4,
        (
            "CREATE TABLE IF NOT EXISTS agent_defs ("
            "agent_id TEXT PRIMARY KEY, "
            "workspace_id TEXT NOT NULL, "
            "name TEXT NOT NULL, "
            "idempotency_key TEXT, "
            "payload TEXT NOT NULL DEFAULT '{}', "
            "created_at TEXT NOT NULL, "
            "updated_at TEXT NOT NULL)",
            "CREATE INDEX IF NOT EXISTS agent_defs_workspace_id_idx "
            "ON agent_defs (workspace_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS agent_defs_idempotency_idx "
            "ON agent_defs (workspace_id, idempotency_key) "
            "WHERE idempotency_key IS NOT NULL",
        ),
    ),
    # v5 (Q1): the leased run work-queue surface on the ``runs`` table + a durable inbound
    # ``trigger_dedup`` table. The eight queue columns promote the at-most-once dispatch +
    # lease + retry/backoff + provider-lane state into INDEXED columns so the Q2
    # ``claim_next_queued_run`` CAS does an index seek (status, next_attempt_at) instead of a
    # full-table json_extract scan, and the reaper finds expired leases by (status,
    # lease_expires_at). ``attempt`` backfills 0 and ``next_attempt_at`` backfills to the
    # existing ``created_at`` so every legacy QUEUED row is immediately claimable. The columns
    # are ALSO carried in the RunRecord JSON ``payload`` (the canonical record), so a write
    # populates both; these columns are the queryable projection. ``trigger_dedup`` backs the
    # Q4 DurableIdempotencyStore (scope+key TTL-CAS) — created here so the queue + dedup share
    # one migration, mirroring the Postgres v7 chain.
    (
        5,
        (
            "ALTER TABLE runs ADD COLUMN owner_id TEXT",
            "ALTER TABLE runs ADD COLUMN lease_expires_at TEXT",
            "ALTER TABLE runs ADD COLUMN heartbeat_at TEXT",
            "ALTER TABLE runs ADD COLUMN attempt INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN max_attempts INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE runs ADD COLUMN next_attempt_at TEXT",
            "ALTER TABLE runs ADD COLUMN last_error TEXT",
            "ALTER TABLE runs ADD COLUMN lane_key TEXT",
            # Backfill: legacy rows are immediately claimable (attempt 0, ready at created_at).
            "UPDATE runs SET attempt = 0 WHERE attempt IS NULL",
            "UPDATE runs SET next_attempt_at = created_at WHERE next_attempt_at IS NULL",
            # Claim seek: oldest READY run in a lane. Reaper seek: expired RUNNING leases.
            "CREATE INDEX IF NOT EXISTS runs_status_next_attempt_idx "
            "ON runs (status, next_attempt_at)",
            "CREATE INDEX IF NOT EXISTS runs_status_lease_idx "
            "ON runs (status, lease_expires_at)",
            "CREATE INDEX IF NOT EXISTS runs_lane_key_idx ON runs (lane_key)",
            # Durable inbound dedup (Q4): a scope+key TTL-CAS ledger. ``result`` is an optional
            # small JSON string the handler may stash; ``expires_at`` is the TTL gate.
            "CREATE TABLE IF NOT EXISTS trigger_dedup ("
            "scope TEXT NOT NULL, "
            "key TEXT NOT NULL, "
            "result TEXT, "
            "expires_at TEXT, "
            "created_at TEXT NOT NULL, "
            "PRIMARY KEY (scope, key))",
            "CREATE INDEX IF NOT EXISTS trigger_dedup_expires_idx "
            "ON trigger_dedup (expires_at)",
        ),
    ),
)

#: Max bound parameters per ``DELETE ... WHERE seq IN (?, …)`` when pruning. SQLite caps
#: a statement at ``SQLITE_MAX_VARIABLE_NUMBER`` host parameters (999 on libsqlite < 3.32,
#: 32,766 after) and raises ``sqlite3.OperationalError: too many SQL variables`` past it —
#: the first prune of a long-lived database can easily exceed that. The doomed seqs are
#: deleted in batches of this size so the IN-list never overruns the lower historical
#: limit, regardless of the linked SQLite version.
_PRUNE_DELETE_CHUNK = 900


def _chunked(items: list[int], size: int) -> Iterator[list[int]]:
    """Yield ``items`` in contiguous slices of at most ``size`` (size >= 1)."""
    step = max(int(size), 1)
    for start in range(0, len(items), step):
        yield items[start : start + step]


class _Unset:
    """Sentinel: ``cipher`` was not passed, so source it from the environment.

    Distinct from an explicit ``cipher=None`` (which forces plaintext regardless of
    ``HIMMY_ENCRYPTION_KEY``) so a caller can opt out even when a key is configured.
    """


_UNSET = _Unset()


class SqliteStorageService:
    """A durable, file-backed SQLite storage backend mirroring ``StorageService`` 1:1.

    Satisfies :class:`~himmy.core.events.EventSink` (``append_event``) and the
    :class:`~himmy.services.storage.protocols.ThreadEventStore` protocol — and every
    other per-concern method the in-memory facade exposes. State is persisted to the
    SQLite file at ``path`` and survives process restarts: two instances on the same
    file see each other's writes. Blocking ``sqlite3`` calls run in
    :func:`asyncio.to_thread`, so the async surface never blocks the event loop.
    """

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        cipher: StorePayloadCipher | None | _Unset = _UNSET,
    ) -> None:
        """Open (or create) the SQLite database at ``path`` and apply the schema.

        ``path``'s parent directory is created when it does not exist, so a default
        like ``.himmy/storage.db`` works on first run. ``:memory:`` is accepted for
        ephemeral use (the connection is shared, so the in-memory DB persists for the
        lifetime of the instance). A process-wide lock serializes writes so concurrent
        ``asyncio.to_thread`` workers can share the single connection safely.

        ``cipher`` controls field encryption at rest for the sensitive payloads (message
        content + run-event payloads). It defaults to :func:`build_store_cipher`, which is
        ``None`` (plaintext, offline path unchanged) unless ``HIMMY_ENCRYPTION_KEY`` /
        ``HIMMY_KEK_PROVIDER`` is configured; pass an explicit cipher (or ``None``) to
        override.
        """
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._conn = connect_hardened(self._path)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self._apply_schema()
        self._cipher = build_store_cipher() if isinstance(cipher, _Unset) else cipher

    def _apply_schema(self) -> None:
        """Create/upgrade the schema and stamp ``PRAGMA user_version`` (idempotent).

        The base :data:`_SCHEMA` is ``CREATE ... IF NOT EXISTS`` so it is a no-op on an
        already-provisioned database — but that means a new column added in a later
        release would *never* reach an existing file, and the first INSERT naming it
        would raise. To evolve safely, a ``PRAGMA user_version`` records the applied
        schema version: a fresh database (version 0, no tables) gets the frozen base DDL
        and an existing database does not, but BOTH then step through every entry in
        :data:`_MIGRATIONS` whose version exceeds the stored ``user_version`` (so a fresh
        db at version 0 runs every migration, converging to the same schema a legacy file
        reaches) and the highest known version is stamped at the end. The whole upgrade
        runs under the write lock so concurrent workers/processes sharing the file can't
        double-apply a step.
        """
        with self._lock:
            current = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
            try:
                if current == 0:
                    # Either a brand-new database, OR a legacy file written before
                    # versioning existed (user_version defaults to 0). Re-running the
                    # idempotent base DDL is safe in both cases.
                    self._conn.executescript(_SCHEMA)
                for version, statements in sorted(_MIGRATIONS):
                    if version > current:
                        for statement in statements:
                            self._conn.execute(statement)  # noqa: S608 - constant DDL
                # Stamp the highest known version (PRAGMA cannot be parameterised).
                target = max(SQLITE_SCHEMA_VERSION, current)
                self._conn.execute(f"PRAGMA user_version = {int(target)}")
                self._conn.commit()
            except BaseException:
                self._rollback_quietly()
                raise

    @property
    def schema_version(self) -> int:
        """The applied schema version (``PRAGMA user_version``) of the backing file."""
        with self._lock:
            return int(self._conn.execute("PRAGMA user_version").fetchone()[0])

    @property
    def path(self) -> str:
        """The filesystem path of the backing SQLite database (or ``:memory:``)."""
        return self._path

    async def close(self) -> None:
        """Close the underlying connection (idempotent; awaited on container teardown)."""
        await asyncio.to_thread(self._conn.close)

    async def __aenter__(self) -> SqliteStorageService:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    # ------------------------------------------------------------- sync primitives
    def _rollback_quietly(self) -> None:
        """Roll back the shared connection, swallowing rollback-time errors.

        Called from write-path exception handlers: the original write failure is
        what the caller must see, so a secondary error rolling back an already-dead
        connection must not mask it.
        """
        try:
            self._conn.rollback()
        except sqlite3.Error:  # pragma: no cover - rollback on a dead connection
            pass

    def _write(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """Run a write statement under the write lock and commit.

        Rolls back on any failure (constraint violation, disk error, failed commit)
        before re-raising, so the shared connection is never left inside an open
        transaction — open-transaction residue would hold the WAL write lock against
        other processes and silently fold the torn write into the next caller's
        commit.
        """
        with self._lock:
            try:
                self._conn.execute(sql, params)
                self._conn.commit()
            except BaseException:
                self._rollback_quietly()
                raise

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        """Run a read returning at most one row."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            return cast("sqlite3.Row | None", cur.fetchone())

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        """Run a read returning all rows."""
        with self._lock:
            cur = self._conn.execute(sql, params)
            return list(cur.fetchall())

    @staticmethod
    def _dump(obj: Any) -> str:
        """Serialize a pydantic model to its canonical JSON string for the payload column."""
        return json.dumps(obj.model_dump(mode="json"))

    # ------------------------------------------------------------------ threads
    async def save_thread(self, thread: ChatThread) -> ChatThread:
        """Upsert a chat thread keyed by ``thread_id`` (message content encrypted)."""
        payload = thread.model_dump(mode="json")
        if self._cipher is not None:
            payload = self._cipher.encrypt_thread_payload(
                payload, thread_id=thread.thread_id
            )
        await asyncio.to_thread(
            self._write,
            "INSERT INTO chat_threads (thread_id, payload, updated_at) "
            "VALUES (?, ?, ?) ON CONFLICT (thread_id) DO UPDATE SET "
            "payload = excluded.payload, updated_at = excluded.updated_at",
            (thread.thread_id, json.dumps(payload), utc_now_iso()),
        )
        return thread

    async def load_thread(self, thread_id: str) -> ChatThread | None:
        """Return a stored chat thread by id, or None (message content decrypted)."""
        from himmy.agents.base_agent.thread import ChatThread

        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM chat_threads WHERE thread_id = ?",
            (thread_id,),
        )
        if row is None:
            return None
        payload = json.loads(row["payload"])
        if self._cipher is not None:
            payload = self._cipher.decrypt_thread_payload(payload, thread_id=thread_id)
        return ChatThread.model_validate(payload)

    # ------------------------------------------------------------------- events
    async def append_event(self, event: RunEvent) -> None:
        """Append a run event to the canonical audit stream (EventSink).

        Idempotent on ``event_id`` (a re-appended event is ignored), mirroring the
        Postgres ``ON CONFLICT (event_id) DO NOTHING``. Insertion order is preserved
        by the autoincrement ``seq`` column.
        """
        record = event.model_dump(mode="json")
        if self._cipher is not None:
            record["payload"] = self._cipher.encrypt_event_payload(
                record.get("payload") or {}, event_id=event.event_id
            )
        # P0-B: denormalise event_type + tool_name into indexed columns so the
        # learning miners query by index. Captured from the live event BEFORE
        # encryption, so tool_name survives even with payload encryption at rest.
        event_type = getattr(event.event_type, "value", str(event.event_type))
        tool_name = (event.payload or {}).get("tool_name")
        await asyncio.to_thread(
            self._write,
            "INSERT INTO run_events "
            "(event_id, thread_id, trace_id, event_type, tool_name, payload) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT (event_id) DO NOTHING",
            (
                event.event_id,
                event.thread_id,
                event.trace_id,
                event_type,
                tool_name,
                json.dumps(record),
            ),
        )

    async def list_events(
        self, thread_id: str | None = None, trace_id: str | None = None
    ) -> list[RunEvent]:
        """List events, optionally filtered by ``thread_id`` and/or ``trace_id``."""
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if trace_id is not None:
            clauses.append("trace_id = ?")
            params.append(trace_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await asyncio.to_thread(
            self._fetchall,
            f"SELECT payload FROM run_events{where} ORDER BY seq ASC",  # noqa: S608
            tuple(params),
        )
        return [self._row_to_event(json.loads(r["payload"])) for r in rows]

    async def prune_events(
        self, *, older_than_days: float | None = None, keep_last: int | None = None
    ) -> int:
        """Delete old run events, bounding the audit stream's unbounded growth.

        Returns the number of rows deleted. This is an explicit, operator-invoked
        retention call — nothing deletes automatically. Callers must drive it from
        their own ops job; the bundled ``scripts/ops_prune.py`` does exactly this for
        the default ``.himmy`` stores. The default policy is conservative: pass
        ``older_than_days`` to drop events whose recorded ``timestamp`` is older than the
        cutoff, and/or ``keep_last`` to additionally cap the stream to the N most recent
        events by insertion order (``seq``). At least one bound must be given; both may
        be combined (a row is pruned if it fails EITHER bound). The recommended starting
        policy for a long-lived Studio server is ``older_than_days=90`` run periodically.

        ``older_than_days`` reads each candidate row's ISO ``timestamp`` from the stored
        payload (decrypting first when at-rest encryption is on), so it is correct
        regardless of clock skew between writers; ``keep_last`` is a pure ``seq`` cap and
        never decrypts. The delete runs under the write lock as a single statement set.
        """
        if older_than_days is None and keep_last is None:
            raise ValueError(
                "prune_events needs at least one of older_than_days / keep_last"
            )
        return await asyncio.to_thread(self._prune_events_sync, older_than_days, keep_last)

    def _prune_events_sync(
        self, older_than_days: float | None, keep_last: int | None
    ) -> int:
        """The locked delete for :meth:`prune_events` (returns rows removed)."""
        from datetime import datetime, timedelta

        doomed: set[int] = set()
        with self._lock:
            try:
                if older_than_days is not None:
                    cutoff = datetime.now().astimezone() - timedelta(days=older_than_days)
                    rows = self._conn.execute(
                        "SELECT seq, payload FROM run_events ORDER BY seq ASC"
                    ).fetchall()
                    for row in rows:
                        ts = self._event_timestamp(json.loads(row["payload"]))
                        if ts is not None and ts < cutoff:
                            doomed.add(int(row["seq"]))
                if keep_last is not None:
                    keep = max(int(keep_last), 0)
                    over = self._conn.execute(
                        "SELECT seq FROM run_events ORDER BY seq DESC LIMIT -1 OFFSET ?",
                        (keep,),
                    ).fetchall()
                    doomed.update(int(r["seq"]) for r in over)
                if not doomed:
                    return 0
                # Delete in chunks so the bound-parameter IN-list never exceeds
                # SQLITE_MAX_VARIABLE_NUMBER (which raises "too many SQL variables").
                # The whole prune is one transaction: all chunks commit together, so a
                # failure midway leaves the stream untouched rather than half-pruned.
                ordered = sorted(doomed)
                for batch in _chunked(ordered, _PRUNE_DELETE_CHUNK):
                    placeholders = ",".join("?" for _ in batch)
                    self._conn.execute(
                        f"DELETE FROM run_events WHERE seq IN ({placeholders})",  # noqa: S608
                        tuple(batch),
                    )
                self._conn.commit()
                return len(doomed)
            except BaseException:
                self._rollback_quietly()
                raise

    def delete_by_subject(self, subject_id: str) -> int:
        """Hard-DELETE a subject's spine ``chat_threads`` + ``run_events`` rows (S4).

        These two tables hold the subject's full conversation transcript and run-event
        stream, encrypted at rest under the store-WIDE KEK (``self._cipher``) — NOT the
        per-subject key — so a crypto-shred of the subject key leaves them recoverable.
        Right-to-erasure therefore reaches them by HARD-DELETE here (mirroring the mutable
        sidecars), the cheaper match to the reach-map design than re-keying every spine
        write per subject.

        The subject->thread/trace linkage is the ``runs`` table (``runs.subject_id`` is an
        indexed column; each run's ``payload`` carries its ``thread_id`` / ``trace_id``):
        a subject's runs name every thread + event-stream the runtime persisted for it.
        We resolve those ids, then delete the matching ``chat_threads`` (by ``thread_id``)
        and ``run_events`` (by ``thread_id`` OR ``trace_id``). Returns the total rows
        removed. Runs as ONE transaction under the write lock so a mid-delete failure
        leaves the tables untouched; idempotent (a re-run deletes whatever survived and
        returns 0 once clean). Synchronous so the sync ``SubjectReachMap.erase`` can drive
        it directly without re-entering the request event loop.
        """
        with self._lock:
            try:
                run_rows = self._conn.execute(
                    "SELECT json_extract(payload, '$.thread_id') AS thread_id, "
                    "json_extract(payload, '$.trace_id') AS trace_id "
                    "FROM runs WHERE subject_id = ?",
                    (subject_id,),
                ).fetchall()
                thread_ids = {r["thread_id"] for r in run_rows if r["thread_id"]}
                trace_ids = {r["trace_id"] for r in run_rows if r["trace_id"]}
                deleted = 0
                for tid in thread_ids:
                    cur = self._conn.execute(
                        "DELETE FROM chat_threads WHERE thread_id = ?", (tid,)
                    )
                    deleted += cur.rowcount
                    cur = self._conn.execute(
                        "DELETE FROM run_events WHERE thread_id = ?", (tid,)
                    )
                    deleted += cur.rowcount
                for trace in trace_ids:
                    cur = self._conn.execute(
                        "DELETE FROM run_events WHERE trace_id = ?", (trace,)
                    )
                    deleted += cur.rowcount
                self._conn.commit()
                return deleted
            except BaseException:
                self._rollback_quietly()
                raise

    def _event_timestamp(self, record: dict[str, Any]) -> Any:
        """Parse a stored event's ISO ``timestamp`` to an aware datetime, or None."""
        from datetime import datetime

        raw = record.get("timestamp")
        if not isinstance(raw, str) or not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError:  # pragma: no cover - malformed legacy timestamp
            return None
        return parsed if parsed.tzinfo else parsed.astimezone()

    def _row_to_event(self, record: dict[str, Any]) -> RunEvent:
        """Reconstruct a ``RunEvent``, decrypting its payload envelope when present."""
        if self._cipher is not None:
            record = dict(record)
            record["payload"] = self._cipher.decrypt_event_payload(
                record.get("payload") or {}, event_id=str(record.get("event_id", ""))
            )
        return RunEvent.model_validate(record)

    # ------------------------------------------------------------------ context
    async def save_context_field(self, field: ContextField) -> ContextField:
        """Upsert a context field keyed by ``(subject_id, key)``.

        ``subject_id`` is read from the field metadata when present, falling back to a
        blank scope so storage-only fields still round-trip (mirrors the in-memory store).
        """
        subject_id = str(getattr(field, "metadata", {}).get("subject_id", ""))
        await asyncio.to_thread(
            self._write,
            "INSERT INTO context_fields (subject_id, key, payload, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (subject_id, key) DO UPDATE SET "
            "payload = excluded.payload, updated_at = excluded.updated_at",
            (subject_id, field.key, self._dump(field), utc_now_iso()),
        )
        return field

    async def get_context_field(self, subject_id: str, key: str) -> ContextField | None:
        """Return the context field for ``(subject_id, key)``, or None."""
        from himmy.services.context.models import ContextField

        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM context_fields WHERE subject_id = ? AND key = ?",
            (subject_id, key),
        )
        return ContextField.model_validate(json.loads(row["payload"])) if row else None

    async def list_context_fields(self, subject_id: str) -> list[ContextField]:
        """Return all context fields for a subject."""
        from himmy.services.context.models import ContextField

        rows = await asyncio.to_thread(
            self._fetchall,
            "SELECT payload FROM context_fields WHERE subject_id = ?",
            (subject_id,),
        )
        return [ContextField.model_validate(json.loads(r["payload"])) for r in rows]

    async def save_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        """Upsert a context snapshot keyed by ``snapshot_id``."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO context_snapshots (snapshot_id, subject_id, payload, "
            "created_at) VALUES (?, ?, ?, ?) ON CONFLICT (snapshot_id) DO UPDATE SET "
            "subject_id = excluded.subject_id, payload = excluded.payload",
            (
                snapshot.snapshot_id,
                snapshot.subject_id,
                self._dump(snapshot),
                snapshot.created_at,
            ),
        )
        return snapshot

    async def load_snapshot(self, snapshot_id: str) -> ContextSnapshot | None:
        """Return a stored snapshot by id, or None."""
        from himmy.services.context.models import ContextSnapshot

        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM context_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        )
        return (
            ContextSnapshot.model_validate(json.loads(row["payload"])) if row else None
        )

    async def save_context_evidence(self, record: object) -> object:
        """Append a context evidence record to the evidence stream (idempotent on id)."""
        from himmy.core.ids import new_uuid

        # Accept either a ContextEvidenceRecord or an EvidenceRef-like object (mirrors
        # the Postgres backend); ``object`` widens to ``Any`` for the duck-typed access.
        rec: Any = record
        evidence_id = str(getattr(rec, "evidence_id", None) or new_uuid())
        payload = (
            rec.model_dump(mode="json") if hasattr(rec, "model_dump") else dict(rec)
        )
        created_at = str(getattr(rec, "created_at", None) or utc_now_iso())
        await asyncio.to_thread(
            self._write,
            "INSERT INTO context_evidence (evidence_id, subject_id, snapshot_id, "
            "key, payload, created_at) VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (evidence_id) DO UPDATE SET "
            "subject_id = excluded.subject_id, snapshot_id = excluded.snapshot_id, "
            "key = excluded.key, payload = excluded.payload",
            (
                evidence_id,
                getattr(rec, "subject_id", None),
                getattr(rec, "snapshot_id", None),
                getattr(rec, "key", None),
                json.dumps(payload),
                created_at,
            ),
        )
        return record

    # --------------------------------------------------------------------- runs
    async def save_run(self, run: RunRecord) -> RunRecord:
        """Upsert a run record keyed by ``run_id``; storage stamps ``updated_at``."""
        run.updated_at = utc_now_iso()
        await asyncio.to_thread(self._write, *self._run_upsert(run))
        return run

    async def save_run_if_absent_by_idempotency(
        self, run: RunRecord
    ) -> tuple[RunRecord, bool]:
        """Atomically create a run unless its idempotency key already exists.

        Returns ``(run, created)``. The write lock + the ``runs_idempotency_idx``
        partial UNIQUE index close the create race: the read-existing and the insert
        happen under one lock acquisition, so two concurrent callers with the same key
        cannot both create a run. Runs without an idempotency key are always created.
        """
        run.updated_at = utc_now_iso()
        return await asyncio.to_thread(self._save_run_if_absent_sync, run)

    def _save_run_if_absent_sync(self, run: RunRecord) -> tuple[RunRecord, bool]:
        """The locked read-then-insert for the idempotent run create.

        Runs as a single ``BEGIN IMMEDIATE`` transaction so the existence check and
        the insert are atomic even across processes sharing the file (the write lock
        is taken before the read), and rolls back on failure so a lost race on
        ``runs_idempotency_idx`` cannot leave the transaction open.
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if run.idempotency_key is not None:
                    existing = self._conn.execute(
                        "SELECT payload FROM runs WHERE workspace_id = ? AND "
                        "idempotency_key = ?",
                        (run.workspace_id, run.idempotency_key),
                    ).fetchone()
                    if existing is not None:
                        self._conn.commit()
                        return self._row_to_run(existing), False
                sql, params = self._run_upsert(run)
                self._conn.execute(sql, params)
                self._conn.commit()
                return run, True
            except BaseException:
                self._rollback_quietly()
                raise

    async def claim_run_for_resume(
        self, run_id: str, *, workspace_id: str
    ) -> bool:
        """Atomically claim an AWAITING_APPROVAL run for resume (True iff we won).

        The run-level mirror of :meth:`SqliteCheckpointStore.claim`: a single conditional
        UPDATE under ``BEGIN IMMEDIATE`` (which takes the write lock up front) flips
        ``AWAITING_APPROVAL`` -> ``RESOLVING`` for ``(run_id, workspace_id)``, returning
        ``rowcount == 1`` for the winner. Two concurrent resumes — even across processes
        sharing this file — serialize on the write lock; the first matches
        ``AWAITING_APPROVAL`` and flips it, the second now sees ``RESOLVING`` and matches
        nothing (rowcount 0), so a SECOND concurrent approve loses the CAS and is refused
        BEFORE any resume launches (closing the orchestration double-advance race). Only
        ``AWAITING_APPROVAL`` is claimable here; a stale ``RESOLVING`` (a crashed resume)
        is re-driven by recovery, not re-claimed by a fresh approve. Both the ``status``
        column and the JSON ``payload``'s ``$.status`` are rewritten (and ``updated_at``)
        so a later ``get_run`` is consistent with the column.
        """
        return await asyncio.to_thread(
            self._claim_run_for_resume_sync, run_id, workspace_id
        )

    def _claim_run_for_resume_sync(self, run_id: str, workspace_id: str) -> bool:
        """The locked conditional UPDATE for the run-level resume claim."""
        now = utc_now_iso()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cur = self._conn.execute(
                    "UPDATE runs SET status = ?, updated_at = ?, "
                    "payload = json_set(payload, '$.status', ?, '$.updated_at', ?) "
                    "WHERE run_id = ? AND workspace_id = ? AND status = ?",
                    (
                        RunStatus.RESOLVING.value,
                        now,
                        RunStatus.RESOLVING.value,
                        now,
                        run_id,
                        workspace_id,
                        RunStatus.AWAITING_APPROVAL.value,
                    ),
                )
                won = cur.rowcount == 1
                self._conn.commit()
                return won
            except BaseException:
                self._rollback_quietly()
                raise

    # ----------------------------------------------------- leased work-queue (Q2)
    async def claim_next_queued_run(
        self,
        owner_id: str,
        lease_seconds: float,
        *,
        lanes: list[str] | None = None,
        now: str | None = None,
    ) -> RunRecord | None:
        """Atomically claim the oldest ready QUEUED run for ``owner_id`` (or None).

        The structural twin of :meth:`claim_run_for_resume`: a ``BEGIN IMMEDIATE`` (write lock
        up front) selects the OLDEST ready row — ``status=QUEUED`` and ``next_attempt_at <=
        now`` — by its ``rowid`` (a sub-SELECT, NOT ``UPDATE … LIMIT`` so we do not depend on
        the optional ``SQLITE_ENABLE_UPDATE_DELETE_LIMIT`` compile flag), flips it to
        ``RUNNING``, stamps ``owner_id``/``lease_expires_at``/``heartbeat_at`` and increments
        ``attempt``. Two workers racing serialize on the write lock: the first flips the row,
        the second's sub-SELECT no longer sees it (status changed) and claims the next one or
        returns None — so a given QUEUED run produces exactly one RUNNING claim.

        ``lanes`` restricts the claim to runs whose ``lane_key`` is in the set (Q2 lane
        keying): a dispatcher whose LOCAL-model probe is failing passes only the cloud/default
        lanes, so a stalled local probe never blocks claiming cloud runs. ``None`` claims any
        lane.
        """
        return await asyncio.to_thread(
            self._claim_next_queued_run_sync, owner_id, lease_seconds, lanes, now
        )

    def _claim_next_queued_run_sync(
        self,
        owner_id: str,
        lease_seconds: float,
        lanes: list[str] | None,
        now: str | None,
    ) -> RunRecord | None:
        """The locked select-oldest-then-CAS claim for the leased queue."""
        now_iso = now or utc_now_iso()
        lease_until = _iso_plus_seconds(now_iso, lease_seconds)
        lane_clause = ""
        lane_params: tuple[Any, ...] = ()
        if lanes is not None:
            if not lanes:
                # An empty allowlist means "no claimable lane" — claim nothing.
                return None
            placeholders = ",".join("?" for _ in lanes)
            lane_clause = f" AND lane_key IN ({placeholders})"
            lane_params = tuple(lanes)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                # Oldest ready row by rowid (created_at order ~ insertion order); rowid is the
                # stable tiebreak. The sub-SELECT avoids the UPDATE…LIMIT compile dependency.
                target = self._conn.execute(
                    "SELECT rowid, payload, attempt FROM runs "  # noqa: S608
                    "WHERE status = ? AND (next_attempt_at IS NULL OR next_attempt_at <= ?)"
                    f"{lane_clause} "
                    "ORDER BY next_attempt_at ASC, rowid ASC LIMIT 1",
                    (RunStatus.QUEUED.value, now_iso, *lane_params),
                ).fetchone()
                if target is None:
                    self._conn.commit()
                    return None
                new_attempt = int(target["attempt"] or 0) + 1
                self._conn.execute(
                    "UPDATE runs SET status = ?, owner_id = ?, lease_expires_at = ?, "
                    "heartbeat_at = ?, attempt = ?, updated_at = ?, "
                    "payload = json_set(payload, '$.status', ?, '$.owner_id', ?, "
                    "'$.lease_expires_at', ?, '$.heartbeat_at', ?, '$.attempt', ?, "
                    "'$.updated_at', ?) "
                    "WHERE rowid = ? AND status = ?",
                    (
                        RunStatus.RUNNING.value,
                        owner_id,
                        lease_until,
                        now_iso,
                        new_attempt,
                        now_iso,
                        RunStatus.RUNNING.value,
                        owner_id,
                        lease_until,
                        now_iso,
                        new_attempt,
                        now_iso,
                        target["rowid"],
                        RunStatus.QUEUED.value,
                    ),
                )
                row = self._conn.execute(
                    "SELECT payload FROM runs WHERE rowid = ?", (target["rowid"],)
                ).fetchone()
                self._conn.commit()
                return self._row_to_run(row) if row is not None else None
            except BaseException:
                self._rollback_quietly()
                raise

    async def renew_lease(
        self,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
        *,
        now: str | None = None,
    ) -> bool:
        """Extend a RUNNING run's lease iff ``owner_id`` still holds it (True iff renewed).

        The lease-keepalive a live worker sends periodically so the reaper never re-queues it
        mid-flight. Conditional on ``status=RUNNING AND owner_id=?`` so a worker that already
        lost its lease (a reaper re-queued it, another worker re-claimed) cannot resurrect its
        ownership — it gets ``False`` and must stop.
        """
        return await asyncio.to_thread(
            self._renew_lease_sync, run_id, owner_id, lease_seconds, now
        )

    def _renew_lease_sync(
        self, run_id: str, owner_id: str, lease_seconds: float, now: str | None
    ) -> bool:
        """The locked conditional lease renewal."""
        now_iso = now or utc_now_iso()
        lease_until = _iso_plus_seconds(now_iso, lease_seconds)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cur = self._conn.execute(
                    "UPDATE runs SET lease_expires_at = ?, heartbeat_at = ?, "
                    "updated_at = ?, payload = json_set(payload, '$.lease_expires_at', ?, "
                    "'$.heartbeat_at', ?, '$.updated_at', ?) "
                    "WHERE run_id = ? AND owner_id = ? AND status = ?",
                    (
                        lease_until,
                        now_iso,
                        now_iso,
                        lease_until,
                        now_iso,
                        now_iso,
                        run_id,
                        owner_id,
                        RunStatus.RUNNING.value,
                    ),
                )
                won = cur.rowcount == 1
                self._conn.commit()
                return won
            except BaseException:
                self._rollback_quietly()
                raise

    async def requeue_expired_leases(
        self, *, now: str | None = None, lanes: list[str] | None = None
    ) -> list[str]:
        """Re-queue RUNNING runs whose lease expired; return the re-queued run_ids.

        The reaper primitive. ONLY ``status=RUNNING`` runs with ``lease_expires_at <= now`` are
        touched — a live peer renewing its lease has a future deadline and is NEVER reaped, and
        ``AWAITING_APPROVAL``/``RESOLVING``/``QUEUED``/terminal runs are excluded by the status
        predicate (preserving the HITL exclusions). Each is reset to ``QUEUED`` with its
        ``owner_id``/``lease_expires_at`` cleared and ``next_attempt_at`` set to ``now`` so it
        is immediately claimable again. ``attempt`` is left as-is (it was incremented at claim);
        the dispatcher's retry budget (Q3) decides PARK-vs-requeue, not the reaper.
        """
        return await asyncio.to_thread(self._requeue_expired_leases_sync, now, lanes)

    def _requeue_expired_leases_sync(
        self, now: str | None, lanes: list[str] | None
    ) -> list[str]:
        """The locked reaper sweep over expired RUNNING leases."""
        now_iso = now or utc_now_iso()
        lane_clause = ""
        lane_params: tuple[Any, ...] = ()
        if lanes is not None:
            if not lanes:
                return []
            placeholders = ",".join("?" for _ in lanes)
            lane_clause = f" AND lane_key IN ({placeholders})"
            lane_params = tuple(lanes)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                doomed = self._conn.execute(
                    "SELECT run_id FROM runs WHERE status = ? AND "  # noqa: S608
                    "lease_expires_at IS NOT NULL AND lease_expires_at <= ?"
                    f"{lane_clause}",
                    (RunStatus.RUNNING.value, now_iso, *lane_params),
                ).fetchall()
                run_ids = [r["run_id"] for r in doomed]
                if run_ids:
                    self._conn.execute(
                        "UPDATE runs SET status = ?, owner_id = NULL, "  # noqa: S608
                        "lease_expires_at = NULL, next_attempt_at = ?, updated_at = ?, "
                        "payload = json_set(payload, '$.status', ?, '$.owner_id', NULL, "
                        "'$.lease_expires_at', NULL, '$.next_attempt_at', ?, "
                        "'$.updated_at', ?) "
                        "WHERE status = ? AND lease_expires_at IS NOT NULL "
                        f"AND lease_expires_at <= ?{lane_clause}",
                        (
                            RunStatus.QUEUED.value,
                            now_iso,
                            now_iso,
                            RunStatus.QUEUED.value,
                            now_iso,
                            now_iso,
                            RunStatus.RUNNING.value,
                            now_iso,
                            *lane_params,
                        ),
                    )
                self._conn.commit()
                return run_ids
            except BaseException:
                self._rollback_quietly()
                raise

    async def redrive_run(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        now: str | None = None,
    ) -> bool:
        """Reset a PARKED/FAILED run back to QUEUED for another attempt (True iff redriven).

        The operator/admin escape hatch: a run the queue gave up on (``PARKED``) — or a
        terminally ``FAILED`` one an operator wants to retry — is reset to ``QUEUED`` with
        ``attempt`` zeroed, ``last_error``/``error``/``owner_id``/``lease_expires_at`` cleared
        and ``next_attempt_at`` set to ``now`` so it is immediately claimable. Only ``PARKED``
        and ``FAILED`` are redrivable (a live RUNNING/QUEUED run is left alone). ``workspace_id``
        scopes it when supplied (tenant safety).
        """
        return await asyncio.to_thread(
            self._redrive_run_sync, run_id, workspace_id, now
        )

    def _redrive_run_sync(
        self, run_id: str, workspace_id: str | None, now: str | None
    ) -> bool:
        """The locked PARKED/FAILED -> QUEUED reset."""
        now_iso = now or utc_now_iso()
        ws_clause = " AND workspace_id = ?" if workspace_id is not None else ""
        ws_params = (workspace_id,) if workspace_id is not None else ()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cur = self._conn.execute(
                    "UPDATE runs SET status = ?, attempt = 0, owner_id = NULL, "  # noqa: S608
                    "lease_expires_at = NULL, last_error = NULL, next_attempt_at = ?, "
                    "updated_at = ?, payload = json_set(payload, '$.status', ?, "
                    "'$.attempt', 0, '$.owner_id', NULL, '$.lease_expires_at', NULL, "
                    "'$.last_error', NULL, '$.error', NULL, '$.next_attempt_at', ?, "
                    "'$.updated_at', ?) "
                    f"WHERE run_id = ?{ws_clause} AND status IN (?, ?)",
                    (
                        RunStatus.QUEUED.value,
                        now_iso,
                        now_iso,
                        RunStatus.QUEUED.value,
                        now_iso,
                        now_iso,
                        run_id,
                        *ws_params,
                        RunStatus.PARKED.value,
                        RunStatus.FAILED.value,
                    ),
                )
                won = cur.rowcount == 1
                self._conn.commit()
                return won
            except BaseException:
                self._rollback_quietly()
                raise

    # -------------------------------------------------- durable inbound dedup (Q4)
    async def dedup_try_claim(
        self,
        scope: str,
        key: str,
        *,
        lease_seconds: float,
        now: str | None = None,
    ) -> DedupClaim:
        """Atomically claim ``(scope, key)`` for execution, or report a duplicate (Q4).

        The durable TTL-CAS backing :class:`DurableIdempotencyStore`. Under ``BEGIN
        IMMEDIATE`` (write lock up front) it reads the existing row: a LIVE COMPLETED row
        (``result`` set, ``expires_at`` in the future) returns :data:`CLAIM_DONE` with the
        stored result; a LIVE in-flight row (``result`` NULL) returns :data:`CLAIM_IN_FLIGHT`;
        an absent or EXPIRED row (a crashed worker's lapsed lease, or a stale COMPLETED entry)
        is (re)claimed as a fresh in-flight lease and returns :data:`CLAIM_WON`. Two workers
        racing serialize on the write lock, so exactly one wins the claim.
        """
        return await asyncio.to_thread(
            self._dedup_try_claim_sync, scope, key, lease_seconds, now
        )

    def _dedup_try_claim_sync(
        self, scope: str, key: str, lease_seconds: float, now: str | None
    ) -> DedupClaim:
        """The locked TTL-CAS claim for the durable dedup ledger."""
        from himmy.services.storage.trigger_dedup import (
            CLAIM_DONE,
            CLAIM_IN_FLIGHT,
            CLAIM_WON,
            DedupClaim,
        )

        now_iso = now or utc_now_iso()
        lease_until = _iso_plus_seconds(now_iso, lease_seconds)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT result, expires_at FROM trigger_dedup "
                    "WHERE scope = ? AND key = ?",
                    (scope, key),
                ).fetchone()
                if row is not None:
                    expires = row["expires_at"]
                    live = expires is None or expires > now_iso
                    if live:
                        if row["result"] is not None:
                            self._conn.commit()
                            return DedupClaim(CLAIM_DONE, result=row["result"])
                        self._conn.commit()
                        return DedupClaim(CLAIM_IN_FLIGHT)
                # Absent or expired: (re)claim a fresh in-flight lease. INSERT OR REPLACE so a
                # lapsed row is overwritten under the same PK.
                self._conn.execute(
                    "INSERT OR REPLACE INTO trigger_dedup "
                    "(scope, key, result, expires_at, created_at) "
                    "VALUES (?, ?, NULL, ?, ?)",
                    (scope, key, lease_until, now_iso),
                )
                self._conn.commit()
                return DedupClaim(CLAIM_WON)
            except BaseException:
                self._rollback_quietly()
                raise

    async def dedup_complete(
        self,
        scope: str,
        key: str,
        *,
        result: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> None:
        """Upgrade a won in-flight dedup row to COMPLETED with ``result`` + the real TTL."""
        await asyncio.to_thread(
            self._dedup_complete_sync, scope, key, result, ttl_seconds, now
        )

    def _dedup_complete_sync(
        self, scope: str, key: str, result: str, ttl_seconds: float, now: str | None
    ) -> None:
        """The locked in-flight -> COMPLETED upgrade."""
        now_iso = now or utc_now_iso()
        expires_at = _iso_plus_seconds(now_iso, ttl_seconds)
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                self._conn.execute(
                    "INSERT OR REPLACE INTO trigger_dedup "
                    "(scope, key, result, expires_at, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (scope, key, result, expires_at, now_iso),
                )
                self._conn.commit()
            except BaseException:
                self._rollback_quietly()
                raise

    async def dedup_release(
        self, scope: str, key: str, *, now: str | None = None
    ) -> None:
        """Drop a won-but-failed in-flight dedup row so a redelivery re-runs (if in-flight)."""
        await asyncio.to_thread(self._dedup_release_sync, scope, key)

    def _dedup_release_sync(self, scope: str, key: str) -> None:
        """The locked release of an in-flight (not COMPLETED) dedup row."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                # Only delete an in-flight row (result IS NULL) — never a COMPLETED one, so a
                # release that races a completion can't erase a real dedup mark.
                self._conn.execute(
                    "DELETE FROM trigger_dedup "
                    "WHERE scope = ? AND key = ? AND result IS NULL",
                    (scope, key),
                )
                self._conn.commit()
            except BaseException:
                self._rollback_quietly()
                raise

    async def dedup_sweep(self, *, now: str | None = None) -> int:
        """Delete expired dedup rows (lazy GC); return the count removed."""
        return await asyncio.to_thread(self._dedup_sweep_sync, now)

    def _dedup_sweep_sync(self, now: str | None) -> int:
        """The locked expired-row prune."""
        now_iso = now or utc_now_iso()
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                cur = self._conn.execute(
                    "DELETE FROM trigger_dedup "
                    "WHERE expires_at IS NOT NULL AND expires_at <= ?",
                    (now_iso,),
                )
                removed = cur.rowcount
                self._conn.commit()
                return int(removed)
            except BaseException:
                self._rollback_quietly()
                raise

    def _run_upsert(self, run: RunRecord) -> tuple[str, tuple[Any, ...]]:
        """The upsert-by-run_id statement + params for a run record.

        The full record JSON rides ``payload`` (the canonical record); the leased-queue
        columns (owner_id … lane_key) are ALSO promoted to indexed columns so the Q2 claim
        CAS + the reaper seek them rather than json_extract-scanning. The Q0 ``input_blob``
        is encrypted at rest (when a cipher is configured) INSIDE the serialized payload,
        bound to the ``run_id`` as AAD — the recoverable launch input is sensitive (it carries
        the prompt), so it gets the same envelope as chat content / event payloads.
        """
        payload = run.model_dump(mode="json")
        if self._cipher is not None and run.input_blob:
            payload = dict(payload)
            payload["input_blob"] = self._cipher.encrypt_run_input(
                run.input_blob, run_id=run.run_id
            )
        sql = (
            "INSERT INTO runs (run_id, workspace_id, subject_id, idempotency_key, "
            "status, payload, created_at, updated_at, owner_id, lease_expires_at, "
            "heartbeat_at, attempt, max_attempts, next_attempt_at, last_error, lane_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (run_id) DO UPDATE SET "
            "workspace_id = excluded.workspace_id, subject_id = excluded.subject_id, "
            "idempotency_key = excluded.idempotency_key, status = excluded.status, "
            "payload = excluded.payload, updated_at = excluded.updated_at, "
            "owner_id = excluded.owner_id, lease_expires_at = excluded.lease_expires_at, "
            "heartbeat_at = excluded.heartbeat_at, attempt = excluded.attempt, "
            "max_attempts = excluded.max_attempts, "
            "next_attempt_at = excluded.next_attempt_at, "
            "last_error = excluded.last_error, lane_key = excluded.lane_key"
        )
        params = (
            run.run_id,
            run.workspace_id,
            run.subject_id,
            run.idempotency_key,
            run.status.value,
            json.dumps(payload),
            run.created_at,
            run.updated_at,
            run.owner_id,
            run.lease_expires_at,
            run.heartbeat_at,
            run.attempt,
            run.max_attempts,
            run.next_attempt_at or run.created_at,
            run.last_error,
            run.lane_key,
        )
        return sql, params

    def _row_to_run(self, row: sqlite3.Row) -> RunRecord:
        """Reconstruct a :class:`RunRecord` from a row's JSON ``payload`` (decrypt input).

        Mirror of the module-level :func:`_row_to_run` but cipher-aware: it decrypts the Q0
        ``input_blob`` envelope back to cleartext on read (idempotent on plaintext / legacy
        rows). Used wherever the cipher might be engaged.
        """
        payload = json.loads(row["payload"])
        if self._cipher is not None and payload.get("input_blob"):
            payload = dict(payload)
            payload["input_blob"] = self._cipher.decrypt_run_input(
                payload["input_blob"], run_id=str(payload.get("run_id", ""))
            )
        return RunRecord.model_validate(payload)

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run record by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone, "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
        )
        return self._row_to_run(row) if row else None

    async def list_runs(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[RunRecord]:
        """List runs filtered by workspace, subject, and/or status."""
        clauses: list[str] = []
        params: list[Any] = []
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        if subject_id is not None:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await asyncio.to_thread(
            self._fetchall,
            f"SELECT payload FROM runs{where} ORDER BY created_at ASC",  # noqa: S608
            tuple(params),
        )
        return [self._row_to_run(r) for r in rows]

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None:
        """Return the existing run for an idempotency key, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM runs WHERE workspace_id = ? AND idempotency_key = ?",
            (workspace_id, idempotency_key),
        )
        return self._row_to_run(row) if row else None

    # ------------------------------------------------------------- agent defs (T2e)
    async def save_agent_def(self, record: AgentDefRecord) -> AgentDefRecord:
        """Upsert a stored agent definition keyed by ``agent_id``."""
        record.updated_at = utc_now_iso()
        await asyncio.to_thread(self._write, *self._agent_def_upsert(record))
        return record

    async def save_agent_def_if_absent(
        self, record: AgentDefRecord
    ) -> tuple[AgentDefRecord, bool]:
        """Atomically create an agent def unless its idempotency key already exists.

        Mirrors :meth:`save_run_if_absent_by_idempotency`: the write lock +
        ``agent_defs_idempotency_idx`` partial UNIQUE index make the existence-check
        and the insert atomic across processes sharing the file.
        """
        record.updated_at = utc_now_iso()
        return await asyncio.to_thread(self._save_agent_def_if_absent_sync, record)

    def _save_agent_def_if_absent_sync(
        self, record: AgentDefRecord
    ) -> tuple[AgentDefRecord, bool]:
        """The locked read-then-insert for the idempotent agent-def create."""
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                if record.idempotency_key is not None:
                    existing = self._conn.execute(
                        "SELECT payload FROM agent_defs WHERE workspace_id = ? AND "
                        "idempotency_key = ?",
                        (record.workspace_id, record.idempotency_key),
                    ).fetchone()
                    if existing is not None:
                        self._conn.commit()
                        return _row_to_agent_def(existing), False
                sql, params = self._agent_def_upsert(record)
                self._conn.execute(sql, params)
                self._conn.commit()
                return record, True
            except BaseException:
                self._rollback_quietly()
                raise

    @staticmethod
    def _agent_def_upsert(record: AgentDefRecord) -> tuple[str, tuple[Any, ...]]:
        """The upsert-by-agent_id statement + params for a stored agent definition."""
        sql = (
            "INSERT INTO agent_defs (agent_id, workspace_id, name, idempotency_key, "
            "payload, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (agent_id) DO UPDATE SET "
            "workspace_id = excluded.workspace_id, name = excluded.name, "
            "idempotency_key = excluded.idempotency_key, payload = excluded.payload, "
            "updated_at = excluded.updated_at"
        )
        params = (
            record.agent_id,
            record.workspace_id,
            record.name,
            record.idempotency_key,
            json.dumps(record.model_dump(mode="json")),
            record.created_at,
            record.updated_at,
        )
        return sql, params

    async def get_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> AgentDefRecord | None:
        """Return a stored agent def by id, tenant-scoped (out-of-workspace → None)."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM agent_defs WHERE agent_id = ?",
            (agent_id,),
        )
        if not row:
            return None
        rec = _row_to_agent_def(row)
        if workspace_id is not None and rec.workspace_id != workspace_id:
            return None
        return rec

    async def list_agent_defs(
        self, *, workspace_id: str | None = None
    ) -> list[AgentDefRecord]:
        """List stored agent defs for a workspace (created_at asc)."""
        if workspace_id is None:
            sql = "SELECT payload FROM agent_defs ORDER BY created_at ASC"
            params: tuple[Any, ...] = ()
        else:
            sql = (
                "SELECT payload FROM agent_defs WHERE workspace_id = ? "
                "ORDER BY created_at ASC"
            )
            params = (workspace_id,)
        rows = await asyncio.to_thread(self._fetchall, sql, params)
        return [_row_to_agent_def(r) for r in rows]

    async def delete_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> bool:
        """Delete a stored agent def, tenant-scoped. Returns True iff a row was removed."""
        return await asyncio.to_thread(
            self._delete_agent_def_sync, agent_id, workspace_id
        )

    def _delete_agent_def_sync(
        self, agent_id: str, workspace_id: str | None
    ) -> bool:
        """Locked tenant-scoped delete; returns whether a row was actually removed."""
        with self._lock:
            try:
                if workspace_id is None:
                    cur = self._conn.execute(
                        "DELETE FROM agent_defs WHERE agent_id = ?", (agent_id,)
                    )
                else:
                    cur = self._conn.execute(
                        "DELETE FROM agent_defs WHERE agent_id = ? AND workspace_id = ?",
                        (agent_id, workspace_id),
                    )
                self._conn.commit()
                return cur.rowcount > 0
            except BaseException:
                self._rollback_quietly()
                raise

    # ---------------------------------------------------------- recommendations
    async def save_recommendation(self, item: RecommendationItem) -> RecommendationItem:
        """Upsert a recommendation item keyed by ``recommendation_id``."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO recommendations (recommendation_id, run_id, workspace_id, "
            "subject_id, kind, status, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (recommendation_id) DO UPDATE SET "
            "run_id = excluded.run_id, workspace_id = excluded.workspace_id, "
            "subject_id = excluded.subject_id, kind = excluded.kind, "
            "status = excluded.status, payload = excluded.payload",
            (
                item.recommendation_id,
                item.run_id,
                item.workspace_id,
                item.subject_id,
                item.kind,
                item.status.value,
                self._dump(item),
                item.created_at,
            ),
        )
        return item

    async def get_recommendation(
        self, recommendation_id: str
    ) -> RecommendationItem | None:
        """Return a recommendation by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM recommendations WHERE recommendation_id = ?",
            (recommendation_id,),
        )
        return _row_to_recommendation(row) if row else None

    async def list_recommendations(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        status: RecommendationStatus | None = None,
    ) -> list[RecommendationItem]:
        """List recommendations filtered by the given dimensions."""
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("workspace_id", workspace_id),
            ("subject_id", subject_id),
            ("run_id", run_id),
            ("kind", kind),
            ("status", status.value if status is not None else None),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await asyncio.to_thread(
            self._fetchall,
            f"SELECT payload FROM recommendations{where} ORDER BY created_at ASC",  # noqa: S608
            tuple(params),
        )
        return [_row_to_recommendation(r) for r in rows]

    async def update_recommendation(
        self,
        recommendation_id: str,
        *,
        status: RecommendationStatus | None = None,
        notes: str | None = None,
    ) -> RecommendationItem | None:
        """Update a recommendation's status/notes in place; return it or None."""
        return await asyncio.to_thread(
            self._update_recommendation_sync,
            recommendation_id,
            status,
            notes,
        )

    def _update_recommendation_sync(
        self,
        recommendation_id: str,
        status: RecommendationStatus | None,
        notes: str | None,
    ) -> RecommendationItem | None:
        """Read-modify-write a recommendation's status/notes under the write lock.

        The read and the write run in one ``BEGIN IMMEDIATE`` transaction so the
        update is atomic even across processes sharing the file (no torn update over
        a concurrent writer), and rolls back on failure so an aborted update cannot
        leave the transaction open.
        """
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                row = self._conn.execute(
                    "SELECT payload FROM recommendations WHERE recommendation_id = ?",
                    (recommendation_id,),
                ).fetchone()
                if row is None:
                    self._conn.rollback()
                    return None
                item = _row_to_recommendation(row)
                if status is not None:
                    item.status = status
                if notes is not None:
                    item.notes = notes
                self._conn.execute(
                    "UPDATE recommendations SET status = ?, payload = ? "
                    "WHERE recommendation_id = ?",
                    (item.status.value, self._dump(item), recommendation_id),
                )
                self._conn.commit()
                return item
            except BaseException:
                self._rollback_quietly()
                raise

    # --------------------------------------------------------------- evaluation
    async def save_evaluation_run(self, run: EvaluationRun) -> EvaluationRun:
        """Upsert an evaluation run keyed by ``run_id`` (stamping its workspace)."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO evaluation_runs "
            "(run_id, suite_id, workspace_id, payload, created_at) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (run_id) DO UPDATE SET "
            "suite_id = excluded.suite_id, "
            "workspace_id = excluded.workspace_id, payload = excluded.payload",
            (
                run.run_id,
                getattr(run, "suite_id", None),
                getattr(run, "workspace_id", None),
                self._dump(run),
                str(getattr(run, "created_at", None) or utc_now_iso()),
            ),
        )
        return run

    async def get_evaluation_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> EvaluationRun | None:
        """Return an evaluation run by id, tenant-scoped (AAEO-4).

        When ``workspace_id`` is supplied, a run belonging to another workspace is
        treated as not found (returns None) — closing the cross-tenant IDOR.
        """
        from himmy.services.evaluation.models import EvaluationRun

        if workspace_id is None:
            sql = "SELECT payload FROM evaluation_runs WHERE run_id = ?"
            params: tuple[Any, ...] = (run_id,)
        else:
            sql = (
                "SELECT payload FROM evaluation_runs "
                "WHERE run_id = ? AND workspace_id = ?"
            )
            params = (run_id, workspace_id)
        row = await asyncio.to_thread(self._fetchone, sql, params)
        return EvaluationRun.model_validate(json.loads(row["payload"])) if row else None

    async def list_evaluation_runs(
        self, suite_id: str | None = None, *, workspace_id: str | None = None
    ) -> list[EvaluationRun]:
        """List evaluation runs, optionally filtered by suite id and workspace (AAEO-4)."""
        from himmy.services.evaluation.models import EvaluationRun

        clauses: list[str] = []
        params: list[Any] = []
        if suite_id is not None:
            clauses.append("suite_id = ?")
            params.append(suite_id)
        if workspace_id is not None:
            clauses.append("workspace_id = ?")
            params.append(workspace_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = await asyncio.to_thread(
            self._fetchall,
            # clauses are constant column literals; values are bound params.
            f"SELECT payload FROM evaluation_runs{where} ORDER BY created_at ASC",  # noqa: S608
            tuple(params),
        )
        return [EvaluationRun.model_validate(json.loads(r["payload"])) for r in rows]

    # --------------------------------------------- memory + orchestration records
    async def save_memory(self, obj: MemoryObject) -> MemoryObject:
        """Upsert a cognitive memory object."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO memory_objects (memory_id, subject_id, payload, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (memory_id) DO UPDATE SET "
            "subject_id = excluded.subject_id, payload = excluded.payload",
            (obj.memory_id, obj.subject_id, self._dump(obj), obj.created_at),
        )
        return obj

    async def get_memory(self, memory_id: str) -> MemoryObject | None:
        """Return a memory object by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM memory_objects WHERE memory_id = ?",
            (memory_id,),
        )
        return MemoryObject.model_validate(json.loads(row["payload"])) if row else None

    async def list_memory(self, subject_id: str | None = None) -> list[MemoryObject]:
        """List memory objects, optionally filtered by subject."""
        rows = await self._list_by_subject("memory_objects", subject_id)
        return [MemoryObject.model_validate(json.loads(r["payload"])) for r in rows]

    async def save_episodic_memory(
        self, obj: EpisodicMemoryObject
    ) -> EpisodicMemoryObject:
        """Upsert an episodic memory object."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO episodic_memory_objects (episode_id, subject_id, payload, "
            "created_at) VALUES (?, ?, ?, ?) ON CONFLICT (episode_id) DO UPDATE SET "
            "subject_id = excluded.subject_id, payload = excluded.payload",
            (obj.episode_id, obj.subject_id, self._dump(obj), obj.created_at),
        )
        return obj

    async def get_episodic_memory(self, episode_id: str) -> EpisodicMemoryObject | None:
        """Return an episodic memory object by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM episodic_memory_objects WHERE episode_id = ?",
            (episode_id,),
        )
        return (
            EpisodicMemoryObject.model_validate(json.loads(row["payload"]))
            if row
            else None
        )

    async def list_episodic_memory(
        self, subject_id: str | None = None
    ) -> list[EpisodicMemoryObject]:
        """List episodic memory objects, optionally filtered by subject."""
        rows = await self._list_by_subject("episodic_memory_objects", subject_id)
        return [
            EpisodicMemoryObject.model_validate(json.loads(r["payload"])) for r in rows
        ]

    async def save_agent_state(self, record: AgentStateRecord) -> AgentStateRecord:
        """Upsert an agent state record."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO agent_states (state_id, environment_name, round, payload, "
            "created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT (state_id) DO UPDATE SET "
            "environment_name = excluded.environment_name, round = excluded.round, "
            "payload = excluded.payload",
            (
                record.state_id,
                record.environment_name,
                record.round,
                self._dump(record),
                record.created_at,
            ),
        )
        return record

    async def get_agent_state(self, state_id: str) -> AgentStateRecord | None:
        """Return an agent state record by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM agent_states WHERE state_id = ?",
            (state_id,),
        )
        return (
            AgentStateRecord.model_validate(json.loads(row["payload"])) if row else None
        )

    async def list_agent_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[AgentStateRecord]:
        """List agent state records filtered by environment and/or round."""
        rows = await self._list_by_env_round("agent_states", environment_name, round)
        return [AgentStateRecord.model_validate(json.loads(r["payload"])) for r in rows]

    async def save_action(self, record: ActionRecord) -> ActionRecord:
        """Upsert an action record."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO actions (action_id, environment_name, round, payload, "
            "created_at) VALUES (?, ?, ?, ?, ?) ON CONFLICT (action_id) DO UPDATE SET "
            "environment_name = excluded.environment_name, round = excluded.round, "
            "payload = excluded.payload",
            (
                record.action_id,
                record.environment_name,
                record.round,
                self._dump(record),
                record.created_at,
            ),
        )
        return record

    async def get_action(self, action_id: str) -> ActionRecord | None:
        """Return an action record by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM actions WHERE action_id = ?",
            (action_id,),
        )
        return ActionRecord.model_validate(json.loads(row["payload"])) if row else None

    async def list_actions(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[ActionRecord]:
        """List action records filtered by environment and/or round."""
        rows = await self._list_by_env_round("actions", environment_name, round)
        return [ActionRecord.model_validate(json.loads(r["payload"])) for r in rows]

    async def save_environment_state(
        self, record: EnvironmentStateRecord
    ) -> EnvironmentStateRecord:
        """Upsert an environment state record."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO environment_states (environment_state_id, environment_name, "
            "round, payload, created_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (environment_state_id) DO UPDATE SET "
            "environment_name = excluded.environment_name, round = excluded.round, "
            "payload = excluded.payload",
            (
                record.environment_state_id,
                record.environment_name,
                record.round,
                self._dump(record),
                record.created_at,
            ),
        )
        return record

    async def get_environment_state(
        self, environment_state_id: str
    ) -> EnvironmentStateRecord | None:
        """Return an environment state record by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM environment_states WHERE environment_state_id = ?",
            (environment_state_id,),
        )
        return (
            EnvironmentStateRecord.model_validate(json.loads(row["payload"]))
            if row
            else None
        )

    async def list_environment_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[EnvironmentStateRecord]:
        """List environment state records filtered by environment and/or round."""
        rows = await self._list_by_env_round(
            "environment_states", environment_name, round
        )
        return [
            EnvironmentStateRecord.model_validate(json.loads(r["payload"]))
            for r in rows
        ]

    # ----------------------------------------------------------- generic listers
    async def _list_by_subject(
        self, table: str, subject_id: str | None
    ) -> list[sqlite3.Row]:
        """List rows optionally filtered by ``subject_id``, ordered by created_at."""
        if subject_id is None:
            sql = f"SELECT payload FROM {table} ORDER BY created_at ASC"  # noqa: S608
            params: tuple[Any, ...] = ()
        else:
            sql = (
                f"SELECT payload FROM {table} WHERE subject_id = ? "  # noqa: S608
                "ORDER BY created_at ASC"
            )
            params = (subject_id,)
        return await asyncio.to_thread(self._fetchall, sql, params)

    async def _list_by_env_round(
        self, table: str, environment_name: str | None, round: int | None
    ) -> list[sqlite3.Row]:
        """List rows filtered by ``environment_name`` and/or ``round``."""
        clauses: list[str] = []
        params: list[Any] = []
        if environment_name is not None:
            clauses.append("environment_name = ?")
            params.append(environment_name)
        if round is not None:
            clauses.append("round = ?")
            params.append(round)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT payload FROM {table}{where} ORDER BY created_at ASC"  # noqa: S608
        return await asyncio.to_thread(self._fetchall, sql, tuple(params))


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    """Reconstruct a :class:`RunRecord` from a row's JSON ``payload``."""
    return RunRecord.model_validate(json.loads(row["payload"]))


def _row_to_agent_def(row: sqlite3.Row) -> AgentDefRecord:
    """Reconstruct an :class:`AgentDefRecord` from a row's JSON ``payload``."""
    return AgentDefRecord.model_validate(json.loads(row["payload"]))


def _row_to_recommendation(row: sqlite3.Row) -> RecommendationItem:
    """Reconstruct a :class:`RecommendationItem` from a row's JSON ``payload``."""
    return RecommendationItem.model_validate(json.loads(row["payload"]))


__all__ = ["SqliteStorageService", "SQLITE_SCHEMA_VERSION"]
