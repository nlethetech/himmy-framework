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
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from himmy.core.events import RunEvent
from himmy.core.ids import utc_now_iso
from himmy.core.sqlite_util import connect_hardened
from himmy.services.storage.at_rest import StorePayloadCipher, build_store_cipher
from himmy.services.storage.models import (
    ActionRecord,
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
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()
        self._cipher = build_store_cipher() if isinstance(cipher, _Unset) else cipher

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
        await asyncio.to_thread(
            self._write,
            "INSERT INTO run_events (event_id, thread_id, trace_id, payload) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (event_id) DO NOTHING",
            (
                event.event_id,
                event.thread_id,
                event.trace_id,
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
                        return _row_to_run(existing), False
                sql, params = self._run_upsert(run)
                self._conn.execute(sql, params)
                self._conn.commit()
                return run, True
            except BaseException:
                self._rollback_quietly()
                raise

    @staticmethod
    def _run_upsert(run: RunRecord) -> tuple[str, tuple[Any, ...]]:
        """The upsert-by-run_id statement + params for a run record."""
        sql = (
            "INSERT INTO runs (run_id, workspace_id, subject_id, idempotency_key, "
            "status, payload, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (run_id) DO UPDATE SET "
            "workspace_id = excluded.workspace_id, subject_id = excluded.subject_id, "
            "idempotency_key = excluded.idempotency_key, status = excluded.status, "
            "payload = excluded.payload, updated_at = excluded.updated_at"
        )
        params = (
            run.run_id,
            run.workspace_id,
            run.subject_id,
            run.idempotency_key,
            run.status.value,
            json.dumps(run.model_dump(mode="json")),
            run.created_at,
            run.updated_at,
        )
        return sql, params

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run record by id, or None."""
        row = await asyncio.to_thread(
            self._fetchone, "SELECT payload FROM runs WHERE run_id = ?", (run_id,)
        )
        return _row_to_run(row) if row else None

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
        return [_row_to_run(r) for r in rows]

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None:
        """Return the existing run for an idempotency key, or None."""
        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM runs WHERE workspace_id = ? AND idempotency_key = ?",
            (workspace_id, idempotency_key),
        )
        return _row_to_run(row) if row else None

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
        """Upsert an evaluation run keyed by ``run_id``."""
        await asyncio.to_thread(
            self._write,
            "INSERT INTO evaluation_runs (run_id, suite_id, payload, created_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT (run_id) DO UPDATE SET "
            "suite_id = excluded.suite_id, payload = excluded.payload",
            (
                run.run_id,
                getattr(run, "suite_id", None),
                self._dump(run),
                str(getattr(run, "created_at", None) or utc_now_iso()),
            ),
        )
        return run

    async def get_evaluation_run(self, run_id: str) -> EvaluationRun | None:
        """Return an evaluation run by id, or None."""
        from himmy.services.evaluation.models import EvaluationRun

        row = await asyncio.to_thread(
            self._fetchone,
            "SELECT payload FROM evaluation_runs WHERE run_id = ?",
            (run_id,),
        )
        return EvaluationRun.model_validate(json.loads(row["payload"])) if row else None

    async def list_evaluation_runs(
        self, suite_id: str | None = None
    ) -> list[EvaluationRun]:
        """List evaluation runs, optionally filtered by suite id."""
        from himmy.services.evaluation.models import EvaluationRun

        if suite_id is None:
            sql = "SELECT payload FROM evaluation_runs ORDER BY created_at ASC"
            params: tuple[Any, ...] = ()
        else:
            sql = (
                "SELECT payload FROM evaluation_runs WHERE suite_id = ? "
                "ORDER BY created_at ASC"
            )
            params = (suite_id,)
        rows = await asyncio.to_thread(self._fetchall, sql, params)
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


def _row_to_recommendation(row: sqlite3.Row) -> RecommendationItem:
    """Reconstruct a :class:`RecommendationItem` from a row's JSON ``payload``."""
    return RecommendationItem.model_validate(json.loads(row["payload"]))


__all__ = ["SqliteStorageService"]
