"""Storage kernel: the Postgres-backed StorageService (requires the [postgres] extra).

This module is import-safe without ``asyncpg`` or a running database. ``asyncpg`` is
imported lazily inside :meth:`PostgresStorageService.connect`; the DDL is a plain
string so it can be inspected offline. The data methods mirror the in-memory
:class:`~himmy.services.storage.service.StorageService` surface 1:1 and are a
true full mirror — upserts via ``INSERT ... ON CONFLICT``, reads via ``SELECT``,
and filtered lists via ``WHERE`` — gated on a live pool via ``_require_pool()``.

Production hardening (see IMPROVEMENTS SE-1/4/5/6/10/11):

- A JSONB type codec is registered on every pooled connection (``init`` callback)
  so dict/list payloads are bound and read back as native Python objects.
- ``close()`` / async-context-manager support and connection-acquire/command
  timeouts are exposed on :meth:`connect`.
- A lightweight versioned migration runner (``schema_migrations`` table + an
  ordered statement list) lets the schema evolve beyond the idempotent base DDL.
- ``runs`` upserts close the idempotency race at the DB
  (``save_run_if_absent_by_idempotency``) and always refresh ``updated_at``.
- Timestamp columns that drive time-windowed analytics are ``TIMESTAMPTZ`` with
  indexes, while the model boundary keeps ISO strings.
- The ``ai_call_log`` view flattens the request/response event pair into one row
  per LLM call. See the payload-key contract in :mod:`himmy.core.events`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

from himmy.core.errors import HimmyError
from himmy.core.events import RunEvent
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


class _Unset:
    """Sentinel: ``cipher`` was not passed, so source it from the environment.

    Distinct from an explicit ``cipher=None`` (which forces plaintext regardless of
    ``HIMMY_ENCRYPTION_KEY``) so a caller can opt out even when a key is configured.
    """


_UNSET = _Unset()

#: Idempotent schema for the full storage surface. 14 tables + the ai_call_log view.
#: Timestamps that drive analytics are TIMESTAMPTZ with indexes; the model boundary
#: keeps ISO strings (asyncpg accepts ISO strings for TIMESTAMPTZ binds and returns
#: ``datetime`` objects, which the row mappers normalise back to ISO).
STORAGE_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version    INTEGER PRIMARY KEY,
    name       TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Threads store their Message list inline in ``payload`` (mirrors the in-memory
-- backend), so a separate messages table is not needed.
CREATE TABLE IF NOT EXISTS chat_threads (
    thread_id  TEXT PRIMARY KEY,
    agent_id   TEXT,
    version    INTEGER NOT NULL DEFAULT 1,
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_threads_updated_at_idx ON chat_threads (updated_at);

CREATE TABLE IF NOT EXISTS run_events (
    event_id     TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    trace_id     TEXT,
    thread_id    TEXT,
    agent_id     TEXT,
    request_id   TEXT,
    tool_call_id TEXT,
    latency_ms   DOUBLE PRECISION,
    cost         DOUBLE PRECISION,
    payload      JSONB NOT NULL DEFAULT '{}'::jsonb,
    error        TEXT,
    timestamp    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS run_events_thread_id_idx ON run_events (thread_id);
CREATE INDEX IF NOT EXISTS run_events_trace_id_idx ON run_events (trace_id);
CREATE INDEX IF NOT EXISTS run_events_request_id_idx ON run_events (request_id);
CREATE INDEX IF NOT EXISTS run_events_timestamp_idx ON run_events (timestamp);

CREATE TABLE IF NOT EXISTS context_fields (
    subject_id TEXT NOT NULL,
    key        TEXT NOT NULL,
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (subject_id, key)
);

CREATE TABLE IF NOT EXISTS context_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    subject_id  TEXT NOT NULL,
    task_id     TEXT,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS context_snapshots_subject_id_idx
    ON context_snapshots (subject_id);

CREATE TABLE IF NOT EXISTS context_evidence (
    evidence_id TEXT PRIMARY KEY,
    subject_id  TEXT,
    snapshot_id TEXT,
    key         TEXT,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata    JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS context_evidence_snapshot_id_idx
    ON context_evidence (snapshot_id);

CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    workspace_id     TEXT NOT NULL,
    subject_id       TEXT NOT NULL,
    task_id          TEXT,
    thread_id        TEXT,
    snapshot_id      TEXT,
    persona_name     TEXT,
    model_key        TEXT,
    idempotency_key  TEXT,
    status           TEXT NOT NULL,
    output_text      TEXT,
    output_structured JSONB,
    error            TEXT,
    trace_id         TEXT,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS runs_workspace_id_idx ON runs (workspace_id);
CREATE INDEX IF NOT EXISTS runs_subject_id_idx ON runs (subject_id);
CREATE INDEX IF NOT EXISTS runs_status_idx ON runs (status);
CREATE INDEX IF NOT EXISTS runs_updated_at_idx ON runs (updated_at);
CREATE UNIQUE INDEX IF NOT EXISTS runs_idempotency_idx
    ON runs (workspace_id, idempotency_key)
    WHERE idempotency_key IS NOT NULL;

CREATE TABLE IF NOT EXISTS recommendations (
    recommendation_id TEXT PRIMARY KEY,
    run_id            TEXT NOT NULL,
    workspace_id      TEXT NOT NULL,
    subject_id        TEXT NOT NULL,
    kind              TEXT NOT NULL,
    title             TEXT NOT NULL,
    summary           TEXT,
    rationale         TEXT,
    confidence        DOUBLE PRECISION NOT NULL DEFAULT 0,
    evidence_refs     JSONB NOT NULL DEFAULT '[]'::jsonb,
    status            TEXT NOT NULL,
    notes             TEXT,
    metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS recommendations_workspace_id_idx
    ON recommendations (workspace_id);
CREATE INDEX IF NOT EXISTS recommendations_subject_id_idx
    ON recommendations (subject_id);
CREATE INDEX IF NOT EXISTS recommendations_run_id_idx ON recommendations (run_id);
CREATE INDEX IF NOT EXISTS recommendations_status_idx ON recommendations (status);

CREATE TABLE IF NOT EXISTS evaluation_suites (
    suite_id   TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id          TEXT PRIMARY KEY,
    suite_id        TEXT,
    suite_name      TEXT,
    aggregate_score DOUBLE PRECISION NOT NULL DEFAULT 0,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS evaluation_runs_suite_id_idx
    ON evaluation_runs (suite_id);

CREATE TABLE IF NOT EXISTS memory_objects (
    memory_id  TEXT PRIMARY KEY,
    subject_id TEXT,
    agent_id   TEXT,
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS memory_objects_subject_id_idx
    ON memory_objects (subject_id);

CREATE TABLE IF NOT EXISTS episodic_memory_objects (
    episode_id TEXT PRIMARY KEY,
    subject_id TEXT,
    agent_id   TEXT,
    payload    JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS episodic_memory_objects_subject_id_idx
    ON episodic_memory_objects (subject_id);

CREATE TABLE IF NOT EXISTS agent_states (
    state_id         TEXT PRIMARY KEY,
    agent_id         TEXT,
    environment_name TEXT,
    round            INTEGER,
    payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_states_env_round_idx
    ON agent_states (environment_name, round);

CREATE TABLE IF NOT EXISTS actions (
    action_id        TEXT PRIMARY KEY,
    agent_id         TEXT,
    environment_name TEXT,
    round            INTEGER,
    payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS actions_env_round_idx
    ON actions (environment_name, round);

CREATE TABLE IF NOT EXISTS environment_states (
    environment_state_id TEXT PRIMARY KEY,
    environment_name     TEXT,
    round                INTEGER,
    payload              JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata             JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS environment_states_env_round_idx
    ON environment_states (environment_name, round);

-- The SQL-native AI trace surface: one row per LLM call, joining the
-- INFERENCE_REQUESTED / INFERENCE_SUCCEEDED|FAILED event pair by request_id.
-- ``model_name`` coalesces the response payload's model_name/model_path keys so
-- the runtime (which emits ``model_path``) and any model_name-emitting caller both
-- populate the column. See himmy.core.events for the payload-key contract.
CREATE OR REPLACE VIEW ai_call_log AS
SELECT
    req.request_id                                   AS request_id,
    req.trace_id                                     AS trace_id,
    req.thread_id                                    AS thread_id,
    req.agent_id                                     AS agent_id,
    req.payload ->> 'prompt_template'                AS prompt_template,
    req.payload ->> 'prompt_version'                 AS prompt_version,
    req.payload ->> 'rendered_prompt'                AS rendered_prompt,
    req.payload ->> 'retrieval_ctx'                  AS retrieval_ctx,
    req.payload ->> 'snapshot_id'                    AS snapshot_id,
    COALESCE(resp.payload ->> 'model_name',
             resp.payload ->> 'model_path')          AS model_name,
    (resp.payload ->> 'input_tokens')::int          AS input_tokens,
    (resp.payload ->> 'output_tokens')::int         AS output_tokens,
    resp.latency_ms                                  AS latency_ms,
    resp.cost                                        AS cost_usd,
    req.timestamp                                    AS requested_at,
    CASE
        WHEN resp.event_type = 'INFERENCE_SUCCEEDED' THEN 'success'
        WHEN resp.event_type = 'INFERENCE_FAILED'    THEN 'failed'
        ELSE 'pending'
    END                                              AS status
FROM run_events req
LEFT JOIN run_events resp
    ON resp.request_id = req.request_id
    AND resp.event_type IN ('INFERENCE_SUCCEEDED', 'INFERENCE_FAILED')
WHERE req.event_type = 'INFERENCE_REQUESTED';
"""


#: Ordered forward migrations applied after the base DDL. Each entry is
#: ``(version, name, [statements...])`` and is applied exactly once (tracked in
#: ``schema_migrations``). Add new entries here to evolve the schema beyond the
#: idempotent base DDL (new columns, indexes, tables) without manual ALTERs.
STORAGE_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (
        1,
        "base_schema",
        [STORAGE_DDL],
    ),
    # v2 (T1.1): promote workspace_id to an indexed column on evaluation_runs so
    # GET /v1/evaluation/runs/{id} can be tenant-scoped (close the cross-tenant
    # IDOR) and lists can filter by workspace. New writes populate the column
    # directly; existing rows are backfilled from the stored JSONB payload. Added
    # via a migration (not the frozen base DDL) so a fresh database lays down the
    # base then runs this ALTER exactly once, never double-applying it.
    (
        2,
        "evaluation_runs_workspace_id",
        [
            "ALTER TABLE evaluation_runs "
            "ADD COLUMN IF NOT EXISTS workspace_id TEXT",
            "CREATE INDEX IF NOT EXISTS evaluation_runs_workspace_id_idx "
            "ON evaluation_runs (workspace_id)",
            "UPDATE evaluation_runs SET workspace_id = payload->>'workspace_id' "
            "WHERE workspace_id IS NULL",
        ],
    ),
]


def _migration_advisory_lock_key() -> int:
    """Derive the stable signed-64-bit advisory-lock key for the migration runner."""
    import hashlib

    digest = hashlib.sha256(b"himmy.services.storage.schema_migrations").digest()[:8]
    value = int.from_bytes(digest, "big", signed=False)
    # pg_advisory_lock takes a signed bigint; fold into that range.
    return value - (1 << 63)


#: Session-level advisory-lock key that serialises concurrent ``migrate()`` callers
#: across processes (same derivation style as the entity-registry writer lock).
MIGRATION_ADVISORY_LOCK_KEY = _migration_advisory_lock_key()


def _require_asyncpg() -> Any:
    """Import asyncpg lazily, raising a clear error when the extra is missing."""
    try:
        import asyncpg  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise HimmyError(
            "PostgresStorageService requires the [postgres] extra "
            "(pip install 'himmy[postgres]') and a running database."
        ) from exc
    return asyncpg


def _ts_encode(value: Any) -> Any:
    """Encode a model timestamp for a timestamp/timestamptz column (text format).

    The model boundary keeps timestamps as ISO-8601 strings; Postgres parses the
    ISO text on input. Datetime instances are accepted too (isoformat()).
    """
    if value is None or isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _ts_decode(value: Any) -> Any:
    """Normalise Postgres' timestamp text (``'... 12:00:00+00'``) back to ISO-8601."""
    if not value:
        return value
    try:
        return datetime.fromisoformat(value).isoformat()
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return value


async def _register_codecs(conn: Any) -> None:
    """Connection ``init`` callback: JSONB and ISO-string timestamp codecs.

    JSONB: without this, asyncpg binds ``json.dumps`` output as a quoted JSON
    *string* literal (so ``{}`` round-trips as the text ``"{}"``); the codec lets
    callers bind dicts/lists directly and read them back parsed.

    Timestamps: the model boundary keeps timestamps as ISO-8601 *strings*, but
    asyncpg's default ``timestamp``/``timestamptz`` codec rejects strings (it
    expects ``datetime`` instances). A text codec lets Postgres parse the ISO
    string on input, and the decoder normalises Postgres' text output back to
    canonical ISO-8601 — preserving the "timestamps are ISO strings" contract
    end to end.
    """
    await conn.set_type_codec(
        "jsonb",
        encoder=json.dumps,
        decoder=json.loads,
        schema="pg_catalog",
    )
    for _ts_type in ("timestamptz", "timestamp"):
        await conn.set_type_codec(
            _ts_type,
            encoder=_ts_encode,
            decoder=_ts_decode,
            schema="pg_catalog",
            format="text",
        )


def _iso(value: Any) -> str | None:
    """Normalise a TIMESTAMPTZ value (datetime or str) back to an ISO string."""
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    if callable(isoformat):
        return cast(str, isoformat())
    return str(value)


# ----------------------------------------------------------------- row mappers
def _row_to_event(row: Any, *, cipher: StorePayloadCipher | None = None) -> RunEvent:
    """Map a run_events row to a RunEvent (payload is native via the codec).

    When a payload cipher is configured the at-rest payload envelope is decrypted back to
    the cleartext dict, bound to the row's ``event_id`` as AAD.
    """
    data = dict(row)
    data["timestamp"] = _iso(data.get("timestamp"))
    payload = data.get("payload") or {}
    if cipher is not None and isinstance(payload, dict):
        payload = cipher.decrypt_event_payload(
            payload, event_id=str(data.get("event_id", ""))
        )
    data["payload"] = payload
    return RunEvent.model_validate(data)


def _row_to_run(row: Any) -> RunRecord:
    """Map a runs row to a RunRecord (timestamps normalised to ISO)."""
    data = dict(row)
    data["created_at"] = _iso(data.get("created_at"))
    data["updated_at"] = _iso(data.get("updated_at"))
    data["metadata"] = data.get("metadata") or {}
    structured = data.get("output_structured")
    if isinstance(structured, dict) and set(structured.keys()) == {"value"}:
        data["output_structured"] = structured["value"]
    return RunRecord.model_validate(data)


def _row_to_recommendation(row: Any) -> RecommendationItem:
    """Map a recommendations row to a RecommendationItem."""
    data = dict(row)
    data["created_at"] = _iso(data.get("created_at"))
    data["metadata"] = data.get("metadata") or {}
    data["evidence_refs"] = data.get("evidence_refs") or []
    return RecommendationItem.model_validate(data)


class _PgStoreBase:
    """Shared base for the focused Postgres stores.

    Holds the injected ``require_pool`` callable (the facade's bound
    :meth:`PostgresStorageService._require_pool`) and the generic SQL helpers
    that several concern-stores reuse (keyed upsert/fetch, subject and
    environment/round listings). Each store's method bodies reference
    ``pool = self._require_pool()`` exactly as the original facade methods did.
    """

    def __init__(
        self,
        require_pool: Callable[[], Any],
        *,
        cipher: StorePayloadCipher | None = None,
    ) -> None:
        self._require_pool = require_pool
        self._cipher = cipher

    # ------------------------------------------------------------- generic helpers
    async def _save_keyed(
        self, table: str, pk: str, pk_value: str, obj: Any, *, extra: dict[str, Any]
    ) -> None:
        """Upsert a record whose columns are (pk, <extra...>, payload, created_at).

        The full model is stored in ``payload``; ``extra`` carries the indexed
        filter columns (subject_id/agent_id/environment_name/round).
        """
        pool = self._require_pool()
        columns = [pk, *extra.keys(), "payload", "created_at"]
        values = [
            pk_value,
            *extra.values(),
            obj.model_dump(mode="json"),
            getattr(obj, "created_at", None) or _utc_now(),
        ]
        placeholders = ", ".join(f"${i + 1}" for i in range(len(columns)))
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in columns if c != pk)
        async with pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) "
                f"ON CONFLICT ({pk}) DO UPDATE SET {updates}",
                *values,
            )

    async def _fetch_keyed(self, table: str, pk: str, pk_value: str) -> Any | None:
        """Fetch a single row's payload by primary key."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            return await conn.fetchrow(
                f"SELECT payload FROM {table} WHERE {pk} = $1", pk_value
            )

    async def _list_by_subject(self, table: str, subject_id: str | None) -> list[Any]:
        """List rows optionally filtered by subject_id, ordered by created_at."""
        pool = self._require_pool()
        if subject_id is None:
            sql = f"SELECT payload FROM {table} ORDER BY created_at ASC"
            params: list[Any] = []
        else:
            sql = (
                f"SELECT payload FROM {table} WHERE subject_id = $1 "
                "ORDER BY created_at ASC"
            )
            params = [subject_id]
        async with pool.acquire() as conn:
            return list(await conn.fetch(sql, *params))

    async def _list_by_env_round(
        self, table: str, environment_name: str | None, round: int | None
    ) -> list[Any]:
        """List rows filtered by environment_name and/or round."""
        pool = self._require_pool()
        clauses: list[str] = []
        params: list[Any] = []
        if environment_name is not None:
            params.append(environment_name)
            clauses.append(f"environment_name = ${len(params)}")
        if round is not None:
            params.append(round)
            clauses.append(f"round = ${len(params)}")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        async with pool.acquire() as conn:
            return list(
                await conn.fetch(
                    f"SELECT payload FROM {table}{where} ORDER BY created_at ASC",
                    *params,
                )
            )


class PostgresThreadStore(_PgStoreBase):
    """Postgres-backed chat-thread persistence keyed by ``thread_id``."""

    async def save_thread(self, thread: Any) -> Any:
        """Upsert a chat thread keyed by ``thread_id`` (refreshes ``updated_at``).

        When a payload cipher is configured each message's ``content`` is encrypted at
        rest (bound to the ``thread_id``); the indexed columns stay clear.
        """
        pool = self._require_pool()
        from himmy.core.ids import utc_now_iso

        payload = thread.model_dump(mode="json")
        if self._cipher is not None:
            payload = self._cipher.encrypt_thread_payload(
                payload, thread_id=thread.thread_id
            )
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO chat_threads
                    (thread_id, agent_id, version, payload, metadata, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (thread_id) DO UPDATE SET
                    agent_id = EXCLUDED.agent_id,
                    version = EXCLUDED.version,
                    payload = EXCLUDED.payload,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                """,
                thread.thread_id,
                thread.agent_id,
                thread.version,
                payload,
                dict(getattr(thread, "metadata", {}) or {}),
                utc_now_iso(),
            )
        return thread

    async def load_thread(self, thread_id: str) -> Any | None:
        """Load a chat thread by id, or None (message content decrypted at rest)."""
        from himmy.agents.base_agent.thread import ChatThread

        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM chat_threads WHERE thread_id = $1", thread_id
            )
        if row is None:
            return None
        payload = row["payload"]
        if self._cipher is not None and isinstance(payload, dict):
            payload = self._cipher.decrypt_thread_payload(payload, thread_id=thread_id)
        return ChatThread.model_validate(payload)


class PostgresEventLog(_PgStoreBase):
    """Postgres-backed append-only run-event stream (EventSink surface)."""

    async def append_event(self, event: RunEvent) -> None:
        """Append a run event (EventSink role); the payload is encrypted at rest."""
        pool = self._require_pool()
        payload = dict(event.payload or {})
        if self._cipher is not None:
            payload = self._cipher.encrypt_event_payload(
                payload, event_id=event.event_id
            )
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO run_events
                    (event_id, event_type, trace_id, thread_id, agent_id,
                     request_id, tool_call_id, latency_ms, cost, payload, error,
                     timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                ON CONFLICT (event_id) DO NOTHING
                """,
                event.event_id,
                event.event_type.value,
                event.trace_id,
                event.thread_id,
                event.agent_id,
                event.request_id,
                event.tool_call_id,
                event.latency_ms,
                event.cost,
                payload,
                event.error,
                event.timestamp,
            )

    async def list_events(
        self, thread_id: str | None = None, trace_id: str | None = None
    ) -> list[RunEvent]:
        """List events, optionally filtered by thread/trace id (insertion order)."""
        pool = self._require_pool()
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            params.append(thread_id)
            clauses.append(f"thread_id = ${len(params)}")
        if trace_id is not None:
            params.append(trace_id)
            clauses.append(f"trace_id = ${len(params)}")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM run_events{where} ORDER BY timestamp ASC", *params
            )
        return [_row_to_event(r, cipher=self._cipher) for r in rows]


class PostgresContextStore(_PgStoreBase):
    """Postgres-backed context fields, snapshots, and evidence."""

    async def save_context_field(self, field: Any) -> Any:
        """Upsert a context field keyed by ``(subject_id, key)``."""
        pool = self._require_pool()
        from himmy.core.ids import utc_now_iso

        subject_id = str(getattr(field, "metadata", {}).get("subject_id", ""))
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO context_fields
                    (subject_id, key, payload, metadata, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (subject_id, key) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    metadata = EXCLUDED.metadata,
                    updated_at = EXCLUDED.updated_at
                """,
                subject_id,
                field.key,
                field.model_dump(mode="json"),
                dict(getattr(field, "metadata", {}) or {}),
                utc_now_iso(),
            )
        return field

    async def get_context_field(self, subject_id: str, key: str) -> Any | None:
        """Return the context field for ``(subject_id, key)``, or None."""
        from himmy.services.context.models import ContextField

        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM context_fields WHERE subject_id = $1 AND key = $2",
                subject_id,
                key,
            )
        if row is None:
            return None
        return ContextField.model_validate(row["payload"])

    async def list_context_fields(self, subject_id: str) -> list[Any]:
        """Return all context fields for a subject."""
        from himmy.services.context.models import ContextField

        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT payload FROM context_fields WHERE subject_id = $1", subject_id
            )
        return [ContextField.model_validate(r["payload"]) for r in rows]

    async def save_snapshot(self, snapshot: Any) -> Any:
        """Upsert a context snapshot keyed by ``snapshot_id``."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO context_snapshots
                    (snapshot_id, subject_id, task_id, payload, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (snapshot_id) DO UPDATE SET
                    subject_id = EXCLUDED.subject_id,
                    task_id = EXCLUDED.task_id,
                    payload = EXCLUDED.payload,
                    metadata = EXCLUDED.metadata
                """,
                snapshot.snapshot_id,
                snapshot.subject_id,
                getattr(snapshot, "task_id", None),
                snapshot.model_dump(mode="json"),
                dict(getattr(snapshot, "metadata", {}) or {}),
                snapshot.created_at,
            )
        return snapshot

    async def load_snapshot(self, snapshot_id: str) -> Any | None:
        """Return a stored snapshot by id, or None."""
        from himmy.services.context.models import ContextSnapshot

        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT payload FROM context_snapshots WHERE snapshot_id = $1",
                snapshot_id,
            )
        if row is None:
            return None
        return ContextSnapshot.model_validate(row["payload"])

    async def save_context_evidence(self, record: Any) -> Any:
        """Persist a context evidence record (idempotent on its id)."""
        pool = self._require_pool()
        # Accept either a ContextEvidenceRecord or an EvidenceRef-like object.
        evidence_id = getattr(record, "evidence_id", None) or new_uuid_safe()
        payload = (
            record.model_dump(mode="json")
            if hasattr(record, "model_dump")
            else dict(record)
        )
        created_at = getattr(record, "created_at", None) or _utc_now()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO context_evidence
                    (evidence_id, subject_id, snapshot_id, key, payload, metadata,
                     created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (evidence_id) DO UPDATE SET
                    subject_id = EXCLUDED.subject_id,
                    snapshot_id = EXCLUDED.snapshot_id,
                    key = EXCLUDED.key,
                    payload = EXCLUDED.payload,
                    metadata = EXCLUDED.metadata
                """,
                evidence_id,
                getattr(record, "subject_id", None),
                getattr(record, "snapshot_id", None),
                getattr(record, "key", None),
                payload,
                dict(getattr(record, "metadata", {}) or {}),
                created_at,
            )
        return record


class PostgresRunStore(_PgStoreBase):
    """Postgres-backed run records, including atomic idempotent creation."""

    async def save_run(self, run: RunRecord) -> RunRecord:
        """Upsert a run record keyed by ``run_id``; storage stamps ``updated_at``.

        Storage owns ``updated_at`` so it can never drift. The partial UNIQUE
        index ``runs_idempotency_idx`` is the source of truth for the idempotency
        constraint; this method is a plain upsert by ``run_id`` (use
        :meth:`save_run_if_absent_by_idempotency` for the race-free create).
        """
        from himmy.core.ids import utc_now_iso

        run.updated_at = utc_now_iso()
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(self._RUN_UPSERT_BY_ID, *self._run_params(run))
        return run

    async def save_run_if_absent_by_idempotency(
        self, run: RunRecord
    ) -> tuple[RunRecord, bool]:
        """Atomically create a run unless its idempotency key already exists.

        Returns ``(run, created)``. The DB partial UNIQUE index closes the race:
        ``INSERT ... ON CONFLICT (workspace_id, idempotency_key) DO NOTHING`` lets
        exactly one concurrent writer win. On conflict the existing row is read
        back and returned with ``created=False``. Runs without an idempotency key
        are always inserted (the partial index does not apply to NULL keys).
        """
        from himmy.core.ids import utc_now_iso

        run.updated_at = utc_now_iso()
        pool = self._require_pool()
        if run.idempotency_key is None:
            async with pool.acquire() as conn:
                await conn.execute(self._RUN_UPSERT_BY_ID, *self._run_params(run))
            return run, True

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO runs
                    (run_id, workspace_id, subject_id, task_id, thread_id,
                     snapshot_id, persona_name, model_key, idempotency_key, status,
                     output_text, output_structured, error, trace_id, metadata,
                     created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17)
                ON CONFLICT (workspace_id, idempotency_key)
                    WHERE idempotency_key IS NOT NULL
                    DO NOTHING
                RETURNING run_id
                """,
                *self._run_params(run),
            )
            if row is not None:
                return run, True
            existing = await conn.fetchrow(
                """
                SELECT * FROM runs
                WHERE workspace_id = $1 AND idempotency_key = $2
                """,
                run.workspace_id,
                run.idempotency_key,
            )
        return _row_to_run(existing), False

    _RUN_UPSERT_BY_ID = """
        INSERT INTO runs
            (run_id, workspace_id, subject_id, task_id, thread_id, snapshot_id,
             persona_name, model_key, idempotency_key, status, output_text,
             output_structured, error, trace_id, metadata, created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17)
        ON CONFLICT (run_id) DO UPDATE SET
            workspace_id = EXCLUDED.workspace_id,
            subject_id = EXCLUDED.subject_id,
            task_id = EXCLUDED.task_id,
            thread_id = EXCLUDED.thread_id,
            snapshot_id = EXCLUDED.snapshot_id,
            persona_name = EXCLUDED.persona_name,
            model_key = EXCLUDED.model_key,
            idempotency_key = EXCLUDED.idempotency_key,
            status = EXCLUDED.status,
            output_text = EXCLUDED.output_text,
            output_structured = EXCLUDED.output_structured,
            error = EXCLUDED.error,
            trace_id = EXCLUDED.trace_id,
            metadata = EXCLUDED.metadata,
            updated_at = EXCLUDED.updated_at
    """

    @staticmethod
    def _run_params(run: RunRecord) -> list[Any]:
        """Positional params for the runs upsert (output_structured as jsonb)."""
        structured = run.output_structured
        # output_structured is a jsonb column: the codec encodes dict/list; wrap
        # scalars/None so the bind type is unambiguous.
        if structured is not None and not isinstance(structured, (dict, list)):
            structured = {"value": structured}
        return [
            run.run_id,
            run.workspace_id,
            run.subject_id,
            run.task_id,
            run.thread_id,
            run.snapshot_id,
            run.persona_name,
            run.model_key,
            run.idempotency_key,
            run.status.value,
            run.output_text,
            structured,
            run.error,
            run.trace_id,
            dict(run.metadata or {}),
            run.created_at,
            run.updated_at,
        ]

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run record by id, or None."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM runs WHERE run_id = $1", run_id)
        return _row_to_run(row) if row else None

    async def list_runs(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[RunRecord]:
        """List runs filtered by workspace, subject, and/or status."""
        pool = self._require_pool()
        clauses: list[str] = []
        params: list[Any] = []
        if workspace_id is not None:
            params.append(workspace_id)
            clauses.append(f"workspace_id = ${len(params)}")
        if subject_id is not None:
            params.append(subject_id)
            clauses.append(f"subject_id = ${len(params)}")
        if status is not None:
            params.append(status.value)
            clauses.append(f"status = ${len(params)}")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM runs{where} ORDER BY created_at ASC", *params
            )
        return [_row_to_run(r) for r in rows]

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None:
        """Return the existing run for an idempotency key, or None."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM runs WHERE workspace_id = $1 AND idempotency_key = $2",
                workspace_id,
                idempotency_key,
            )
        return _row_to_run(row) if row else None


class PostgresRecommendationStore(_PgStoreBase):
    """Postgres-backed recommendation items."""

    async def save_recommendation(self, item: RecommendationItem) -> RecommendationItem:
        """Upsert a recommendation keyed by ``recommendation_id``."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO recommendations
                    (recommendation_id, run_id, workspace_id, subject_id, kind,
                     title, summary, rationale, confidence, evidence_refs, status,
                     notes, metadata, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
                ON CONFLICT (recommendation_id) DO UPDATE SET
                    run_id = EXCLUDED.run_id,
                    workspace_id = EXCLUDED.workspace_id,
                    subject_id = EXCLUDED.subject_id,
                    kind = EXCLUDED.kind,
                    title = EXCLUDED.title,
                    summary = EXCLUDED.summary,
                    rationale = EXCLUDED.rationale,
                    confidence = EXCLUDED.confidence,
                    evidence_refs = EXCLUDED.evidence_refs,
                    status = EXCLUDED.status,
                    notes = EXCLUDED.notes,
                    metadata = EXCLUDED.metadata
                """,
                item.recommendation_id,
                item.run_id,
                item.workspace_id,
                item.subject_id,
                item.kind,
                item.title,
                item.summary,
                item.rationale,
                item.confidence,
                list(item.evidence_refs or []),
                item.status.value,
                item.notes,
                dict(item.metadata or {}),
                item.created_at,
            )
        return item

    async def get_recommendation(
        self, recommendation_id: str
    ) -> RecommendationItem | None:
        """Return a recommendation by id, or None."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM recommendations WHERE recommendation_id = $1",
                recommendation_id,
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
        pool = self._require_pool()
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
                params.append(value)
                clauses.append(f"{column} = ${len(params)}")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM recommendations{where} ORDER BY created_at ASC",
                *params,
            )
        return [_row_to_recommendation(r) for r in rows]

    async def update_recommendation(
        self,
        recommendation_id: str,
        *,
        status: RecommendationStatus | None = None,
        notes: str | None = None,
    ) -> RecommendationItem | None:
        """Update a recommendation's status/notes; return it or None."""
        pool = self._require_pool()
        sets: list[str] = []
        params: list[Any] = []
        if status is not None:
            params.append(status.value)
            sets.append(f"status = ${len(params)}")
        if notes is not None:
            params.append(notes)
            sets.append(f"notes = ${len(params)}")
        if not sets:
            return await self.get_recommendation(recommendation_id)
        params.append(recommendation_id)
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                f"UPDATE recommendations SET {', '.join(sets)} "
                f"WHERE recommendation_id = ${len(params)} RETURNING *",
                *params,
            )
        return _row_to_recommendation(row) if row else None


class PostgresEvaluationStore(_PgStoreBase):
    """Postgres-backed evaluation runs keyed by ``run_id``."""

    async def save_evaluation_run(self, run: Any) -> Any:
        """Upsert an evaluation run keyed by ``run_id`` (stamping its workspace)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO evaluation_runs
                    (run_id, suite_id, suite_name, workspace_id, aggregate_score,
                     payload, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (run_id) DO UPDATE SET
                    suite_id = EXCLUDED.suite_id,
                    suite_name = EXCLUDED.suite_name,
                    workspace_id = EXCLUDED.workspace_id,
                    aggregate_score = EXCLUDED.aggregate_score,
                    payload = EXCLUDED.payload
                """,
                run.run_id,
                getattr(run, "suite_id", None),
                getattr(run, "suite_name", None),
                getattr(run, "workspace_id", None),
                getattr(run, "aggregate_score", 0.0),
                run.model_dump(mode="json"),
                getattr(run, "created_at", None) or _utc_now(),
            )
        return run

    async def get_evaluation_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> Any | None:
        """Return an evaluation run by id, tenant-scoped (AAEO-4).

        When ``workspace_id`` is supplied, a run belonging to another workspace is
        treated as not found (returns None) — closing the cross-tenant IDOR.
        """
        from himmy.services.evaluation.models import EvaluationRun

        pool = self._require_pool()
        if workspace_id is None:
            sql = "SELECT payload FROM evaluation_runs WHERE run_id = $1"
            params: list[Any] = [run_id]
        else:
            sql = (
                "SELECT payload FROM evaluation_runs "
                "WHERE run_id = $1 AND workspace_id = $2"
            )
            params = [run_id, workspace_id]
        async with pool.acquire() as conn:
            row = await conn.fetchrow(sql, *params)
        if row is None:
            return None
        return EvaluationRun.model_validate(row["payload"])

    async def list_evaluation_runs(
        self, suite_id: str | None = None, *, workspace_id: str | None = None
    ) -> list[Any]:
        """List evaluation runs, optionally filtered by suite id and workspace (AAEO-4)."""
        from himmy.services.evaluation.models import EvaluationRun

        pool = self._require_pool()
        clauses: list[str] = []
        params: list[Any] = []
        if suite_id is not None:
            params.append(suite_id)
            clauses.append(f"suite_id = ${len(params)}")
        if workspace_id is not None:
            params.append(workspace_id)
            clauses.append(f"workspace_id = ${len(params)}")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT payload FROM evaluation_runs{where} ORDER BY created_at ASC"
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        return [EvaluationRun.model_validate(r["payload"]) for r in rows]


class PostgresOrchestrationStore(_PgStoreBase):
    """Postgres-backed multi-agent + world-model records.

    Cognitive memory objects, episodic memories, agent states, actions, and
    environment states used by the multi-agent orchestrators and simulations.
    """

    async def save_memory(self, obj: MemoryObject) -> MemoryObject:
        """Upsert a cognitive memory object."""
        await self._save_keyed(
            "memory_objects",
            "memory_id",
            obj.memory_id,
            obj,
            extra={"subject_id": obj.subject_id, "agent_id": obj.agent_id},
        )
        return obj

    async def get_memory(self, memory_id: str) -> MemoryObject | None:
        """Return a memory object by id, or None."""
        row = await self._fetch_keyed("memory_objects", "memory_id", memory_id)
        return MemoryObject.model_validate(row["payload"]) if row else None

    async def list_memory(self, subject_id: str | None = None) -> list[MemoryObject]:
        """List memory objects, optionally filtered by subject."""
        rows = await self._list_by_subject("memory_objects", subject_id)
        return [MemoryObject.model_validate(r["payload"]) for r in rows]

    async def save_episodic_memory(
        self, obj: EpisodicMemoryObject
    ) -> EpisodicMemoryObject:
        """Upsert an episodic memory object."""
        await self._save_keyed(
            "episodic_memory_objects",
            "episode_id",
            obj.episode_id,
            obj,
            extra={"subject_id": obj.subject_id, "agent_id": obj.agent_id},
        )
        return obj

    async def get_episodic_memory(self, episode_id: str) -> EpisodicMemoryObject | None:
        """Return an episodic memory object by id, or None."""
        row = await self._fetch_keyed(
            "episodic_memory_objects", "episode_id", episode_id
        )
        return EpisodicMemoryObject.model_validate(row["payload"]) if row else None

    async def list_episodic_memory(
        self, subject_id: str | None = None
    ) -> list[EpisodicMemoryObject]:
        """List episodic memory objects, optionally filtered by subject."""
        rows = await self._list_by_subject("episodic_memory_objects", subject_id)
        return [EpisodicMemoryObject.model_validate(r["payload"]) for r in rows]

    async def save_agent_state(self, record: AgentStateRecord) -> AgentStateRecord:
        """Upsert an agent state record."""
        await self._save_keyed(
            "agent_states",
            "state_id",
            record.state_id,
            record,
            extra={
                "agent_id": record.agent_id,
                "environment_name": record.environment_name,
                "round": record.round,
            },
        )
        return record

    async def get_agent_state(self, state_id: str) -> AgentStateRecord | None:
        """Return an agent state record by id, or None."""
        row = await self._fetch_keyed("agent_states", "state_id", state_id)
        return AgentStateRecord.model_validate(row["payload"]) if row else None

    async def list_agent_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[AgentStateRecord]:
        """List agent state records filtered by environment and/or round."""
        rows = await self._list_by_env_round("agent_states", environment_name, round)
        return [AgentStateRecord.model_validate(r["payload"]) for r in rows]

    async def save_action(self, record: ActionRecord) -> ActionRecord:
        """Upsert an action record."""
        await self._save_keyed(
            "actions",
            "action_id",
            record.action_id,
            record,
            extra={
                "agent_id": record.agent_id,
                "environment_name": record.environment_name,
                "round": record.round,
            },
        )
        return record

    async def get_action(self, action_id: str) -> ActionRecord | None:
        """Return an action record by id, or None."""
        row = await self._fetch_keyed("actions", "action_id", action_id)
        return ActionRecord.model_validate(row["payload"]) if row else None

    async def list_actions(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[ActionRecord]:
        """List action records filtered by environment and/or round."""
        rows = await self._list_by_env_round("actions", environment_name, round)
        return [ActionRecord.model_validate(r["payload"]) for r in rows]

    async def save_environment_state(
        self, record: EnvironmentStateRecord
    ) -> EnvironmentStateRecord:
        """Upsert an environment state record."""
        await self._save_keyed(
            "environment_states",
            "environment_state_id",
            record.environment_state_id,
            record,
            extra={
                "environment_name": record.environment_name,
                "round": record.round,
            },
        )
        return record

    async def get_environment_state(
        self, environment_state_id: str
    ) -> EnvironmentStateRecord | None:
        """Return an environment state record by id, or None."""
        row = await self._fetch_keyed(
            "environment_states", "environment_state_id", environment_state_id
        )
        return EnvironmentStateRecord.model_validate(row["payload"]) if row else None

    async def list_environment_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[EnvironmentStateRecord]:
        """List environment state records filtered by environment and/or round."""
        rows = await self._list_by_env_round(
            "environment_states", environment_name, round
        )
        return [EnvironmentStateRecord.model_validate(r["payload"]) for r in rows]


class PostgresStorageService:
    """Postgres-backed storage mirroring :class:`StorageService` 1:1.

    Construct via :meth:`connect` (or inject a pool for tests). Every data method
    is a real SQL body gated on a live pool via :meth:`_require_pool`; the
    in-memory ``StorageService`` remains the offline default. The DDL in
    :data:`STORAGE_DDL` is the full target schema (14 tables + the ``ai_call_log``
    view + a ``schema_migrations`` table) and can be inspected without a database.

    Supports ``async with`` for deterministic pool teardown::

        async with await PostgresStorageService.connect(dsn) as storage:
            await storage.migrate()
            ...
    """

    def __init__(
        self,
        pool: Any | None = None,
        *,
        cipher: StorePayloadCipher | None | _Unset = _UNSET,
    ) -> None:
        self._pool = pool
        # Field encryption at rest for the sensitive payloads (message content +
        # run-event payloads). Defaults to the env-configured cipher (``None`` ⇒ plaintext,
        # offline path unchanged); pass an explicit cipher (or ``None``) to override.
        self._cipher = build_store_cipher() if isinstance(cipher, _Unset) else cipher
        # Compose the focused per-concern stores behind this facade, injecting the
        # bound ``_require_pool`` so each store reads the live pool lazily. Only the
        # stores that persist sensitive payloads receive the cipher.
        self._thread_store = PostgresThreadStore(
            self._require_pool, cipher=self._cipher
        )
        self._event_log = PostgresEventLog(self._require_pool, cipher=self._cipher)
        self._context_store = PostgresContextStore(self._require_pool)
        self._run_store = PostgresRunStore(self._require_pool)
        self._recommendation_store = PostgresRecommendationStore(self._require_pool)
        self._evaluation_store = PostgresEvaluationStore(self._require_pool)
        self._orchestration_store = PostgresOrchestrationStore(self._require_pool)

    @classmethod
    async def connect(
        cls,
        dsn: str,
        *,
        min_size: int = 1,
        max_size: int = 10,
        timeout: float = 30.0,
        command_timeout: float | None = 30.0,
        max_inactive_connection_lifetime: float = 300.0,
    ) -> PostgresStorageService:
        """Create an asyncpg pool (with JSONB codec + timeouts) and wire a service.

        ``timeout`` bounds connection acquisition, ``command_timeout`` bounds each
        statement, and the JSONB codec is registered on every pooled connection.
        Requires the [postgres] extra and a running database.
        """
        asyncpg = _require_asyncpg()
        pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            timeout=timeout,
            command_timeout=command_timeout,
            max_inactive_connection_lifetime=max_inactive_connection_lifetime,
            init=_register_codecs,
        )
        return cls(pool=pool)

    async def close(self) -> None:
        """Close the connection pool (idempotent)."""
        if self._pool is not None:
            await self._pool.close()

    async def __aenter__(self) -> PostgresStorageService:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def create_schema(self) -> None:
        """Apply the idempotent storage DDL (CREATE TABLE/VIEW IF NOT EXISTS)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(STORAGE_DDL)

    # ----------------------------------------------------------- knowledge / pgvector
    def knowledge_backend(self) -> Any:
        """Return a pgvector knowledge backend bound to this service's pool.

        The backend (``PgVectorKnowledgeBackend``) implements the documented KB
        persistence + cosine search; wire it into a ``KnowledgeBase`` via
        ``KnowledgeBase(..., backend=storage.knowledge_backend())``. Import is lazy
        so this module stays import-safe without the knowledge extras.
        """
        from himmy.services.knowledge.backend import PgVectorKnowledgeBackend

        return PgVectorKnowledgeBackend(self._require_pool())

    async def create_knowledge_schema(
        self,
        *,
        vector_dim: int,
        embedding_column_type: str | None = None,
        index_method: str | None = None,
        ivfflat_lists: int = 100,
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 64,
    ) -> None:
        """Create the pgvector knowledge schema (call after :meth:`create_schema`).

        Picks ``vector(N)``/``halfvec(N)`` + ``ivfflat``/``hnsw`` from ``vector_dim``
        (overridable) and auto-detects the committed column type from
        ``pg_attribute`` so reconnects don't miscast. See
        :class:`~himmy.services.knowledge.backend.PgVectorKnowledgeBackend`.
        """
        backend = self.knowledge_backend()
        await backend.create_knowledge_schema(
            vector_dim=vector_dim,
            embedding_column_type=embedding_column_type,
            index_method=index_method,
            ivfflat_lists=ivfflat_lists,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
        )

    async def migrate(self) -> list[int]:
        """Run pending forward migrations; return the versions newly applied.

        Each migration in :data:`STORAGE_MIGRATIONS` is applied at most once,
        tracked in the ``schema_migrations`` table, inside a transaction. The base
        schema (which creates ``schema_migrations``) is migration 1 and uses
        CREATE ... IF NOT EXISTS, so this is safe on a fresh or partially-migrated
        database.

        Concurrent migrators (e.g. two processes booting simultaneously) are
        serialised by a session-level ``pg_advisory_lock`` on
        :data:`MIGRATION_ADVISORY_LOCK_KEY`, held for the duration of the run: the
        loser blocks until the winner finishes, then re-reads ``schema_migrations``
        and applies nothing. The already-migrated fast path costs one extra
        lock/unlock round trip.
        """
        pool = self._require_pool()
        applied: list[int] = []
        async with pool.acquire() as conn:
            await conn.execute(
                "SELECT pg_advisory_lock($1)", MIGRATION_ADVISORY_LOCK_KEY
            )
            try:
                # Bootstrap the tracking table so the first migration can record
                # itself.
                await conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS schema_migrations (
                        version    INTEGER PRIMARY KEY,
                        name       TEXT NOT NULL,
                        applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                )
                rows = await conn.fetch("SELECT version FROM schema_migrations")
                done = {r["version"] for r in rows}
                for version, name, statements in sorted(STORAGE_MIGRATIONS):
                    if version in done:
                        continue
                    async with conn.transaction():
                        for statement in statements:
                            await conn.execute(statement)
                        await conn.execute(
                            "INSERT INTO schema_migrations (version, name) "
                            "VALUES ($1, $2) ON CONFLICT (version) DO NOTHING",
                            version,
                            name,
                        )
                    applied.append(version)
            finally:
                await conn.execute(
                    "SELECT pg_advisory_unlock($1)", MIGRATION_ADVISORY_LOCK_KEY
                )
        return applied

    def _require_pool(self) -> Any:
        """Return the live pool or raise a clear HimmyError."""
        if self._pool is None:
            raise HimmyError(
                "PostgresStorageService requires the [postgres] extra "
                "+ a running DB; use PostgresStorageService.connect(dsn)."
            )
        return self._pool

    # ------------------------------------------------------------------ threads
    async def save_thread(self, thread: Any) -> Any:
        """Upsert a chat thread keyed by ``thread_id`` (refreshes ``updated_at``)."""
        return await self._thread_store.save_thread(thread)

    async def load_thread(self, thread_id: str) -> Any | None:
        """Load a chat thread by id, or None."""
        return await self._thread_store.load_thread(thread_id)

    # ------------------------------------------------------------------- events
    async def append_event(self, event: RunEvent) -> None:
        """Append a run event (EventSink role)."""
        await self._event_log.append_event(event)

    async def list_events(
        self, thread_id: str | None = None, trace_id: str | None = None
    ) -> list[RunEvent]:
        """List events, optionally filtered by thread/trace id (insertion order)."""
        return await self._event_log.list_events(thread_id, trace_id)

    # ------------------------------------------------------------------ context
    async def save_context_field(self, field: Any) -> Any:
        """Upsert a context field keyed by ``(subject_id, key)``."""
        return await self._context_store.save_context_field(field)

    async def get_context_field(self, subject_id: str, key: str) -> Any | None:
        """Return the context field for ``(subject_id, key)``, or None."""
        return await self._context_store.get_context_field(subject_id, key)

    async def list_context_fields(self, subject_id: str) -> list[Any]:
        """Return all context fields for a subject."""
        return await self._context_store.list_context_fields(subject_id)

    async def save_snapshot(self, snapshot: Any) -> Any:
        """Upsert a context snapshot keyed by ``snapshot_id``."""
        return await self._context_store.save_snapshot(snapshot)

    async def load_snapshot(self, snapshot_id: str) -> Any | None:
        """Return a stored snapshot by id, or None."""
        return await self._context_store.load_snapshot(snapshot_id)

    async def save_context_evidence(self, record: Any) -> Any:
        """Persist a context evidence record (idempotent on its id)."""
        return await self._context_store.save_context_evidence(record)

    # --------------------------------------------------------------------- runs
    async def save_run(self, run: RunRecord) -> RunRecord:
        """Upsert a run record keyed by ``run_id``; storage stamps ``updated_at``."""
        return await self._run_store.save_run(run)

    async def save_run_if_absent_by_idempotency(
        self, run: RunRecord
    ) -> tuple[RunRecord, bool]:
        """Atomically create a run unless its idempotency key already exists."""
        return await self._run_store.save_run_if_absent_by_idempotency(run)

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run record by id, or None."""
        return await self._run_store.get_run(run_id)

    async def list_runs(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        status: RunStatus | None = None,
    ) -> list[RunRecord]:
        """List runs filtered by workspace, subject, and/or status."""
        return await self._run_store.list_runs(workspace_id, subject_id, status)

    async def load_run_by_idempotency(
        self, workspace_id: str, idempotency_key: str
    ) -> RunRecord | None:
        """Return the existing run for an idempotency key, or None."""
        return await self._run_store.load_run_by_idempotency(
            workspace_id, idempotency_key
        )

    # ---------------------------------------------------------- recommendations
    async def save_recommendation(self, item: RecommendationItem) -> RecommendationItem:
        """Upsert a recommendation keyed by ``recommendation_id``."""
        return await self._recommendation_store.save_recommendation(item)

    async def get_recommendation(
        self, recommendation_id: str
    ) -> RecommendationItem | None:
        """Return a recommendation by id, or None."""
        return await self._recommendation_store.get_recommendation(recommendation_id)

    async def list_recommendations(
        self,
        workspace_id: str | None = None,
        subject_id: str | None = None,
        run_id: str | None = None,
        kind: str | None = None,
        status: RecommendationStatus | None = None,
    ) -> list[RecommendationItem]:
        """List recommendations filtered by the given dimensions."""
        return await self._recommendation_store.list_recommendations(
            workspace_id, subject_id, run_id, kind, status
        )

    async def update_recommendation(
        self,
        recommendation_id: str,
        *,
        status: RecommendationStatus | None = None,
        notes: str | None = None,
    ) -> RecommendationItem | None:
        """Update a recommendation's status/notes; return it or None."""
        return await self._recommendation_store.update_recommendation(
            recommendation_id, status=status, notes=notes
        )

    # --------------------------------------------------------------- evaluation
    async def save_evaluation_run(self, run: Any) -> Any:
        """Upsert an evaluation run keyed by ``run_id``."""
        return await self._evaluation_store.save_evaluation_run(run)

    async def get_evaluation_run(
        self, run_id: str, *, workspace_id: str | None = None
    ) -> Any | None:
        """Return an evaluation run by id, tenant-scoped (AAEO-4)."""
        return await self._evaluation_store.get_evaluation_run(
            run_id, workspace_id=workspace_id
        )

    async def list_evaluation_runs(
        self, suite_id: str | None = None, *, workspace_id: str | None = None
    ) -> list[Any]:
        """List evaluation runs, optionally filtered by suite id and workspace (AAEO-4)."""
        return await self._evaluation_store.list_evaluation_runs(
            suite_id, workspace_id=workspace_id
        )

    # --------------------------------------------- memory + orchestration records
    async def save_memory(self, obj: MemoryObject) -> MemoryObject:
        """Upsert a cognitive memory object."""
        return await self._orchestration_store.save_memory(obj)

    async def get_memory(self, memory_id: str) -> MemoryObject | None:
        """Return a memory object by id, or None."""
        return await self._orchestration_store.get_memory(memory_id)

    async def list_memory(self, subject_id: str | None = None) -> list[MemoryObject]:
        """List memory objects, optionally filtered by subject."""
        return await self._orchestration_store.list_memory(subject_id)

    async def save_episodic_memory(
        self, obj: EpisodicMemoryObject
    ) -> EpisodicMemoryObject:
        """Upsert an episodic memory object."""
        return await self._orchestration_store.save_episodic_memory(obj)

    async def get_episodic_memory(self, episode_id: str) -> EpisodicMemoryObject | None:
        """Return an episodic memory object by id, or None."""
        return await self._orchestration_store.get_episodic_memory(episode_id)

    async def list_episodic_memory(
        self, subject_id: str | None = None
    ) -> list[EpisodicMemoryObject]:
        """List episodic memory objects, optionally filtered by subject."""
        return await self._orchestration_store.list_episodic_memory(subject_id)

    async def save_agent_state(self, record: AgentStateRecord) -> AgentStateRecord:
        """Upsert an agent state record."""
        return await self._orchestration_store.save_agent_state(record)

    async def get_agent_state(self, state_id: str) -> AgentStateRecord | None:
        """Return an agent state record by id, or None."""
        return await self._orchestration_store.get_agent_state(state_id)

    async def list_agent_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[AgentStateRecord]:
        """List agent state records filtered by environment and/or round."""
        return await self._orchestration_store.list_agent_states(
            environment_name, round
        )

    async def save_action(self, record: ActionRecord) -> ActionRecord:
        """Upsert an action record."""
        return await self._orchestration_store.save_action(record)

    async def get_action(self, action_id: str) -> ActionRecord | None:
        """Return an action record by id, or None."""
        return await self._orchestration_store.get_action(action_id)

    async def list_actions(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[ActionRecord]:
        """List action records filtered by environment and/or round."""
        return await self._orchestration_store.list_actions(environment_name, round)

    async def save_environment_state(
        self, record: EnvironmentStateRecord
    ) -> EnvironmentStateRecord:
        """Upsert an environment state record."""
        return await self._orchestration_store.save_environment_state(record)

    async def get_environment_state(
        self, environment_state_id: str
    ) -> EnvironmentStateRecord | None:
        """Return an environment state record by id, or None."""
        return await self._orchestration_store.get_environment_state(
            environment_state_id
        )

    async def list_environment_states(
        self, environment_name: str | None = None, round: int | None = None
    ) -> list[EnvironmentStateRecord]:
        """List environment state records filtered by environment and/or round."""
        return await self._orchestration_store.list_environment_states(
            environment_name, round
        )


def _utc_now() -> str:
    """Return the current UTC time as an ISO string (lazy import of the helper)."""
    from himmy.core.ids import utc_now_iso

    return utc_now_iso()


def new_uuid_safe() -> str:
    """Return a fresh UUID string (lazy import of the helper)."""
    from himmy.core.ids import new_uuid

    return new_uuid()


__all__ = ["PostgresStorageService", "STORAGE_DDL", "STORAGE_MIGRATIONS"]
