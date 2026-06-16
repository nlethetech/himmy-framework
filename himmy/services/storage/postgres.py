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
from typing import TYPE_CHECKING, Any, cast

from himmy.core.errors import HimmyError
from himmy.core.events import RunEvent
from himmy.core.ids import iso_plus_seconds as _iso_plus_seconds
from himmy.core.ids import utc_now_iso
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
from himmy.services.storage.protocols import normalize_event_type

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.storage.trigger_dedup import DedupClaim


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
    seq          BIGSERIAL,
    event_id     TEXT PRIMARY KEY,
    event_type   TEXT NOT NULL,
    trace_id     TEXT,
    thread_id    TEXT,
    agent_id     TEXT,
    request_id   TEXT,
    tool_call_id TEXT,
    tool_name    TEXT,
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
CREATE INDEX IF NOT EXISTS run_events_seq_idx ON run_events (seq);
CREATE INDEX IF NOT EXISTS run_events_type_tool_idx ON run_events (event_type, tool_name);

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

CREATE TABLE IF NOT EXISTS agent_defs (
    agent_id        TEXT PRIMARY KEY,
    workspace_id    TEXT NOT NULL,
    name            TEXT NOT NULL,
    idempotency_key TEXT,
    payload         JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS agent_defs_workspace_id_idx ON agent_defs (workspace_id);
CREATE UNIQUE INDEX IF NOT EXISTS agent_defs_idempotency_idx
    ON agent_defs (workspace_id, idempotency_key)
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
    # v3 (T2e): durable workspace-scoped stored agent definitions backing /v1/agents.
    # The base DDL also declares this table (IF NOT EXISTS), so a fresh database lays it
    # down with the base; this migration reaches an EXISTING database that provisioned
    # the base before this table existed. ``IF NOT EXISTS`` makes the double-path safe.
    (
        3,
        "agent_defs",
        [
            "CREATE TABLE IF NOT EXISTS agent_defs ("
            "agent_id TEXT PRIMARY KEY, "
            "workspace_id TEXT NOT NULL, "
            "name TEXT NOT NULL, "
            "idempotency_key TEXT, "
            "payload JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now())",
            "CREATE INDEX IF NOT EXISTS agent_defs_workspace_id_idx "
            "ON agent_defs (workspace_id)",
            "CREATE UNIQUE INDEX IF NOT EXISTS agent_defs_idempotency_idx "
            "ON agent_defs (workspace_id, idempotency_key) "
            "WHERE idempotency_key IS NOT NULL",
        ],
    ),
    # v4 (K3 + K4): the auxiliary stores' Postgres mirrors, routed by HIMMY_DATABASE_URL
    # through the K2 aux-store selector so a Postgres deployment stops forking these to
    # local .himmy/*.db sidecars. Every aux table carries a ``tenant`` discriminator (K3
    # reviewer must_fix): Studio's HITL inbox (``approvals.db``) and /v1's
    # (``v1_approvals.db``) are DELIBERATELY separate surfaces, so a single undifferentiated
    # table would let a /v1 tenant read Studio's (or another tenant's) checkpoints. All reads
    # are scoped on ``tenant``. The per-record JSON ``schema_version`` migration
    # (``_migrate_*`` in himmy.runtime.checkpoint) is still applied ON READ on the Postgres
    # path — appending a table here does nothing for the per-record blob schema.
    #
    # The bodies are JSON blobs (``data``) mirroring the SQLite stores 1:1 (durability
    # routing only; per-subject envelope-encryption is a later phase, S3). Indexed columns
    # carry the discriminators the SQLite stores filter on (status / workspace_id / origin).
    (
        4,
        "aux_store_mirrors",
        [
            # --- K3: HITL approval checkpoints (Studio + /v1, tenant-scoped) ---
            "CREATE TABLE IF NOT EXISTS aux_agent_checkpoints ("
            "tenant TEXT NOT NULL, "
            "checkpoint_id TEXT NOT NULL, "
            "status TEXT NOT NULL, "
            "data JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, checkpoint_id))",
            "CREATE INDEX IF NOT EXISTS aux_agent_checkpoints_status_idx "
            "ON aux_agent_checkpoints (tenant, status)",
            # --- K3: durable graph-run checkpoints (long team/workflow runs) ---
            "CREATE TABLE IF NOT EXISTS aux_graph_checkpoints ("
            "tenant TEXT NOT NULL, "
            "checkpoint_id TEXT NOT NULL, "
            "graph_name TEXT NOT NULL DEFAULT '', "
            "status TEXT NOT NULL, "
            "data JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, checkpoint_id))",
            "CREATE INDEX IF NOT EXISTS aux_graph_checkpoints_status_idx "
            "ON aux_graph_checkpoints (tenant, status)",
            # --- K4: unified conversation store (CLI sessions + Studio chats) ---
            "CREATE TABLE IF NOT EXISTS aux_conversations ("
            "tenant TEXT NOT NULL, "
            "conversation_id TEXT NOT NULL, "
            "thread JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "origin TEXT NOT NULL DEFAULT 'cli', "
            "title TEXT, "
            "agent_path TEXT, "
            "provider TEXT, "
            "project_id TEXT, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, conversation_id))",
            "CREATE INDEX IF NOT EXISTS aux_conversations_updated_idx "
            "ON aux_conversations (tenant, updated_at DESC)",
            "CREATE TABLE IF NOT EXISTS aux_conversation_messages ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "conversation_id TEXT NOT NULL, "
            "role TEXT NOT NULL, "
            "text TEXT NOT NULL, "
            "seq INTEGER NOT NULL, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_conversation_messages_conv_idx "
            "ON aux_conversation_messages (tenant, conversation_id, seq)",
            "CREATE TABLE IF NOT EXISTS aux_projects ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "name TEXT NOT NULL, "
            "description TEXT NOT NULL DEFAULT '', "
            "kb_id TEXT, "
            "agent_path TEXT, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "updated_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, id))",
            # --- K4: scheduled routines (atomic cluster-wide tick via last_run_at CAS) ---
            "CREATE TABLE IF NOT EXISTS aux_routines ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "workspace_id TEXT NOT NULL DEFAULT '__local__', "
            "data JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "last_run_at TEXT, "
            "agent_id TEXT, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_routines_workspace_idx "
            "ON aux_routines (tenant, workspace_id)",
            # --- K4: teams + workflows (stored-agent orchestration bindings) ---
            "CREATE TABLE IF NOT EXISTS aux_teams ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "workspace_id TEXT NOT NULL DEFAULT '__local__', "
            "data JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "idempotency_key TEXT, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_teams_workspace_idx "
            "ON aux_teams (tenant, workspace_id)",
            "CREATE TABLE IF NOT EXISTS aux_workflows ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "workspace_id TEXT NOT NULL DEFAULT '__local__', "
            "data JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "idempotency_key TEXT, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_workflows_workspace_idx "
            "ON aux_workflows (tenant, workspace_id)",
        ],
    ),
    # v5 (K5): the remaining auxiliary stores' Postgres mirrors — the Studio CRUD sidecars
    # (calendar / cookbook / notes / tasks / notify) plus the long-term memory store. With
    # this migration a ``postgres://`` deployment routes EVERY durable write to Postgres:
    # none of these fork to a local ``.himmy/*.db`` sidecar anymore. Each table carries the
    # ``tenant`` discriminator the rest of the aux surface uses (``"local"`` for these shared,
    # single-surface singletons), so the schema is uniform with K3/K4 and a future per-tenant
    # split is a column change, not a table change. Bodies mirror the SQLite stores 1:1
    # (durability routing only; per-subject envelope-encryption is the later S3 phase). The
    # ``memory`` mirror is durability-routing ONLY — its O(N) recall + pgvector ANN index is a
    # separate roadmap line, NOT in this item.
    (
        5,
        "aux_store_mirrors_k5",
        [
            # --- K5: Studio Calendar (dated events) ---
            "CREATE TABLE IF NOT EXISTS aux_calendar_events ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "date TEXT NOT NULL, "
            "time TEXT, "
            "title TEXT NOT NULL, "
            "notes TEXT NOT NULL DEFAULT '', "
            "created_at TEXT NOT NULL, "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_calendar_events_date_idx "
            "ON aux_calendar_events (tenant, date)",
            # --- K5: Studio Cookbook (saved agent+prompt recipes) ---
            "CREATE TABLE IF NOT EXISTS aux_recipes ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "name TEXT NOT NULL, "
            "agent_path TEXT NOT NULL DEFAULT '', "
            "prompt TEXT NOT NULL DEFAULT '', "
            "notes TEXT NOT NULL DEFAULT '', "
            "created_at TEXT NOT NULL, "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_recipes_created_idx "
            "ON aux_recipes (tenant, created_at DESC)",
            # --- K5: Studio Notes (durable markdown, shared with the notes tool pack) ---
            "CREATE TABLE IF NOT EXISTS aux_notes ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "title TEXT NOT NULL DEFAULT '', "
            "body TEXT NOT NULL DEFAULT '', "
            "updated_at TEXT NOT NULL, "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_notes_updated_idx "
            "ON aux_notes (tenant, updated_at DESC)",
            # --- K5: Studio Tasks (shared with the tasks tool pack) ---
            "CREATE TABLE IF NOT EXISTS aux_tasks ("
            "tenant TEXT NOT NULL, "
            "id TEXT NOT NULL, "
            "title TEXT NOT NULL, "
            "done BOOLEAN NOT NULL DEFAULT FALSE, "
            "due TEXT, "
            "created_at TEXT NOT NULL, "
            "PRIMARY KEY (tenant, id))",
            "CREATE INDEX IF NOT EXISTS aux_tasks_order_idx "
            "ON aux_tasks (tenant, done, created_at DESC)",
            # --- K5: Studio Notifications (the durable bell mirror + its one setting) ---
            # ``id`` is a per-tenant monotonic integer matching the SQLite mirror's ring id.
            "CREATE TABLE IF NOT EXISTS aux_notifications ("
            "tenant TEXT NOT NULL, "
            "id BIGINT NOT NULL, "
            "kind TEXT NOT NULL, "
            "title TEXT NOT NULL, "
            "body TEXT NOT NULL DEFAULT '', "
            "link TEXT NOT NULL DEFAULT '', "
            "created_at TEXT NOT NULL, "
            "read BOOLEAN NOT NULL DEFAULT FALSE, "
            "PRIMARY KEY (tenant, id))",
            "CREATE TABLE IF NOT EXISTS aux_notify_settings ("
            "tenant TEXT NOT NULL, "
            "key TEXT NOT NULL, "
            "value TEXT NOT NULL, "
            "PRIMARY KEY (tenant, key))",
            # --- K5: long-term memory facts + typed graph links ---
            "CREATE TABLE IF NOT EXISTS aux_memories ("
            "tenant TEXT NOT NULL, "
            "memory_id TEXT NOT NULL, "
            "subject_id TEXT NOT NULL, "
            "kind TEXT NOT NULL, "
            "text TEXT NOT NULL, "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "created_at TEXT NOT NULL, "
            "tier TEXT NOT NULL DEFAULT 'recall', "
            "valid_from TEXT NOT NULL DEFAULT '', "
            "valid_to TEXT, "
            "superseded_by TEXT, "
            "confidence DOUBLE PRECISION NOT NULL DEFAULT 1.0, "
            "source TEXT NOT NULL DEFAULT 'user', "
            "stable_key TEXT, "
            "PRIMARY KEY (tenant, memory_id))",
            "CREATE INDEX IF NOT EXISTS aux_memories_subject_idx "
            "ON aux_memories (tenant, subject_id)",
            "CREATE TABLE IF NOT EXISTS aux_memory_links ("
            "tenant TEXT NOT NULL, "
            "link_id TEXT NOT NULL, "
            "from_memory_id TEXT NOT NULL, "
            "to_memory_id TEXT NOT NULL, "
            "relation TEXT NOT NULL DEFAULT 'relates_to', "
            "metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
            "created_at TEXT NOT NULL, "
            "PRIMARY KEY (tenant, link_id))",
            "CREATE INDEX IF NOT EXISTS aux_memory_links_from_idx "
            "ON aux_memory_links (tenant, from_memory_id)",
            "CREATE INDEX IF NOT EXISTS aux_memory_links_to_idx "
            "ON aux_memory_links (tenant, to_memory_id)",
        ],
    ),
    # v6 (S3): the data-subject linkage column on the conversation mirror, so
    # right-to-erasure (S4 SubjectReachMap.delete_by_subject) can find + hard-delete a
    # subject's transcripts on the Postgres path too. Mirrors the SQLite ConversationStore
    # ``subject_id`` ADD COLUMN + index, keeping the two backends in lockstep (the K2
    # parity guard enforces it). ``IF NOT EXISTS`` makes a fresh-vs-legacy run converge.
    (
        6,
        "aux_conversations_subject_id",
        [
            "ALTER TABLE aux_conversations "
            "ADD COLUMN IF NOT EXISTS subject_id TEXT",
            "CREATE INDEX IF NOT EXISTS aux_conversations_subject_idx "
            "ON aux_conversations (tenant, subject_id)",
        ],
    ),
    # v7 (Q1): the leased run work-queue surface on ``runs`` + the durable inbound
    # ``trigger_dedup`` table — the Postgres lockstep of the SQLite v5 migration (the K2
    # parity guard enforces it). The eight queue columns are explicit TYPED columns (the
    # Postgres ``runs`` table is column-exploded, not a JSON blob), so the hand-written
    # ``_RUN_UPSERT_BY_ID`` + ``_row_to_run`` are extended to round-trip them. ``input_blob``
    # is the Q0 recoverable launch-input envelope (TEXT; encrypted at rest when a cipher is
    # configured). Backfill keeps every legacy QUEUED row immediately claimable (attempt 0,
    # ready at created_at). The two partial-ordered indexes give the Q2 claim CAS its
    # (status, next_attempt_at) seek and the reaper its (status, lease_expires_at) seek;
    # ``SKIP LOCKED`` rides the same index. ``IF NOT EXISTS`` makes fresh-vs-legacy converge.
    (
        7,
        "run_queue_and_trigger_dedup",
        [
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS owner_id TEXT",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS attempt INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS max_attempts INTEGER NOT NULL "
            "DEFAULT 1",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS next_attempt_at TIMESTAMPTZ",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS last_error TEXT",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS lane_key TEXT",
            "ALTER TABLE runs ADD COLUMN IF NOT EXISTS input_blob TEXT",
            "UPDATE runs SET next_attempt_at = created_at WHERE next_attempt_at IS NULL",
            "CREATE INDEX IF NOT EXISTS runs_status_next_attempt_idx "
            "ON runs (status, next_attempt_at)",
            "CREATE INDEX IF NOT EXISTS runs_status_lease_idx "
            "ON runs (status, lease_expires_at)",
            "CREATE INDEX IF NOT EXISTS runs_lane_key_idx ON runs (lane_key)",
            "CREATE TABLE IF NOT EXISTS trigger_dedup ("
            "scope TEXT NOT NULL, "
            "key TEXT NOT NULL, "
            "result TEXT, "
            "expires_at TIMESTAMPTZ, "
            "created_at TIMESTAMPTZ NOT NULL DEFAULT now(), "
            "PRIMARY KEY (scope, key))",
            "CREATE INDEX IF NOT EXISTS trigger_dedup_expires_idx "
            "ON trigger_dedup (expires_at)",
        ],
    ),
    # v8 (P1): bring run_events to parity with the SQLite backend for the
    # self-learning reputation miner. Two gaps are closed: (1) a denormalised
    # ``tool_name`` column captured pre-encryption on append (mirroring SQLite's
    # sqlite.py:543-544) so a ``WHERE tool_name=?`` filter works even when payloads are
    # encrypted at rest, with a composite ``(event_type, tool_name)`` index for the
    # hot-path conjunction read; and (2) a ``seq BIGSERIAL`` insertion-order column so
    # ``ORDER BY seq`` is deterministic — Postgres previously ordered by ``timestamp``,
    # which ties ambiguously for sub-ms tool events on one run. Adding a BIGSERIAL
    # column backfills existing rows in physical (insertion) order. Existing rows'
    # ``tool_name`` backfills best-effort from the JSONB payload (plaintext only; an
    # encrypted payload's nested tool_name is unrecoverable, exactly as SQLite's v2
    # backfill is bounded to unencrypted databases).
    #
    # OPERATIONAL NOTE: ``ADD COLUMN ... BIGSERIAL`` has a non-constant (sequence) default,
    # so Postgres takes an ACCESS EXCLUSIVE lock and rewrites the whole table — on a large
    # long-lived ``run_events`` audit stream this blocks reads/writes for the duration.
    # Run this upgrade in a maintenance window for large tables. For a true online upgrade,
    # prefer: ADD COLUMN seq BIGINT (nullable, fast) → backfill in batches → attach an
    # identity/sequence → CREATE INDEX CONCURRENTLY, rather than one BIGSERIAL statement.
    (
        8,
        "run_events_tool_name_and_seq",
        [
            "ALTER TABLE run_events ADD COLUMN IF NOT EXISTS tool_name TEXT",
            "ALTER TABLE run_events ADD COLUMN IF NOT EXISTS seq BIGSERIAL",
            "CREATE INDEX IF NOT EXISTS run_events_seq_idx ON run_events (seq)",
            "CREATE INDEX IF NOT EXISTS run_events_type_tool_idx "
            "ON run_events (event_type, tool_name)",
            "UPDATE run_events SET tool_name = payload->>'tool_name' "
            "WHERE tool_name IS NULL AND payload ? 'tool_name'",
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


def _row_to_run(row: Any, *, cipher: StorePayloadCipher | None = None) -> RunRecord:
    """Map a runs row to a RunRecord (timestamps normalised to ISO).

    The Q1 leased-queue TIMESTAMPTZ columns (lease_expires_at / heartbeat_at /
    next_attempt_at) are normalised to ISO strings exactly like created_at/updated_at. When a
    cipher is configured the Q0 ``input_blob`` envelope is decrypted back to cleartext
    (idempotent on plaintext / legacy rows).
    """
    data = dict(row)
    data["created_at"] = _iso(data.get("created_at"))
    data["updated_at"] = _iso(data.get("updated_at"))
    for _ts_col in ("lease_expires_at", "heartbeat_at", "next_attempt_at"):
        if _ts_col in data:
            data[_ts_col] = _iso(data.get(_ts_col))
    data["metadata"] = data.get("metadata") or {}
    structured = data.get("output_structured")
    if isinstance(structured, dict) and set(structured.keys()) == {"value"}:
        data["output_structured"] = structured["value"]
    blob = data.get("input_blob")
    if cipher is not None and isinstance(blob, str) and blob:
        data["input_blob"] = cipher.decrypt_run_input(
            blob, run_id=str(data.get("run_id", ""))
        )
    return RunRecord.model_validate(data)


def _row_to_agent_def(row: Any) -> AgentDefRecord:
    """Map an agent_defs row's JSONB ``payload`` to an :class:`AgentDefRecord`.

    The pooled jsonb codec decodes ``payload`` to a dict already, so the model is
    rehydrated straight from it (the full record, incl. the sanitized AgentSpec).
    """
    return AgentDefRecord.model_validate(row["payload"])


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
        """Append a run event (EventSink role); the payload is encrypted at rest.

        ``tool_name`` is denormalised into an indexed column captured from the live
        event BEFORE encryption (mirroring the SQLite backend), so the learning
        reputation miner's ``WHERE tool_name=?`` filter works at parity even when the
        payload is encrypted at rest.
        """
        pool = self._require_pool()
        plain = dict(event.payload or {})
        tool_name = plain.get("tool_name")
        payload = plain
        if self._cipher is not None:
            payload = self._cipher.encrypt_event_payload(
                plain, event_id=event.event_id
            )
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO run_events
                    (event_id, event_type, trace_id, thread_id, agent_id,
                     request_id, tool_call_id, tool_name, latency_ms, cost, payload,
                     error, timestamp)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (event_id) DO NOTHING
                """,
                event.event_id,
                event.event_type.value,
                event.trace_id,
                event.thread_id,
                event.agent_id,
                event.request_id,
                event.tool_call_id,
                tool_name,
                event.latency_ms,
                event.cost,
                payload,
                event.error,
                event.timestamp,
            )

    async def list_events(
        self,
        thread_id: str | None = None,
        trace_id: str | None = None,
        *,
        event_type: Any = None,
        tool_name: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[RunEvent]:
        """List events filtered by thread/trace/event_type/tool_name (insertion order).

        Orders by the ``seq`` BIGSERIAL insertion-order column (not ``timestamp``) so
        the result is deterministic even for sub-ms ties — at parity with the SQLite
        backend's ``ORDER BY seq``. ``limit`` bounds the set and ``newest_first`` flips
        the order so the learning miner takes the most recent N.
        """
        pool = self._require_pool()
        clauses: list[str] = []
        params: list[Any] = []
        if thread_id is not None:
            params.append(thread_id)
            clauses.append(f"thread_id = ${len(params)}")
        if trace_id is not None:
            params.append(trace_id)
            clauses.append(f"trace_id = ${len(params)}")
        want_type = normalize_event_type(event_type)
        if want_type is not None:
            params.append(want_type)
            clauses.append(f"event_type = ${len(params)}")
        if tool_name is not None:
            params.append(tool_name)
            clauses.append(f"tool_name = ${len(params)}")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        order = "DESC" if newest_first else "ASC"
        tail = ""
        if limit is not None:
            params.append(int(limit))
            tail = f" LIMIT ${len(params)}"
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"SELECT * FROM run_events{where} ORDER BY seq {order}{tail}", *params
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
                     created_at, updated_at, owner_id, lease_expires_at, heartbeat_at,
                     attempt, max_attempts, next_attempt_at, last_error, lane_key,
                     input_blob)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13,
                        $14, $15, $16, $17, $18, $19, $20, $21, $22, $23, $24, $25,
                        $26)
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
        return _row_to_run(existing, cipher=self._cipher), False

    async def claim_run_for_resume(
        self, run_id: str, *, workspace_id: str
    ) -> bool:
        """Atomically claim an AWAITING_APPROVAL run for resume (True iff we won).

        The run-level mirror of the member checkpoint claim: a single conditional
        ``UPDATE ... WHERE run_id = $1 AND workspace_id = $2 AND status = 'AWAITING_APPROVAL'``
        flips the status to ``RESOLVING`` and ``RETURNING run_id`` proves the winner. Row-
        level locking serializes two concurrent resumes (even across pool connections /
        processes): the first matches ``AWAITING_APPROVAL`` and flips it; the second now
        sees ``RESOLVING`` and matches nothing, so a SECOND concurrent approve loses the
        CAS and is refused BEFORE any resume launches (closing the orchestration double-
        advance race). Only ``AWAITING_APPROVAL`` is claimable; a stale ``RESOLVING`` (a
        crashed resume) is re-driven by recovery, not re-claimed by a fresh approve.
        """
        from himmy.core.ids import utc_now_iso

        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE runs
                SET status = $3, updated_at = $4
                WHERE run_id = $1 AND workspace_id = $2 AND status = $5
                RETURNING run_id
                """,
                run_id,
                workspace_id,
                RunStatus.RESOLVING.value,
                utc_now_iso(),
                RunStatus.AWAITING_APPROVAL.value,
            )
        return row is not None

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

        The structural twin of :meth:`claim_run_for_resume`, scaled to many concurrent
        workers: ``SELECT … FOR UPDATE SKIP LOCKED`` grabs the oldest READY row (``QUEUED`` and
        ``next_attempt_at <= now``) WITHOUT blocking on a row another worker already locked, so
        N workers each claim a DISTINCT run with no contention — the Postgres-native at-most-
        once dispatch. The selected row is flipped to ``RUNNING`` with owner/lease/heartbeat
        stamped and ``attempt`` incremented, in the SAME transaction. ``lanes`` filters on
        ``lane_key`` (Q2 lane keying) so a dispatcher whose LOCAL probe is down passes only the
        cloud/default lanes and never blocks claiming cloud runs.
        """
        now_iso = now or utc_now_iso()
        lease_until = _iso_plus_seconds(now_iso, lease_seconds)
        if lanes is not None and not lanes:
            return None
        lane_clause = ""
        params: list[Any] = [RunStatus.QUEUED.value, now_iso]
        if lanes is not None:
            params.append(list(lanes))
            lane_clause = f" AND lane_key = ANY(${len(params)}::text[])"
        pool = self._require_pool()
        async with pool.acquire() as conn, conn.transaction():
            target = await conn.fetchrow(
                "SELECT run_id FROM runs "
                "WHERE status = $1 AND (next_attempt_at IS NULL OR next_attempt_at <= $2)"
                f"{lane_clause} "
                "ORDER BY next_attempt_at ASC NULLS FIRST, created_at ASC "
                "FOR UPDATE SKIP LOCKED LIMIT 1",
                *params,
            )
            if target is None:
                return None
            row = await conn.fetchrow(
                """
                UPDATE runs
                SET status = $2, owner_id = $3, lease_expires_at = $4,
                    heartbeat_at = $5, attempt = attempt + 1, updated_at = $5
                WHERE run_id = $1
                RETURNING *
                """,
                target["run_id"],
                RunStatus.RUNNING.value,
                owner_id,
                lease_until,
                now_iso,
            )
        return _row_to_run(row, cipher=self._cipher) if row is not None else None

    async def renew_lease(
        self,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
        *,
        now: str | None = None,
    ) -> bool:
        """Extend a RUNNING run's lease iff ``owner_id`` still holds it (True iff renewed)."""
        now_iso = now or utc_now_iso()
        lease_until = _iso_plus_seconds(now_iso, lease_seconds)
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                UPDATE runs
                SET lease_expires_at = $3, heartbeat_at = $4, updated_at = $4
                WHERE run_id = $1 AND owner_id = $2 AND status = $5
                RETURNING run_id
                """,
                run_id,
                owner_id,
                lease_until,
                now_iso,
                RunStatus.RUNNING.value,
            )
        return row is not None

    async def requeue_expired_leases(
        self, *, now: str | None = None, lanes: list[str] | None = None
    ) -> list[str]:
        """Re-queue RUNNING runs whose lease expired; return the re-queued run_ids.

        ONLY ``status=RUNNING`` runs with ``lease_expires_at <= now`` are touched: a live peer
        renewing its lease has a future deadline and is NEVER reaped, and AWAITING_APPROVAL/
        RESOLVING/QUEUED/terminal runs are excluded by the status predicate (the HITL
        exclusions). Each resets to ``QUEUED`` with owner/lease cleared and ``next_attempt_at``
        set to ``now`` so it is immediately claimable. ``attempt`` is left as-is.
        """
        now_iso = now or utc_now_iso()
        if lanes is not None and not lanes:
            return []
        lane_clause = ""
        params: list[Any] = [
            RunStatus.QUEUED.value,
            now_iso,
            RunStatus.RUNNING.value,
        ]
        if lanes is not None:
            params.append(list(lanes))
            lane_clause = f" AND lane_key = ANY(${len(params)}::text[])"
        pool = self._require_pool()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "UPDATE runs SET status = $1, owner_id = NULL, lease_expires_at = NULL, "
                "next_attempt_at = $2, updated_at = $2 "
                "WHERE status = $3 AND lease_expires_at IS NOT NULL "
                f"AND lease_expires_at <= $2{lane_clause} "
                "RETURNING run_id",
                *params,
            )
        return [r["run_id"] for r in rows]

    async def redrive_run(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        now: str | None = None,
    ) -> bool:
        """Reset a PARKED/FAILED run back to QUEUED for another attempt (True iff redriven).

        Zeros ``attempt`` and clears ``last_error``/``error``/``owner_id``/``lease_expires_at``,
        setting ``next_attempt_at`` to ``now`` so it is immediately claimable. Only ``PARKED``
        and ``FAILED`` are redrivable; ``workspace_id`` scopes it (tenant safety).
        """
        now_iso = now or utc_now_iso()
        ws_clause = ""
        params: list[Any] = [run_id, RunStatus.QUEUED.value, now_iso]
        if workspace_id is not None:
            params.append(workspace_id)
            ws_clause = f" AND workspace_id = ${len(params)}"
        params.extend([RunStatus.PARKED.value, RunStatus.FAILED.value])
        status_clause = f"status IN (${len(params) - 1}, ${len(params)})"
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "UPDATE runs SET status = $2, attempt = 0, owner_id = NULL, "
                "lease_expires_at = NULL, last_error = NULL, error = NULL, "
                "next_attempt_at = $3, updated_at = $3 "
                f"WHERE run_id = $1{ws_clause} AND {status_clause} "
                "RETURNING run_id",
                *params,
            )
        return row is not None

    #: The 17 base columns + the 9 Q0/Q1 queue columns (owner_id … input_blob), so a run
    #: upsert round-trips the full leased-queue surface. The base columns keep their original
    #: positions ($1..$17) for an unchanged read; the queue columns are appended ($18..$26).
    _RUN_UPSERT_BY_ID = """
        INSERT INTO runs
            (run_id, workspace_id, subject_id, task_id, thread_id, snapshot_id,
             persona_name, model_key, idempotency_key, status, output_text,
             output_structured, error, trace_id, metadata, created_at, updated_at,
             owner_id, lease_expires_at, heartbeat_at, attempt, max_attempts,
             next_attempt_at, last_error, lane_key, input_blob)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
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
            updated_at = EXCLUDED.updated_at,
            owner_id = EXCLUDED.owner_id,
            lease_expires_at = EXCLUDED.lease_expires_at,
            heartbeat_at = EXCLUDED.heartbeat_at,
            attempt = EXCLUDED.attempt,
            max_attempts = EXCLUDED.max_attempts,
            next_attempt_at = EXCLUDED.next_attempt_at,
            last_error = EXCLUDED.last_error,
            lane_key = EXCLUDED.lane_key,
            input_blob = EXCLUDED.input_blob
    """

    def _run_params(self, run: RunRecord) -> list[Any]:
        """Positional params for the runs upsert (output_structured as jsonb).

        The Q0 ``input_blob`` is encrypted at rest (when a cipher is configured) bound to the
        ``run_id`` as AAD — the recoverable launch input carries the prompt and gets the same
        envelope as chat content / event payloads on this backend.
        """
        structured = run.output_structured
        # output_structured is a jsonb column: the codec encodes dict/list; wrap
        # scalars/None so the bind type is unambiguous.
        if structured is not None and not isinstance(structured, (dict, list)):
            structured = {"value": structured}
        input_blob = run.input_blob
        if self._cipher is not None and input_blob:
            input_blob = self._cipher.encrypt_run_input(input_blob, run_id=run.run_id)
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
            run.owner_id,
            run.lease_expires_at,
            run.heartbeat_at,
            run.attempt,
            run.max_attempts,
            run.next_attempt_at or run.created_at,
            run.last_error,
            run.lane_key,
            input_blob,
        ]

    async def get_run(self, run_id: str) -> RunRecord | None:
        """Return a run record by id, or None."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow("SELECT * FROM runs WHERE run_id = $1", run_id)
        return _row_to_run(row, cipher=self._cipher) if row else None

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
        return [_row_to_run(r, cipher=self._cipher) for r in rows]

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
        return _row_to_run(row, cipher=self._cipher) if row else None


class PostgresAgentDefStore(_PgStoreBase):
    """Postgres-backed workspace-scoped stored agent definitions (T2e)."""

    async def save_agent_def(self, record: AgentDefRecord) -> AgentDefRecord:
        """Upsert a stored agent definition keyed by ``agent_id``."""
        from himmy.core.ids import utc_now_iso

        record.updated_at = utc_now_iso()
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(self._UPSERT_BY_ID, *self._params(record))
        return record

    async def save_agent_def_if_absent(
        self, record: AgentDefRecord
    ) -> tuple[AgentDefRecord, bool]:
        """Atomically create an agent def unless its idempotency key already exists.

        The partial UNIQUE index ``agent_defs_idempotency_idx`` closes the race via
        ``INSERT ... ON CONFLICT (workspace_id, idempotency_key) DO NOTHING``: exactly
        one concurrent writer wins; on conflict the existing row is read back.
        """
        from himmy.core.ids import utc_now_iso

        record.updated_at = utc_now_iso()
        pool = self._require_pool()
        if record.idempotency_key is None:
            async with pool.acquire() as conn:
                await conn.execute(self._UPSERT_BY_ID, *self._params(record))
            return record, True

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO agent_defs
                    (agent_id, workspace_id, name, idempotency_key, payload,
                     created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (workspace_id, idempotency_key)
                    WHERE idempotency_key IS NOT NULL
                    DO NOTHING
                RETURNING agent_id
                """,
                *self._params(record),
            )
            if row is not None:
                return record, True
            existing = await conn.fetchrow(
                "SELECT payload FROM agent_defs "
                "WHERE workspace_id = $1 AND idempotency_key = $2",
                record.workspace_id,
                record.idempotency_key,
            )
        return _row_to_agent_def(existing), False

    _UPSERT_BY_ID = """
        INSERT INTO agent_defs
            (agent_id, workspace_id, name, idempotency_key, payload,
             created_at, updated_at)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (agent_id) DO UPDATE SET
            workspace_id = EXCLUDED.workspace_id,
            name = EXCLUDED.name,
            idempotency_key = EXCLUDED.idempotency_key,
            payload = EXCLUDED.payload,
            updated_at = EXCLUDED.updated_at
    """

    @staticmethod
    def _params(record: AgentDefRecord) -> list[Any]:
        """Positional params for the agent_defs upsert (payload as jsonb)."""
        return [
            record.agent_id,
            record.workspace_id,
            record.name,
            record.idempotency_key,
            record.model_dump(mode="json"),
            record.created_at,
            record.updated_at,
        ]

    async def get_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> AgentDefRecord | None:
        """Return a stored agent def by id, tenant-scoped (out-of-workspace → None)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if workspace_id is None:
                row = await conn.fetchrow(
                    "SELECT payload FROM agent_defs WHERE agent_id = $1", agent_id
                )
            else:
                row = await conn.fetchrow(
                    "SELECT payload FROM agent_defs "
                    "WHERE agent_id = $1 AND workspace_id = $2",
                    agent_id,
                    workspace_id,
                )
        return _row_to_agent_def(row) if row else None

    async def list_agent_defs(
        self, *, workspace_id: str | None = None
    ) -> list[AgentDefRecord]:
        """List stored agent defs for a workspace (created_at asc)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if workspace_id is None:
                rows = await conn.fetch(
                    "SELECT payload FROM agent_defs ORDER BY created_at ASC"
                )
            else:
                rows = await conn.fetch(
                    "SELECT payload FROM agent_defs WHERE workspace_id = $1 "
                    "ORDER BY created_at ASC",
                    workspace_id,
                )
        return [_row_to_agent_def(r) for r in rows]

    async def delete_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> bool:
        """Delete a stored agent def, tenant-scoped. Returns True iff a row was removed."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            if workspace_id is None:
                result = await conn.execute(
                    "DELETE FROM agent_defs WHERE agent_id = $1", agent_id
                )
            else:
                result = await conn.execute(
                    "DELETE FROM agent_defs WHERE agent_id = $1 AND workspace_id = $2",
                    agent_id,
                    workspace_id,
                )
        # asyncpg returns a command tag e.g. "DELETE 1" / "DELETE 0".
        return str(result).rsplit(" ", 1)[-1] != "0"


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


class PostgresTriggerDedupStore(_PgStoreBase):
    """Postgres-backed durable inbound-trigger dedup (the ``trigger_dedup`` TTL-CAS, Q4).

    The cluster-scalable twin of the SQLite/in-memory dedup: the atomic claim rides
    ``INSERT … ON CONFLICT DO UPDATE … WHERE`` so the CAS is decided server-side under the
    primary-key lock — N webhook replicas racing the same delivery id produce exactly one
    :data:`CLAIM_WON`. ``result IS NULL`` marks an in-flight lease; a string marks a
    COMPLETED row. Timestamps are bound as ISO strings via the text codec the pool installs
    (see :func:`_register_codecs`).
    """

    async def dedup_try_claim(
        self,
        scope: str,
        key: str,
        *,
        lease_seconds: float,
        now: str | None = None,
    ) -> DedupClaim:
        """Atomically claim ``(scope, key)`` for execution, or report a duplicate (Q4)."""
        from himmy.services.storage.trigger_dedup import (
            CLAIM_DONE,
            CLAIM_IN_FLIGHT,
            CLAIM_WON,
            DedupClaim,
        )

        now_iso = now or _utc_now()
        lease_until = _iso_plus_seconds(now_iso, lease_seconds)
        pool = self._require_pool()
        async with pool.acquire() as conn, conn.transaction():
            # CAS: insert a fresh in-flight lease, but if a row already exists, only TAKE it
            # OVER when it is EXPIRED (a crashed worker's lapsed lease or a stale COMPLETED
            # row). A live row is left untouched and the RETURNING clause is empty.
            won = await conn.fetchrow(
                """
                INSERT INTO trigger_dedup (scope, key, result, expires_at, created_at)
                VALUES ($1, $2, NULL, $3, $4)
                ON CONFLICT (scope, key) DO UPDATE
                    SET result = NULL, expires_at = EXCLUDED.expires_at,
                        created_at = EXCLUDED.created_at
                    WHERE trigger_dedup.expires_at IS NOT NULL
                      AND trigger_dedup.expires_at <= $4
                RETURNING scope
                """,
                scope,
                key,
                lease_until,
                now_iso,
            )
            if won is not None:
                return DedupClaim(CLAIM_WON)
            # The CONFLICT predicate failed -> a LIVE row exists; read it to classify.
            existing = await conn.fetchrow(
                "SELECT result FROM trigger_dedup WHERE scope = $1 AND key = $2",
                scope,
                key,
            )
        if existing is not None and existing["result"] is not None:
            return DedupClaim(CLAIM_DONE, result=existing["result"])
        return DedupClaim(CLAIM_IN_FLIGHT)

    async def dedup_complete(
        self,
        scope: str,
        key: str,
        *,
        result: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> None:
        """Upgrade the in-flight row to COMPLETED with ``result`` + the real TTL."""
        now_iso = now or _utc_now()
        expires_at = _iso_plus_seconds(now_iso, ttl_seconds)
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO trigger_dedup (scope, key, result, expires_at, created_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (scope, key) DO UPDATE
                    SET result = EXCLUDED.result, expires_at = EXCLUDED.expires_at
                """,
                scope,
                key,
                result,
                expires_at,
                now_iso,
            )

    async def dedup_release(
        self, scope: str, key: str, *, now: str | None = None
    ) -> None:
        """Drop a won-but-failed in-flight row so a redelivery re-runs (only if in-flight)."""
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM trigger_dedup "
                "WHERE scope = $1 AND key = $2 AND result IS NULL",
                scope,
                key,
            )

    async def dedup_sweep(self, *, now: str | None = None) -> int:
        """Delete expired dedup rows (lazy GC); return the count removed."""
        now_iso = now or _utc_now()
        pool = self._require_pool()
        async with pool.acquire() as conn:
            status = await conn.execute(
                "DELETE FROM trigger_dedup "
                "WHERE expires_at IS NOT NULL AND expires_at <= $1",
                now_iso,
            )
        # asyncpg returns a status like "DELETE 3"; parse the trailing count.
        try:
            return int(str(status).rsplit(" ", 1)[-1])
        except (ValueError, IndexError):  # pragma: no cover - defensive
            return 0


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
        self._run_store = PostgresRunStore(self._require_pool, cipher=self._cipher)
        self._agent_def_store = PostgresAgentDefStore(self._require_pool)
        self._recommendation_store = PostgresRecommendationStore(self._require_pool)
        self._evaluation_store = PostgresEvaluationStore(self._require_pool)
        self._orchestration_store = PostgresOrchestrationStore(self._require_pool)
        self._trigger_dedup_store = PostgresTriggerDedupStore(self._require_pool)

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

    async def ping(self, *, timeout: float = 2.0) -> bool:
        """Timeboxed ``SELECT 1`` over the pool — True iff the database is reachable (G4).

        A PUBLIC accessor reusing the service's own pool (no fresh ``asyncpg.connect``,
        no reach into the private ``_require_pool``) so ``/readyz`` can probe liveness of
        the durable backend on every readiness check. Any failure (pool closed, DB down,
        timeout) returns ``False`` rather than raising, so the caller maps it to a 503.
        """
        import asyncio

        pool = self._pool
        if pool is None:
            return False
        try:
            async with pool.acquire() as conn:
                await asyncio.wait_for(conn.execute("SELECT 1"), timeout=timeout)
            return True
        except Exception:  # noqa: BLE001 - readiness probe: never raise, report down
            return False

    async def applied_migration_versions(self, *, timeout: float = 2.0) -> list[int]:
        """The migration versions recorded in ``schema_migrations`` (G4).

        PUBLIC, pool-backed (reuses the :meth:`migrate` / ``commands.py`` query shape
        ``SELECT version FROM schema_migrations``) so ``/readyz`` can compare the
        APPLIED schema against the code's ``max(STORAGE_MIGRATIONS)`` and refuse readiness
        when a pod is running ahead of its database (pending migrations). Returns ``[]``
        on any error so the caller can treat "can't read migrations" as not-ready.
        """
        import asyncio

        pool = self._pool
        if pool is None:
            return []
        try:
            async with pool.acquire() as conn:
                rows = await asyncio.wait_for(
                    conn.fetch("SELECT version FROM schema_migrations"), timeout=timeout
                )
            return sorted(int(row["version"]) for row in rows)
        except Exception:  # noqa: BLE001 - readiness probe: never raise, report empty
            return []

    @staticmethod
    def code_migration_version() -> int:
        """The highest migration version shipped in this code (``max`` of the chain) (G4)."""
        return max(version for version, _, _ in STORAGE_MIGRATIONS)

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
        self,
        thread_id: str | None = None,
        trace_id: str | None = None,
        *,
        event_type: Any = None,
        tool_name: str | None = None,
        limit: int | None = None,
        newest_first: bool = False,
    ) -> list[RunEvent]:
        """List events filtered by thread/trace/event_type/tool_name (insertion order)."""
        return await self._event_log.list_events(
            thread_id,
            trace_id,
            event_type=event_type,
            tool_name=tool_name,
            limit=limit,
            newest_first=newest_first,
        )

    async def delete_by_subject(self, subject_id: str) -> int:
        """Hard-DELETE a subject's chat_threads + run_events (S4 right-to-erasure).

        The Postgres mirror of :meth:`SqliteStorageService.delete_by_subject`: these tables
        are encrypted at rest under the store-WIDE cipher (not the per-subject key), so a
        crypto-shred alone leaves them recoverable — erasure hard-deletes them. A subject's
        runs (``runs.subject_id``) name every ``thread_id`` / ``trace_id`` the runtime
        persisted, resolved here via the ``runs`` row's typed columns. Runs in ONE
        transaction so a mid-delete failure rolls back; idempotent. Returns rows removed.
        """
        pool = self._require_pool()
        async with pool.acquire() as conn, conn.transaction():
            rows = await conn.fetch(
                "SELECT thread_id, trace_id FROM runs WHERE subject_id = $1",
                subject_id,
            )
            thread_ids = [r["thread_id"] for r in rows if r["thread_id"]]
            trace_ids = [r["trace_id"] for r in rows if r["trace_id"]]
            deleted = 0
            if thread_ids:
                tag = await conn.execute(
                    "DELETE FROM chat_threads WHERE thread_id = ANY($1::text[])",
                    thread_ids,
                )
                deleted += int(str(tag).rsplit(" ", 1)[-1])
            if thread_ids or trace_ids:
                tag = await conn.execute(
                    "DELETE FROM run_events "
                    "WHERE thread_id = ANY($1::text[]) OR trace_id = ANY($2::text[])",
                    thread_ids,
                    trace_ids,
                )
                deleted += int(str(tag).rsplit(" ", 1)[-1])
        return deleted

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

    async def claim_run_for_resume(
        self, run_id: str, *, workspace_id: str
    ) -> bool:
        """Atomically claim an AWAITING_APPROVAL run for resume (True iff we won)."""
        return await self._run_store.claim_run_for_resume(
            run_id, workspace_id=workspace_id
        )

    async def claim_next_queued_run(
        self,
        owner_id: str,
        lease_seconds: float,
        *,
        lanes: list[str] | None = None,
        now: str | None = None,
    ) -> RunRecord | None:
        """Atomically claim the oldest ready QUEUED run for ``owner_id`` (Q2; or None)."""
        return await self._run_store.claim_next_queued_run(
            owner_id, lease_seconds, lanes=lanes, now=now
        )

    async def renew_lease(
        self,
        run_id: str,
        owner_id: str,
        lease_seconds: float,
        *,
        now: str | None = None,
    ) -> bool:
        """Extend a RUNNING run's lease iff ``owner_id`` still holds it (Q2)."""
        return await self._run_store.renew_lease(
            run_id, owner_id, lease_seconds, now=now
        )

    async def requeue_expired_leases(
        self, *, now: str | None = None, lanes: list[str] | None = None
    ) -> list[str]:
        """Re-queue RUNNING runs whose lease expired; return the re-queued ids (Q2)."""
        return await self._run_store.requeue_expired_leases(now=now, lanes=lanes)

    async def redrive_run(
        self,
        run_id: str,
        *,
        workspace_id: str | None = None,
        now: str | None = None,
    ) -> bool:
        """Reset a PARKED/FAILED run back to QUEUED for another attempt (Q2)."""
        return await self._run_store.redrive_run(
            run_id, workspace_id=workspace_id, now=now
        )

    # ------------------------------------------------------ inbound dedup (Q4)
    async def dedup_try_claim(
        self,
        scope: str,
        key: str,
        *,
        lease_seconds: float,
        now: str | None = None,
    ) -> DedupClaim:
        """Atomically claim ``(scope, key)`` for execution, or report a duplicate (Q4)."""
        return await self._trigger_dedup_store.dedup_try_claim(
            scope, key, lease_seconds=lease_seconds, now=now
        )

    async def dedup_complete(
        self,
        scope: str,
        key: str,
        *,
        result: str,
        ttl_seconds: float,
        now: str | None = None,
    ) -> None:
        """Upgrade a won in-flight dedup claim to COMPLETED with ``result`` + TTL (Q4)."""
        await self._trigger_dedup_store.dedup_complete(
            scope, key, result=result, ttl_seconds=ttl_seconds, now=now
        )

    async def dedup_release(
        self, scope: str, key: str, *, now: str | None = None
    ) -> None:
        """Drop a won-but-failed in-flight dedup claim so a redelivery re-runs (Q4)."""
        await self._trigger_dedup_store.dedup_release(scope, key, now=now)

    async def dedup_sweep(self, *, now: str | None = None) -> int:
        """Delete expired dedup rows (lazy GC); return the count removed (Q4)."""
        return await self._trigger_dedup_store.dedup_sweep(now=now)

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

    # ------------------------------------------------------------- agent defs (T2e)
    async def save_agent_def(self, record: AgentDefRecord) -> AgentDefRecord:
        """Upsert a stored agent definition keyed by ``agent_id``."""
        return await self._agent_def_store.save_agent_def(record)

    async def save_agent_def_if_absent(
        self, record: AgentDefRecord
    ) -> tuple[AgentDefRecord, bool]:
        """Atomically create an agent def unless its idempotency key already exists."""
        return await self._agent_def_store.save_agent_def_if_absent(record)

    async def get_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> AgentDefRecord | None:
        """Return a stored agent def by id, tenant-scoped (out-of-workspace → None)."""
        return await self._agent_def_store.get_agent_def(
            agent_id, workspace_id=workspace_id
        )

    async def list_agent_defs(
        self, *, workspace_id: str | None = None
    ) -> list[AgentDefRecord]:
        """List stored agent defs for a workspace."""
        return await self._agent_def_store.list_agent_defs(workspace_id=workspace_id)

    async def delete_agent_def(
        self, agent_id: str, *, workspace_id: str | None = None
    ) -> bool:
        """Delete a stored agent def, tenant-scoped. Returns True iff removed."""
        return await self._agent_def_store.delete_agent_def(
            agent_id, workspace_id=workspace_id
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
