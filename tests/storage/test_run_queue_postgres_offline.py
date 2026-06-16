"""Q1+Q2: Postgres leased-queue wiring, OFFLINE (no live database required).

There is no live Postgres in the unit lane (no ``asyncpg``, no DSN), so this module exercises
the parts of the Postgres path that DON'T need a connection: the committed migration declares
the queue columns + ``trigger_dedup``; the hand-written ``_RUN_UPSERT_BY_ID`` + ``_run_params``
+ ``_row_to_run`` round-trip all 26 columns (the reviewer must_fix — the Postgres ``runs`` is
column-exploded, so a forgotten column would silently drop queue state); and the Q0
``input_blob`` is encrypted on write / decrypted on read. The LIVE claim CAS (SKIP LOCKED) is
exercised in ``test_run_queue_postgres_live`` against a real DB when ``HIMMY_TEST_POSTGRES_DSN``
is set.
"""

from __future__ import annotations

from himmy.services.storage.at_rest import StorePayloadCipher
from himmy.services.storage.encryption import FieldEncryptor
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.postgres import (
    STORAGE_MIGRATIONS,
    PostgresRunStore,
    _row_to_run,
)

_QUEUE_COLUMNS = (
    "owner_id",
    "lease_expires_at",
    "heartbeat_at",
    "attempt",
    "max_attempts",
    "next_attempt_at",
    "last_error",
    "lane_key",
    "input_blob",
)


def _migration_text() -> str:
    """All STORAGE_MIGRATIONS statements joined (offline string inspection)."""
    return "\n".join(
        stmt for _v, _n, stmts in STORAGE_MIGRATIONS for stmt in stmts
    )


def test_migration_declares_queue_columns_and_dedup_table() -> None:
    text = _migration_text()
    for col in _QUEUE_COLUMNS:
        assert f"ADD COLUMN IF NOT EXISTS {col}" in text, col
    assert "CREATE TABLE IF NOT EXISTS trigger_dedup" in text
    # The two seek indexes the claim CAS + reaper depend on.
    assert "runs_status_next_attempt_idx" in text
    assert "runs_status_lease_idx" in text
    assert "runs_lane_key_idx" in text


def test_queue_migration_is_a_new_version_not_an_edit() -> None:
    """The queue columns ride a NEW migration version (v7), never an edit of a shipped one."""
    versions = [v for v, _n, _s in STORAGE_MIGRATIONS]
    assert versions == sorted(versions)
    # v7 is the queue migration; later migrations (e.g. v8 run_events parity) may follow,
    # so assert v7 is present rather than pinning it as the highest version forever.
    assert 7 in versions
    v7 = next(stmts for v, _n, stmts in STORAGE_MIGRATIONS if v == 7)
    joined = "\n".join(v7)
    assert "trigger_dedup" in joined
    assert "ADD COLUMN IF NOT EXISTS input_blob" in joined


def test_run_params_round_trips_all_columns() -> None:
    """``_run_params`` emits all 26 positional columns and ``_row_to_run`` reads them back."""
    store = PostgresRunStore(lambda: None, cipher=None)
    run = RunRecord(
        workspace_id="w",
        subject_id="s",
        status=RunStatus.RUNNING,
        owner_id="worker-A",
        lease_expires_at="2026-06-14T01:00:00+00:00",
        heartbeat_at="2026-06-14T00:30:00+00:00",
        attempt=3,
        max_attempts=5,
        next_attempt_at="2026-06-14T00:00:00+00:00",
        last_error="transient",
        lane_key="cloud",
        input_blob="PLAINTEXT-BLOB",
    )
    params = store._run_params(run)
    assert len(params) == 26  # 17 base + 9 queue

    # Reconstruct an asyncpg-style decoded row (codecs already parsed jsonb/timestamps).
    row = {
        "run_id": run.run_id,
        "workspace_id": "w",
        "subject_id": "s",
        "task_id": None,
        "thread_id": None,
        "snapshot_id": None,
        "persona_name": None,
        "model_key": None,
        "idempotency_key": None,
        "status": "RUNNING",
        "output_text": None,
        "output_structured": None,
        "error": None,
        "trace_id": None,
        "metadata": {},
        "created_at": "2026-06-14T00:00:00+00:00",
        "updated_at": "2026-06-14T00:00:00+00:00",
        "owner_id": "worker-A",
        "lease_expires_at": "2026-06-14T01:00:00+00:00",
        "heartbeat_at": "2026-06-14T00:30:00+00:00",
        "attempt": 3,
        "max_attempts": 5,
        "next_attempt_at": "2026-06-14T00:00:00+00:00",
        "last_error": "transient",
        "lane_key": "cloud",
        "input_blob": "PLAINTEXT-BLOB",
    }
    rec = _row_to_run(row, cipher=None)
    assert rec.owner_id == "worker-A"
    assert rec.attempt == 3
    assert rec.max_attempts == 5
    assert rec.lane_key == "cloud"
    assert rec.last_error == "transient"
    assert rec.lease_expires_at == "2026-06-14T01:00:00+00:00"
    assert rec.next_attempt_at == "2026-06-14T00:00:00+00:00"
    assert rec.input_blob == "PLAINTEXT-BLOB"


def test_input_blob_encrypted_in_params_and_decrypted_in_row() -> None:
    """With a cipher, ``_run_params`` encrypts ``input_blob`` and ``_row_to_run`` decrypts it."""
    cipher = StorePayloadCipher(FieldEncryptor.generate())
    store = PostgresRunStore(lambda: None, cipher=cipher)
    run = RunRecord(
        workspace_id="w",
        subject_id="s",
        status=RunStatus.QUEUED,
        input_blob="SECRET-INPUT",
    )
    params = store._run_params(run)
    enc_blob = params[-1]
    assert isinstance(enc_blob, str)
    assert enc_blob != "SECRET-INPUT"
    assert "SECRET-INPUT" not in enc_blob

    row = {
        "run_id": run.run_id,
        "workspace_id": "w",
        "subject_id": "s",
        "task_id": None,
        "thread_id": None,
        "snapshot_id": None,
        "persona_name": None,
        "model_key": None,
        "idempotency_key": None,
        "status": "QUEUED",
        "output_text": None,
        "output_structured": None,
        "error": None,
        "trace_id": None,
        "metadata": {},
        "created_at": "2026-06-14T00:00:00+00:00",
        "updated_at": "2026-06-14T00:00:00+00:00",
        "owner_id": None,
        "lease_expires_at": None,
        "heartbeat_at": None,
        "attempt": 0,
        "max_attempts": 1,
        "next_attempt_at": "2026-06-14T00:00:00+00:00",
        "last_error": None,
        "lane_key": None,
        "input_blob": enc_blob,
    }
    rec = _row_to_run(row, cipher=cipher)
    assert rec.input_blob == "SECRET-INPUT"


def test_run_store_exposes_queue_methods() -> None:
    """The facade + run store expose the Q2 surface (signature presence, offline)."""
    store = PostgresRunStore(lambda: None, cipher=None)
    for name in (
        "claim_next_queued_run",
        "renew_lease",
        "requeue_expired_leases",
        "redrive_run",
    ):
        assert callable(getattr(store, name))


def test_postgres_facade_exposes_dedup_surface() -> None:
    """Q4: the Postgres facade + the dedup store expose the trigger_dedup surface (offline)."""
    from himmy.services.storage.postgres import (
        PostgresStorageService,
        PostgresTriggerDedupStore,
    )

    facade = PostgresStorageService(pool=None)
    for name in (
        "dedup_try_claim",
        "dedup_complete",
        "dedup_release",
        "dedup_sweep",
    ):
        assert callable(getattr(facade, name)), name
    store = PostgresTriggerDedupStore(lambda: None)
    # The CAS uses ON CONFLICT ... DO UPDATE ... WHERE so the take-over is server-side.
    for name in (
        "dedup_try_claim",
        "dedup_complete",
        "dedup_release",
        "dedup_sweep",
    ):
        assert callable(getattr(store, name)), name


def test_q4_dedup_store_satisfies_protocol() -> None:
    """The in-memory + SQLite dedup stores satisfy the TriggerDedupStore protocol."""
    from himmy.services.storage.service import StorageService
    from himmy.services.storage.trigger_dedup import TriggerDedupStore

    assert isinstance(StorageService(), TriggerDedupStore)
