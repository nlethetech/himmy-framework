"""S6: OPTIONAL DB-resident hash chain — tamper-evidence from the DB alone.

:mod:`himmy.entities.integrity` already ships a full opt-in hash chain whose verification
needs a separately-held signed bundle. S6 makes that SAME chain DB-resident: each
``register`` records the predecessor's chain hash in the ``entity_records.prev_hash`` column
(added by the S1 spine migration), so an in-place row mutation is detectable from the
database ALONE — no bundle. It REUSES ``content_hash`` + ``_chain_step`` (no second
mechanism) and stays OFF by default (the column is NULL). These tests pin: off-by-default,
intact verification, in-place-mutation detection, resume-across-restart, and the must_fix —
a register rollback under ``register_buffer>1`` must reset the in-RAM prev pointer so it never
falsely reports tampering.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from himmy.core.errors import HimmyError
from himmy.entities.db_chain import (
    CHAIN_GENESIS_HASH,
    DbChainRow,
    chain_step_for,
    verify_db_chain,
)
from himmy.entities.records import EntityRecord
from himmy.entities.sqlite_registry import _DB_CHAIN_ENV, SqliteEntityRegistry


def _ev(i: int) -> EntityRecord:
    return EntityRecord.create(
        stable_id=f"evt-{i}", version=1, kind="security_event", payload={"seq": i}
    )


# ----------------------------------------------------------------- off by default


def test_chain_off_by_default_leaves_prev_hash_null(tmp_path: Path) -> None:
    reg = SqliteEntityRegistry(str(tmp_path / "spine.db"))
    assert reg.db_chain_enabled is False
    reg.register(_ev(0))
    rows = reg.db_chain_rows()
    assert all(r.prev_hash is None for r in rows)
    # An all-NULL set verifies trivially (nothing joined the chain) and reports unchained.
    result = verify_db_chain(rows)
    assert result.ok is True
    assert result.chained == 0
    assert result.unchained_record_ids
    reg.close()


def test_chain_env_flag_enables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_DB_CHAIN_ENV, "1")
    reg = SqliteEntityRegistry(str(tmp_path / "spine.db"))
    assert reg.db_chain_enabled is True
    reg.close()


# ------------------------------------------------------------- intact + tamper


def test_intact_db_chain_verifies_from_db_alone(tmp_path: Path) -> None:
    reg = SqliteEntityRegistry(str(tmp_path / "spine.db"), db_chain=True)
    for i in range(5):
        reg.register(_ev(i))
    result = verify_db_chain(reg.db_chain_rows())
    assert result.ok is True
    assert result.chained == 5
    assert result.broken_at_record_id is None
    reg.close()


def test_in_place_payload_mutation_detected_from_db_alone(tmp_path: Path) -> None:
    """The core S6 acceptance: mutate a row's payload directly in SQLite; the chain breaks."""
    path = str(tmp_path / "spine.db")
    reg = SqliteEntityRegistry(path, db_chain=True)
    recs = [_ev(i) for i in range(4)]
    for r in recs:
        reg.register(r)
    reg.close()

    # Tamper IN PLACE via raw SQLite (no bundle held anywhere).
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE entity_records SET payload = ? WHERE stable_id = ?",
        ('{"seq": 999}', "evt-1"),
    )
    conn.commit()
    conn.close()

    reg2 = SqliteEntityRegistry(path, db_chain=True)
    result = verify_db_chain(reg2.db_chain_rows())
    assert result.ok is False
    assert result.broken_at_record_id is not None
    assert "mutation" in result.reason or "unreachable" in result.reason
    reg2.close()


def test_head_deletion_detected(tmp_path: Path) -> None:
    path = str(tmp_path / "spine.db")
    reg = SqliteEntityRegistry(path, db_chain=True)
    recs = [_ev(i) for i in range(3)]
    for r in recs:
        reg.register(r)
    reg.close()

    # Delete the genesis-linked head row: nothing links to genesis any more.
    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM entity_records WHERE stable_id = ?", ("evt-0",))
    conn.commit()
    conn.close()

    reg2 = SqliteEntityRegistry(path, db_chain=True)
    result = verify_db_chain(reg2.db_chain_rows())
    assert result.ok is False
    assert "no chain head" in result.reason or "breaks" in result.reason
    reg2.close()


def test_forged_prev_hash_fork_detected() -> None:
    """Two rows declaring the same predecessor (a forged prev_hash) is flagged."""
    r0, r1 = _ev(0), _ev(1)
    # Both rows claim genesis as predecessor -> a fork.
    rows = [
        DbChainRow(record=r0, prev_hash=CHAIN_GENESIS_HASH),
        DbChainRow(record=r1, prev_hash=CHAIN_GENESIS_HASH),
    ]
    result = verify_db_chain(rows)
    assert result.ok is False
    assert "fork" in result.reason


# ------------------------------------------------------------- restart resume


def test_chain_resumes_across_restart(tmp_path: Path) -> None:
    path = str(tmp_path / "spine.db")
    reg = SqliteEntityRegistry(path, db_chain=True)
    for i in range(3):
        reg.register(_ev(i))
    reg.close()

    # A fresh registry recomputes the committed tip and appends to the SAME chain.
    reg2 = SqliteEntityRegistry(path, db_chain=True)
    reg2.register(_ev(3))
    result = verify_db_chain(reg2.db_chain_rows())
    assert result.ok is True
    assert result.chained == 4
    reg2.close()


def test_legacy_rows_written_before_chain_are_unchained_not_a_break(
    tmp_path: Path,
) -> None:
    """Rows written with the chain OFF (prev_hash NULL) are unchained, not a tamper signal."""
    path = str(tmp_path / "spine.db")
    # Write the first two rows with the chain off.
    off = SqliteEntityRegistry(path, db_chain=False)
    off.register(_ev(0))
    off.register(_ev(1))
    off.close()

    # Turn the chain on; new rows chain from genesis, old rows stay NULL.
    on = SqliteEntityRegistry(path, db_chain=True)
    on.register(_ev(2))
    on.register(_ev(3))
    result = verify_db_chain(on.db_chain_rows())
    assert result.ok is True
    assert result.chained == 2
    assert len(result.unchained_record_ids) == 2
    on.close()


# ------------------------------------------------- rollback resets prev pointer


def test_rollback_under_buffer_resets_chain_tip(tmp_path: Path) -> None:
    """S6 must_fix: a rollback under register_buffer>1 must not leave the tip ahead of disk.

    A discarded batch advances the in-RAM tip for rows that never commit; without a reset the
    NEXT committed row would chain off a phantom predecessor and the from-DB walk would falsely
    report tampering. The reset re-anchors the tip to the committed tail so the chain stays
    verifiable.
    """
    path = str(tmp_path / "spine.db")
    reg = SqliteEntityRegistry(path, db_chain=True, register_buffer=5)
    # Commit two rows durably.
    reg.register(_ev(0))
    reg.register(_ev(1))
    reg.flush()
    committed_tip = reg._chain_tip

    # Buffer a third row (pending, not committed), then force a rollback via a
    # content-address violation (same record_id, different payload).
    reg.register(_ev(2))
    clashing = EntityRecord(
        record_id=_ev(0).record_id,
        stable_id="evt-0",
        version=1,
        kind="security_event",
        payload={"seq": 12345},
        metadata={},
    )
    with pytest.raises(HimmyError):
        reg.register(clashing)

    # The tip is re-anchored to the committed tail (the pending batch incl evt-2 was discarded).
    assert reg._chain_tip == committed_tip

    # A subsequent register chains cleanly off the committed tail; the chain still verifies.
    reg.register(_ev(3))
    result = verify_db_chain(reg.db_chain_rows())
    assert result.ok is True
    # Only the two committed rows + the new one are chained (evt-2 was rolled back).
    assert result.chained == 3
    reg.close()


def test_chain_step_for_reuses_integrity_primitives() -> None:
    """chain_step_for is exactly _chain_step(content_hash(record)) over the prev hash."""
    from himmy.entities.integrity import _chain_step, content_hash

    r = _ev(7)
    assert chain_step_for(CHAIN_GENESIS_HASH, r) == _chain_step(
        CHAIN_GENESIS_HASH, content_hash(r)
    )
