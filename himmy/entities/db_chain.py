"""S6: a DB-resident, from-DB-alone tamper-evidence chain over the audit spine.

:mod:`himmy.entities.integrity` ALREADY ships a full opt-in hash chain
(:func:`~himmy.entities.integrity.export_audit_chain` /
:func:`~himmy.entities.integrity.verify_audit_chain`, :class:`ChainEntry.prev_hash`,
:data:`CHAIN_GENESIS_HASH`, :func:`_chain_step`). Its one limitation is that verification
needs a *separately held* signed bundle: the chain lives in memory at export time, so a
verifier who only has the database cannot detect an in-place row mutation.

This module closes exactly that gap WITHOUT adding a second mechanism: it persists the
EXISTING chain's ``prev_hash`` into the ``entity_records.prev_hash`` column (added by the S1
spine migration) at ``register()`` time, REUSING ``content_hash`` + ``_chain_step``. The
chain is then a linked list inside the table itself — start at the genesis pointer, recompute
each row's ``chain_hash``, follow it to the row whose ``prev_hash`` matches, and walk to the
tip. Any in-place edit changes a row's ``content_hash``, so its recomputed ``chain_hash`` no
longer matches its successor's stored ``prev_hash`` and the walk breaks AT that row — all from
the DB alone.

It stays OFF by default (the column is NULL): enabling is opt-in via the registry flag /
``HIMMY_SPINE_DB_CHAIN`` so a default install is byte-identical beyond the (NULL) column.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from himmy.entities.integrity import (
    CHAIN_GENESIS_HASH,
    _chain_step,
    content_hash,
)
from himmy.entities.records import EntityRecord


@dataclass(frozen=True)
class DbChainRow:
    """One chained row read back from the spine: its content + its stored ``prev_hash``."""

    record: EntityRecord
    prev_hash: str | None


@dataclass(frozen=True)
class DbChainVerification:
    """The outcome of verifying the DB-resident chain from the rows alone.

    ``ok`` is ``True`` when every chained row links to its predecessor's recomputed
    ``chain_hash`` back to genesis with no gap. ``chained`` is how many rows carried a
    ``prev_hash`` (participated in the chain); ``tip_hash`` is the chain hash after the last
    linked row (``CHAIN_GENESIS_HASH`` for an empty chain). ``broken_at_record_id`` names the
    FIRST row whose recomputed link does not match what the next row points at (an in-place
    mutation, a deletion, or a forged ``prev_hash``); ``reason`` describes it.
    """

    ok: bool
    chained: int
    tip_hash: str
    broken_at_record_id: str | None = None
    reason: str = ""
    unchained_record_ids: list[str] = field(default_factory=list)


def chain_step_for(prev_hash: str, record: EntityRecord) -> str:
    """The chain hash after ``record`` given its predecessor's ``prev_hash`` (reuses integrity).

    A thin, public alias over :func:`himmy.entities.integrity._chain_step` +
    :func:`content_hash` so the registry's write path and this verifier compute the link the
    SAME way — there is exactly one chain algorithm, the one in :mod:`integrity`.
    """
    return _chain_step(prev_hash, content_hash(record))


def verify_db_chain(rows: list[DbChainRow]) -> DbChainVerification:
    """Verify the DB-resident chain from the rows alone (no external bundle).

    ``rows`` are every ``entity_records`` row (in any order — the chain is reconstructed by
    following ``prev_hash`` pointers, NOT row order). A row with a NULL ``prev_hash`` never
    joined the chain (e.g. written before the chain was enabled, or with the chain off) and is
    reported in ``unchained_record_ids`` rather than treated as a break.

    Algorithm: build the set of chained rows; the head is the row whose ``prev_hash`` is the
    genesis hash. Walk forward — recompute the head's ``chain_hash`` (``chain_step_for``), find
    the row whose stored ``prev_hash`` equals it, and continue. The walk succeeds iff it visits
    every chained row exactly once and ends with no row pointing past the tip. Any in-place
    mutation changes a row's ``content_hash`` → its recomputed ``chain_hash`` matches no
    successor's ``prev_hash`` → the walk halts AT the mutated row.
    """
    chained = [r for r in rows if r.prev_hash is not None]
    unchained = [r.record.record_id for r in rows if r.prev_hash is None]
    if not chained:
        return DbChainVerification(
            ok=True,
            chained=0,
            tip_hash=CHAIN_GENESIS_HASH,
            unchained_record_ids=unchained,
        )

    # Index chained rows by the prev_hash they declare, so we can follow the linked list.
    by_prev: dict[str, list[DbChainRow]] = {}
    for row in chained:
        assert row.prev_hash is not None
        by_prev.setdefault(row.prev_hash, []).append(row)

    # Detect a fork (two rows claiming the same predecessor) — a tamper signal on its own.
    for prev, group in by_prev.items():
        if len(group) > 1:
            return DbChainVerification(
                ok=False,
                chained=len(chained),
                tip_hash=CHAIN_GENESIS_HASH,
                broken_at_record_id=group[1].record.record_id,
                reason=(
                    f"fork: {len(group)} rows declare prev_hash {prev[:12]}… "
                    "(a chained row was duplicated or a prev_hash forged)"
                ),
                unchained_record_ids=unchained,
            )

    heads = by_prev.get(CHAIN_GENESIS_HASH, [])
    if not heads:
        return DbChainVerification(
            ok=False,
            chained=len(chained),
            tip_hash=CHAIN_GENESIS_HASH,
            reason="no chain head: no row links to the genesis hash (head deleted/mutated)",
            unchained_record_ids=unchained,
        )

    cursor: str = CHAIN_GENESIS_HASH
    visited = 0
    last_record_id: str | None = None
    while cursor in by_prev:
        row = by_prev[cursor][0]
        recomputed = chain_step_for(cursor, row.record)
        cursor = recomputed
        last_record_id = row.record.record_id
        visited += 1

    if visited != len(chained):
        # Some chained rows were never reached: the walk broke before the end (an in-place
        # mutation re-pointed the recomputed chain hash away from the next row's prev_hash).
        reached = set()
        c = CHAIN_GENESIS_HASH
        while c in by_prev:
            r = by_prev[c][0]
            reached.add(r.record.record_id)
            c = chain_step_for(c, r.record)
        orphan = next(
            (r.record.record_id for r in chained if r.record.record_id not in reached),
            None,
        )
        return DbChainVerification(
            ok=False,
            chained=len(chained),
            tip_hash=cursor,
            broken_at_record_id=last_record_id,
            reason=(
                f"chain breaks after {visited}/{len(chained)} rows: a row's recomputed "
                "hash does not match its successor's prev_hash (in-place mutation or "
                f"deletion); first unreachable row {orphan!r}"
            ),
            unchained_record_ids=unchained,
        )

    return DbChainVerification(
        ok=True,
        chained=len(chained),
        tip_hash=cursor,
        unchained_record_ids=unchained,
    )


__all__ = [
    "DbChainRow",
    "DbChainVerification",
    "chain_step_for",
    "verify_db_chain",
]
