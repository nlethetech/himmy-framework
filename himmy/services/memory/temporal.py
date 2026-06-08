"""Bi-temporal filtering helpers for memory recall.

Kept separate from ``store.py`` (a thin persistence layer) and ``service.py`` (recall
ranking) so the valid-time interval logic lives in one place. A memory fact is true
over the half-open interval ``[valid_from, valid_to)`` (Graphiti semantics): ``valid_to
is None`` means "currently true". ``as_of`` point-in-time recall asks "which facts were
true at this instant", which is what makes "what did the user believe in March"
answerable after the fact has since been superseded.
"""

from __future__ import annotations

from himmy.services.memory.store import MemoryRecord


def is_valid_at(record: MemoryRecord, as_of: str) -> bool:
    """Return True when ``record`` is true at instant ``as_of`` (ISO-8601 string).

    The interval is half-open ``[valid_from, valid_to)``: a fact is in effect from
    ``valid_from`` (inclusive) until ``valid_to`` (exclusive), or indefinitely when
    ``valid_to`` is ``None``. ISO-8601 timestamps compare correctly as strings, so no
    parsing is needed for the common ``utc_now_iso()`` format.
    """
    if as_of < record.valid_from:
        return False
    if record.valid_to is not None and as_of >= record.valid_to:
        return False
    return True


def filter_as_of(records: list[MemoryRecord], as_of: str) -> list[MemoryRecord]:
    """Keep only the records whose validity window contains ``as_of``."""
    return [r for r in records if is_valid_at(r, as_of)]


__all__ = ["is_valid_at", "filter_as_of"]
