"""Memory -> EntityRecord projection: memory facts become first-class spine nodes.

This is the load-bearing differentiator: a remembered fact is not an island next to
the audit spine — it IS an :class:`~himmy.entities.records.EntityRecord` (kind
``memory_fact``), projected through the single :func:`himmy.entities.projection.project`
algorithm exactly like every other domain artefact (events, recommendations, threads).

The projection deliberately captures only the IMMUTABLE content of a fact in its
``payload`` — the mutable validity fields (``valid_to``/``superseded_by``) live in
``metadata`` instead. That mirrors the recommendation precedent
(:meth:`himmy.services.storage.models.Recommendation.to_record`, which excludes the
mutable ``status``): the content-addressed ``record_id`` must stay stable when a fact
is later *invalidated*, so re-registering the same fact remains idempotent. Validity
transitions are expressed as NEW VERSIONS plus typed ``supersedes``/``invalidated_by``
links — not by editing a registered payload (which would trip the registry's
content-address-violation guard).
"""

from __future__ import annotations

from himmy.entities.projection import project
from himmy.entities.records import EntityRecord
from himmy.services.memory.store import MemoryRecord

#: The spine ``kind``/``namespace`` every projected memory fact carries.
MEMORY_FACT_KIND = "memory_fact"


def memory_to_record(record: MemoryRecord, *, version: int = 1) -> EntityRecord:
    """Project a :class:`MemoryRecord` into its canonical ``memory_fact`` record.

    Successive versions of the SAME fact (an UPDATE that supersedes a prior value)
    share a ``stable_id`` derived from ``stable_key``, so the version chain is the
    fact's transaction-time history; a fresh fact (no ``stable_key``) is keyed by its
    ``memory_id``. The ``payload`` is the immutable content (subject/kind/text/
    valid_from/source); ``valid_to``/``superseded_by``/``confidence``/``tier`` go to
    ``metadata`` so they can change across versions without drifting ``record_id``.
    """
    stable_value = record.stable_key or record.memory_id
    return project(
        record,
        stable_value=stable_value,
        namespace=MEMORY_FACT_KIND,
        kind=MEMORY_FACT_KIND,
        version=version,
        fallback_key=record.memory_id,
        payload={
            "memory_id": record.memory_id,
            "subject_id": record.subject_id,
            "kind": record.kind,
            "text": record.text,
            "valid_from": record.valid_from,
            "source": record.source,
            "stable_key": record.stable_key,
        },
        metadata={
            "subject_id": record.subject_id,
            "tier": record.tier,
            "valid_to": record.valid_to,
            "superseded_by": record.superseded_by,
            "confidence": record.confidence,
            "source": record.source,
        },
    )


__all__ = ["MEMORY_FACT_KIND", "memory_to_record"]
