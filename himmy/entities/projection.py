"""Entities kernel: the single domain -> EntityRecord projection algorithm.

Every first-class domain artefact (events, personas, threads, tasks, agents,
workflows, skills, tool definitions, context snapshots, recommendations) is
projected into a canonical :class:`EntityRecord` the same way: derive a
deterministic ``stable_id`` from a semantic key, then content-address it via
:meth:`EntityRecord.create`. This module owns that algorithm so the ~dozen domain
models only declare WHAT identifies them (their stable key, namespace, and kind) —
not HOW projection works. If the record schema or versioning ever changes, it
changes here once instead of in every model.

Each model keeps a thin ``to_record()`` wrapper that lazily imports :func:`project`.
The lazy import is deliberate: importing ``himmy.entities`` eagerly from a core or
service model would reintroduce the historical ``core <-> entities`` import cycle.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from himmy.entities.records import EntityRecord, stable_id_for


def project(
    entity: BaseModel,
    *,
    stable_value: str,
    namespace: str,
    kind: str,
    version: int = 1,
    fallback_key: str | None = None,
    payload: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> EntityRecord:
    """Project a domain model into its canonical :class:`EntityRecord`.

    ``stable_value`` is the model's semantic identity key (e.g. ``event_id`` or
    ``name``); it is run through :func:`stable_id_for` with ``namespace`` (and an
    optional ``fallback_key`` for models whose primary key may be blank, like a
    persona falling back to its name). ``payload`` defaults to
    ``entity.model_dump(mode="json")`` — pass an explicit payload only when the
    record must capture a curated subset (e.g. a recommendation excludes its
    mutable ``status``/``notes`` so the content-addressed ``record_id`` stays
    stable across status transitions).
    """
    stable_id = stable_id_for(
        stable_value, namespace=namespace, fallback_key=fallback_key
    )
    return EntityRecord.create(
        stable_id=stable_id,
        version=version,
        kind=kind,
        payload=entity.model_dump(mode="json") if payload is None else payload,
        metadata=metadata or {},
    )


__all__ = ["project"]
