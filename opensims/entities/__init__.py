"""Entities kernel: the lineage backbone (records, links, registry)."""

from __future__ import annotations

from opensims.entities.lineage import (
    DEFAULT_TRACE_DEPTH,
    LineageDirection,
    LineageGraph,
)
from opensims.entities.postgres import PostgresEntityRegistry
from opensims.entities.records import (
    EntityLink,
    EntityQuery,
    EntityRecord,
    record_id_for,
    stable_id_for,
)
from opensims.entities.registry import EntityRegistry

__all__ = [
    "EntityRecord",
    "EntityLink",
    "EntityQuery",
    "EntityRegistry",
    "PostgresEntityRegistry",
    "LineageGraph",
    "LineageDirection",
    "DEFAULT_TRACE_DEPTH",
    "stable_id_for",
    "record_id_for",
]
