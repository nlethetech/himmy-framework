"""Entities kernel: the lineage backbone (records, links, registry)."""

from __future__ import annotations

from himmy.entities.integrity import (
    AuditBundle,
    VerificationResult,
    content_hash,
    export_audit_bundle,
    link_hash,
    verify_audit_bundle,
)
from himmy.entities.lineage import (
    DEFAULT_TRACE_DEPTH,
    LineageDirection,
    LineageGraph,
)
from himmy.entities.postgres import PostgresEntityRegistry
from himmy.entities.projection import project
from himmy.entities.records import (
    EntityLink,
    EntityQuery,
    EntityRecord,
    record_id_for,
    stable_id_for,
)
from himmy.entities.registry import EntityRegistry
from himmy.entities.sqlite_registry import SqliteEntityRegistry

__all__ = [
    "EntityRecord",
    "EntityLink",
    "EntityQuery",
    "EntityRegistry",
    "PostgresEntityRegistry",
    "SqliteEntityRegistry",
    "LineageGraph",
    "LineageDirection",
    "DEFAULT_TRACE_DEPTH",
    "stable_id_for",
    "record_id_for",
    "project",
    # tamper-evident audit
    "AuditBundle",
    "VerificationResult",
    "content_hash",
    "link_hash",
    "export_audit_bundle",
    "verify_audit_bundle",
]
