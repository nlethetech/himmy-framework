"""Entities kernel: the structural :class:`EntityRegistryProtocol`.

A single ``typing.Protocol`` describing the registry surface every consumer uses,
so a call site can accept ANY backend that implements it — the volatile in-memory
:class:`~himmy.entities.registry.EntityRegistry`, the durable file-backed
:class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry`, or a future backend —
without binding to one concrete class.

**Why a Protocol (structural), not a base class (nominal).** The two registries are
already API-compatible by hand (their docstrings say so), but nothing PROVED it and
nothing let a caller depend on the shared surface. Worse, the nominal type leaked into
a Pydantic field — :class:`~himmy.services.evaluation.privacy.models.RecordedDataContext`
annotated ``entity_registry: EntityRegistry`` with ``arbitrary_types_allowed=True``, so
Pydantic validated it with ``isinstance(value, EntityRegistry)`` and REJECTED a
durable :class:`SqliteEntityRegistry` at runtime (see ``cli/audit.py``'s
"validates the registry against the concrete EntityRegistry type" note, and the
``cast(EntityRegistry, SqliteEntityRegistry(...))`` bridges in ``cli/consent.py`` /
``cli/security_audit_cmd.py``). Retyping these sites against this Protocol removes the
casts and lets a durable spine be injected where today an in-memory registry is required —
the hard prerequisite for the SpineFactory swap (plan item T2.1).

The Protocol is :func:`~typing.runtime_checkable` so a test (and Pydantic's
``arbitrary_types_allowed`` validation) can assert membership via ``isinstance`` — note
that, per the stdlib, a runtime ``isinstance`` check verifies only that the methods are
PRESENT, not that their signatures match; the static ``mypy`` check enforces signature
conformance. This is a PURE typing refactor: no behaviour changes and no registry is
re-implemented — both existing registries already satisfy the surface verbatim.
"""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from himmy.entities.lineage import (
    DEFAULT_TRACE_DEPTH,
    LineageDirection,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.lineage import LineageGraph
    from himmy.entities.records import (
        EntityLink,
        EntityQuery,
        EntityRecord,
    )


@runtime_checkable
class EntityRegistryProtocol(Protocol):
    """The structural surface shared by every entity registry backend.

    This is the union of the methods consumers actually call — registration and
    versioning (``register`` / ``new_version`` / ``link``), point/version reads
    (``get`` / ``get_latest`` / ``get_history`` / ``list_by_kind`` / ``query``), and
    the lineage walk (``links_from`` / ``links_to`` / ``neighbors`` / ``trace``).
    Both :class:`~himmy.entities.registry.EntityRegistry` and
    :class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry` satisfy it verbatim.

    A registry MAY expose more (the SQLite backend adds ``all_records`` / ``all_links``
    for tamper-evidence exports, and a ``close``); the Protocol is the LOWEST common
    surface every consumer is allowed to depend on.
    """

    # ----------------------------------------------------------------- writes
    def register(self, record: EntityRecord) -> EntityRecord:
        """Register a record; idempotent on identical content, raises on collision."""
        ...

    def new_version(
        self,
        *,
        stable_id: str,
        kind: str,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = ...,
        expected_version: int | None = ...,
    ) -> EntityRecord:
        """Create the next version of an artefact with optimistic concurrency."""
        ...

    def link(
        self,
        *,
        from_record_id: str,
        to_record_id: str,
        relation: str,
        metadata: dict[str, Any] | None = ...,
    ) -> EntityLink:
        """Create and store a typed link between two records."""
        ...

    # ------------------------------------------------------------------ reads
    def get(self, record_id: str) -> EntityRecord | None:
        """Return a record by its physical id, or ``None``."""
        ...

    def get_latest(self, stable_id: str) -> EntityRecord | None:
        """Return the highest-version record for an artefact, or ``None``."""
        ...

    def get_history(self, stable_id: str) -> list[EntityRecord]:
        """Return all versions of an artefact in ascending version order."""
        ...

    def list_by_kind(self, kind: str) -> list[EntityRecord]:
        """Return every record of the given kind."""
        ...

    def query(self, q: EntityQuery) -> list[EntityRecord]:
        """Return records matching the supplied query filter."""
        ...

    # --------------------------------------------------------------- lineage
    def links_from(self, record_id: str) -> list[EntityLink]:
        """Return all links originating from a record."""
        ...

    def links_to(self, record_id: str) -> list[EntityLink]:
        """Return all links pointing INTO a record."""
        ...

    def neighbors(
        self,
        record_id: str,
        *,
        direction: LineageDirection = LineageDirection.BOTH,
        relation: str | None = ...,
    ) -> list[EntityLink]:
        """Return the links incident to a record in the requested direction."""
        ...

    def trace(
        self,
        record_id: str,
        *,
        max_depth: int = DEFAULT_TRACE_DEPTH,
        direction: LineageDirection = LineageDirection.BOTH,
        relations: Collection[str] | None = ...,
    ) -> LineageGraph:
        """Walk the lineage graph from ``record_id`` and return the reached subgraph."""
        ...


__all__ = ["EntityRegistryProtocol"]
