"""Memory store: durable persistence for an agent's long-term memories.

A :class:`MemoryRecord` is one remembered fact or episode for a subject (a user, an
agent, a session). :class:`InMemoryMemoryStore` is the volatile default;
:class:`SqliteMemoryStore` persists to a stdlib-``sqlite3`` file so memories survive
across processes (mirroring :class:`~himmy.entities.sqlite_registry.SqliteEntityRegistry`).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from himmy.entities.records import EntityRecord

#: The Letta-style memory tiers, in promotion order (hot -> cold).
MEMORY_TIERS: tuple[str, ...] = ("core", "recall", "archival")

# -- typed relations between memories ---------------------------------------
#: The canonical relation vocabulary for :class:`MemoryLink`. ``relation`` stays an
#: open ``str`` (callers may coin their own), but these names are the recommended
#: set so graph recall and downstream tooling can reason about edge semantics:
#:
#: * ``relates_to`` — a generic, untyped association (the safe default);
#: * ``caused_by`` — the target fact is a cause of the source fact;
#: * ``contradicts`` — the two facts disagree (drives invalidation review);
#: * ``about_entity`` — the source fact is about the target entity/fact;
#: * ``supersedes`` — the source fact replaces the (now-stale) target;
#: * ``part_of`` — the source fact is a component of the target whole;
#: * ``depends_on`` — the source fact is only true if the target is.
RELATES_TO = "relates_to"
CAUSED_BY = "caused_by"
CONTRADICTS = "contradicts"
ABOUT_ENTITY = "about_entity"
SUPERSEDES = "supersedes"
PART_OF = "part_of"
DEPENDS_ON = "depends_on"

#: The canonical relation names as a tuple (for validation/iteration helpers).
MEMORY_RELATIONS: tuple[str, ...] = (
    RELATES_TO,
    CAUSED_BY,
    CONTRADICTS,
    ABOUT_ENTITY,
    SUPERSEDES,
    PART_OF,
    DEPENDS_ON,
)


class MemoryLink(BaseModel):
    """A typed directed relationship between two :class:`MemoryRecord`s.

    Mirrors :class:`himmy.entities.records.EntityLink` but lives entirely inside the
    memory module: memory links are synchronous, store-local, and have their own
    relation vocabulary (:data:`MEMORY_RELATIONS`), so they deliberately do NOT reuse
    the entities lineage graph (whose async model and relation set differ).
    ``relation`` is an open ``str`` — the canonical names in :data:`MEMORY_RELATIONS`
    are the recommended set, but a caller may coin its own.
    """

    link_id: str = Field(default_factory=new_uuid)
    from_memory_id: str
    to_memory_id: str
    relation: str = RELATES_TO
    metadata: dict[str, Any] = {}
    created_at: str = Field(default_factory=utc_now_iso)


@dataclass
class MemoryGraph:
    """A traced subgraph of memory: the records reached and the links between.

    ``nodes`` maps ``memory_id -> MemoryRecord`` (the root is included when it exists
    and passes the traversal's filters); ``edges`` are the typed links traversed.
    ``truncated`` is True when a ``max_depth`` cutoff stopped the walk while reachable
    nodes remained — so a caller can tell "this is the whole story" apart from "there
    is more, deeper". Mirrors :class:`himmy.entities.lineage.LineageGraph`, but it is
    a plain dataclass (no spine coupling) over memory records.
    """

    root_memory_id: str
    nodes: dict[str, MemoryRecord] = field(default_factory=dict)
    edges: list[MemoryLink] = field(default_factory=list)
    truncated: bool = False

    @property
    def node_count(self) -> int:
        """Number of distinct memories in the graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of distinct links in the graph."""
        return len(self.edges)

    def memory_ids(self) -> set[str]:
        """The set of memory ids present as nodes."""
        return set(self.nodes)

    def relations(self) -> set[str]:
        """The distinct relation names present among the edges."""
        return {edge.relation for edge in self.edges}


#: Alias for a list of links. Because the store classes below define a method named
#: ``list``, the bare ``list[...]`` subscript would (under PEP 563 string annotations)
#: resolve to that method inside the class body; this module-level alias keeps the
#: link-method return annotations referring to the builtin ``list``.
_MemoryLinkList = list[MemoryLink]


class MemoryRecord(BaseModel):
    """One stored memory for a subject.

    A memory is *bi-temporal*: ``created_at`` (and the transaction-time version
    chain on the spine) records WHEN the fact was ingested, while
    ``valid_from``/``valid_to`` record WHEN the fact is true in the world (Graphiti
    semantics). A fact is *invalidated*, not deleted, by stamping ``valid_to`` and
    pointing ``superseded_by`` at its replacement — so point-in-time recall
    ("what was true in March") stays answerable. ``stable_key`` is the semantic
    identity of the fact (e.g. ``"alice/home_city"``) so successive versions of the
    same fact share a spine ``stable_id``. All new fields are defaulted, so existing
    rows and zero-arg constructors keep working unchanged.
    """

    memory_id: str = Field(default_factory=new_uuid)
    subject_id: str = "default"
    kind: str = "semantic"  # semantic | episodic | ...
    text: str = ""
    metadata: dict[str, Any] = {}
    created_at: str = Field(default_factory=utc_now_iso)
    # --- tiered recall (Letta core/recall/archival) ------------------------
    tier: str = "recall"
    # --- bi-temporal validity (Graphiti invalidate-not-delete) -------------
    valid_from: str = ""  # filled from created_at in model_post_init when blank
    valid_to: str | None = None  # None = currently true
    superseded_by: str | None = None  # memory_id of the fact that replaced it
    # --- provenance --------------------------------------------------------
    confidence: float = 1.0
    source: str = "user"  # user | llm_extracted | tool | imported
    stable_key: str | None = None  # semantic identity of the FACT (vs the row id)

    def model_post_init(self, __context: Any) -> None:
        """Default ``valid_from`` to ``created_at`` so valid-time always has a start."""
        if not self.valid_from:
            self.valid_from = self.created_at

    def to_record(self, *, version: int = 1) -> EntityRecord:
        """Project this memory into its canonical ``EntityRecord`` (kind ``memory_fact``).

        Delegates to :func:`himmy.services.memory.projection.memory_to_record`; the
        import is lazy to avoid a service -> entities import cycle (mirroring
        :meth:`himmy.core.events.RunEvent.to_record`).
        """
        from himmy.services.memory.projection import memory_to_record

        return memory_to_record(self, version=version)


@runtime_checkable
class MemoryStore(Protocol):
    """Persistence contract for memories (synchronous)."""

    def save(self, record: MemoryRecord) -> MemoryRecord: ...

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[MemoryRecord]: ...

    def get(self, memory_id: str) -> MemoryRecord | None: ...

    def delete(self, memory_id: str) -> bool: ...

    def delete_by_subject(self, subject_id: str) -> int: ...

    # -- typed relational links (graph memory) ------------------------------
    def save_link(self, link: MemoryLink) -> MemoryLink: ...

    def links_from(self, memory_id: str) -> _MemoryLinkList: ...

    def links_to(self, memory_id: str) -> _MemoryLinkList: ...


def _passes_filters(
    record: MemoryRecord, *, active_only: bool, tier: str | None
) -> bool:
    """Apply the (backend-agnostic) ``active_only``/``tier`` row filters."""
    if active_only and record.valid_to is not None:
        return False
    if tier is not None and record.tier != tier:
        return False
    return True


class InMemoryMemoryStore:
    """A volatile, process-local :class:`MemoryStore`."""

    def __init__(self) -> None:
        self._records: dict[str, MemoryRecord] = {}
        self._links: dict[str, MemoryLink] = {}

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self._records[record.memory_id] = record
        return record

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[MemoryRecord]:
        records = [
            r
            for r in self._records.values()
            if (subject_id is None or r.subject_id == subject_id)
            and _passes_filters(r, active_only=active_only, tier=tier)
        ]
        return sorted(records, key=lambda r: r.created_at)

    def get(self, memory_id: str) -> MemoryRecord | None:
        return self._records.get(memory_id)

    def delete(self, memory_id: str) -> bool:
        return self._records.pop(memory_id, None) is not None

    def delete_by_subject(self, subject_id: str) -> int:
        """Hard-DELETE every memory for ``subject_id`` (the S4 reach for this store)."""
        doomed = [
            mid for mid, r in self._records.items() if r.subject_id == subject_id
        ]
        for mid in doomed:
            del self._records[mid]
        return len(doomed)

    # -- typed relational links (graph memory) ------------------------------
    def save_link(self, link: MemoryLink) -> MemoryLink:
        self._links[link.link_id] = link
        return link

    def links_from(self, memory_id: str) -> _MemoryLinkList:
        return [
            link for link in self._links.values() if link.from_memory_id == memory_id
        ]

    def links_to(self, memory_id: str) -> _MemoryLinkList:
        return [link for link in self._links.values() if link.to_memory_id == memory_id]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    memory_id  TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memories_subject_idx ON memories (subject_id);
"""

#: Columns persisted for a :class:`MemoryLink` row (insert/select order).
_LINK_COLS = "link_id, from_memory_id, to_memory_id, relation, metadata, created_at"

#: Additive DDL for the typed-link table + its lookup indexes. Created on every
#: open (``CREATE TABLE IF NOT EXISTS``) and also folded into ``_MIGRATIONS`` so a
#: legacy ``.himmy/memory.db`` (pre-graph) gains the table in place with no data
#: loss. Kept out of ``_SCHEMA`` only so the migration table-existence guard owns it.
_LINK_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_links (
    link_id        TEXT PRIMARY KEY,
    from_memory_id TEXT NOT NULL,
    to_memory_id   TEXT NOT NULL,
    relation       TEXT NOT NULL DEFAULT 'relates_to',
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS memory_links_from_idx ON memory_links (from_memory_id);
CREATE INDEX IF NOT EXISTS memory_links_to_idx ON memory_links (to_memory_id);
"""

#: Columns added after the original 6-column schema, with the ``ALTER TABLE``
#: definition used to backfill them on a legacy database. Each is PRAGMA-guarded
#: so :meth:`SqliteMemoryStore._migrate` is idempotent and upgrades existing
#: ``.himmy/memory.db`` files in place (no data loss; old rows read with defaults).
_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("tier", "ALTER TABLE memories ADD COLUMN tier TEXT NOT NULL DEFAULT 'recall'"),
    (
        "valid_from",
        "ALTER TABLE memories ADD COLUMN valid_from TEXT NOT NULL DEFAULT ''",
    ),
    ("valid_to", "ALTER TABLE memories ADD COLUMN valid_to TEXT"),
    ("superseded_by", "ALTER TABLE memories ADD COLUMN superseded_by TEXT"),
    (
        "confidence",
        "ALTER TABLE memories ADD COLUMN confidence REAL NOT NULL DEFAULT 1.0",
    ),
    ("source", "ALTER TABLE memories ADD COLUMN source TEXT NOT NULL DEFAULT 'user'"),
    ("stable_key", "ALTER TABLE memories ADD COLUMN stable_key TEXT"),
)


class SqliteMemoryStore:
    """A durable, file-backed :class:`MemoryStore` (stdlib sqlite3)."""

    def __init__(self, path: str = ":memory:") -> None:
        """Open (or create) the SQLite database at ``path`` and migrate it in place."""
        from himmy.core.sqlite_util import connect_hardened

        self._conn = connect_hardened(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additively add any missing columns/tables on a legacy db (idempotent).

        Bi-temporal/tier columns are backfilled per-column (PRAGMA-guarded), and the
        typed-link table (graph memory) is created in place — both upgrade an existing
        ``.himmy/memory.db`` with no data loss. ``CREATE TABLE IF NOT EXISTS`` makes
        the link DDL safe to re-run, so the whole migration stays idempotent.
        """
        existing = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        for column, ddl in _MIGRATIONS:
            if column not in existing:
                self._conn.execute(ddl)  # noqa: S608 - constant DDL, no interpolation
        # Additive graph-memory link table (created in place on a pre-graph db).
        self._conn.executescript(_LINK_SCHEMA)

    def save(self, record: MemoryRecord) -> MemoryRecord:
        self._conn.execute(
            "INSERT OR REPLACE INTO memories "
            "(memory_id, subject_id, kind, text, metadata, created_at, "
            " tier, valid_from, valid_to, superseded_by, confidence, source, "
            " stable_key) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.memory_id,
                record.subject_id,
                record.kind,
                record.text,
                json.dumps(record.metadata),
                record.created_at,
                record.tier,
                record.valid_from,
                record.valid_to,
                record.superseded_by,
                record.confidence,
                record.source,
                record.stable_key,
            ),
        )
        self._conn.commit()
        return record

    def list(
        self,
        subject_id: str | None = None,
        *,
        active_only: bool = False,
        tier: str | None = None,
    ) -> list[MemoryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if subject_id is not None:
            clauses.append("subject_id = ?")
            params.append(subject_id)
        if active_only:
            clauses.append("valid_to IS NULL")
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM memories{where} ORDER BY created_at",  # noqa: S608
            params,
        ).fetchall()
        return [self._row(r) for r in rows]

    def get(self, memory_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        return self._row(row) if row else None

    def delete(self, memory_id: str) -> bool:
        cur = self._conn.execute(
            "DELETE FROM memories WHERE memory_id = ?", (memory_id,)
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete_by_subject(self, subject_id: str) -> int:
        """Hard-DELETE every memory row for ``subject_id`` (the S4 reach for this store).

        Idempotent: a re-run after the subject is gone deletes nothing and returns 0.
        """
        cur = self._conn.execute(
            "DELETE FROM memories WHERE subject_id = ?", (subject_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        self._conn.close()

    # -- typed relational links (graph memory) ------------------------------
    def save_link(self, link: MemoryLink) -> MemoryLink:
        """Persist a typed link (insert-or-replace by ``link_id``)."""
        self._conn.execute(
            f"INSERT OR REPLACE INTO memory_links ({_LINK_COLS}) "  # noqa: S608
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                link.link_id,
                link.from_memory_id,
                link.to_memory_id,
                link.relation,
                json.dumps(link.metadata),
                link.created_at,
            ),
        )
        self._conn.commit()
        return link

    def links_from(self, memory_id: str) -> _MemoryLinkList:
        """Return all links originating from a memory."""
        rows = self._conn.execute(
            f"SELECT {_LINK_COLS} FROM memory_links "  # noqa: S608
            "WHERE from_memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    def links_to(self, memory_id: str) -> _MemoryLinkList:
        """Return all links pointing INTO a memory (the reverse of ``links_from``)."""
        rows = self._conn.execute(
            f"SELECT {_LINK_COLS} FROM memory_links "  # noqa: S608
            "WHERE to_memory_id = ?",
            (memory_id,),
        ).fetchall()
        return [self._row_to_link(r) for r in rows]

    @staticmethod
    def _row_to_link(row: sqlite3.Row) -> MemoryLink:
        return MemoryLink(
            link_id=row["link_id"],
            from_memory_id=row["from_memory_id"],
            to_memory_id=row["to_memory_id"],
            relation=row["relation"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
        )

    @staticmethod
    def _row(row: sqlite3.Row) -> MemoryRecord:
        keys = set(row.keys())
        # ``stable_key`` etc. may be absent only if a migration was skipped; guard
        # each new column so a partially-migrated row never raises a KeyError.
        return MemoryRecord(
            memory_id=row["memory_id"],
            subject_id=row["subject_id"],
            kind=row["kind"],
            text=row["text"],
            metadata=json.loads(row["metadata"]),
            created_at=row["created_at"],
            tier=row["tier"] if "tier" in keys else "recall",
            valid_from=(
                row["valid_from"]
                if "valid_from" in keys and row["valid_from"]
                else row["created_at"]
            ),
            valid_to=row["valid_to"] if "valid_to" in keys else None,
            superseded_by=row["superseded_by"] if "superseded_by" in keys else None,
            confidence=row["confidence"] if "confidence" in keys else 1.0,
            source=row["source"] if "source" in keys else "user",
            stable_key=row["stable_key"] if "stable_key" in keys else None,
        )


__all__ = [
    "ABOUT_ENTITY",
    "CAUSED_BY",
    "CONTRADICTS",
    "DEPENDS_ON",
    "MEMORY_RELATIONS",
    "MEMORY_TIERS",
    "PART_OF",
    "RELATES_TO",
    "SUPERSEDES",
    "InMemoryMemoryStore",
    "MemoryGraph",
    "MemoryLink",
    "MemoryRecord",
    "MemoryStore",
    "SqliteMemoryStore",
]
