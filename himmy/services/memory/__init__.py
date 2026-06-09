"""Long-term memory: durable remember/recall + auto-injection into context.

from himmy.services.memory import MemoryService, SqliteMemoryStore

memory = MemoryService(SqliteMemoryStore("memory.db"))
memory.remember("The user prefers metric units.", subject_id="alice")
hits = await memory.recall("units preference", subject_id="alice")
"""

from __future__ import annotations

from himmy.services.memory.adapter import MemoryContextAdapter
from himmy.services.memory.consolidation import (
    ConsolidationAction,
    ConsolidationResult,
    MemoryConsolidator,
)
from himmy.services.memory.projection import MEMORY_FACT_KIND, memory_to_record
from himmy.services.memory.service import MemoryHit, MemoryService
from himmy.services.memory.store import (
    MEMORY_RELATIONS,
    MEMORY_TIERS,
    InMemoryMemoryStore,
    MemoryGraph,
    MemoryLink,
    MemoryRecord,
    MemoryStore,
    SqliteMemoryStore,
)
from himmy.services.memory.temporal import filter_as_of, is_valid_at

__all__ = [
    "MEMORY_FACT_KIND",
    "MEMORY_RELATIONS",
    "MEMORY_TIERS",
    "ConsolidationAction",
    "ConsolidationResult",
    "InMemoryMemoryStore",
    "MemoryConsolidator",
    "MemoryContextAdapter",
    "MemoryGraph",
    "MemoryHit",
    "MemoryLink",
    "MemoryRecord",
    "MemoryService",
    "MemoryStore",
    "SqliteMemoryStore",
    "filter_as_of",
    "is_valid_at",
    "memory_to_record",
]
