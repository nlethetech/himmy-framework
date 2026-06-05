"""Long-term memory: durable remember/recall + auto-injection into context.

from himmy.services.memory import MemoryService, SqliteMemoryStore

memory = MemoryService(SqliteMemoryStore("memory.db"))
memory.remember("The user prefers metric units.", subject_id="alice")
hits = await memory.recall("units preference", subject_id="alice")
"""

from __future__ import annotations

from himmy.services.memory.adapter import MemoryContextAdapter
from himmy.services.memory.service import MemoryHit, MemoryService
from himmy.services.memory.store import (
    InMemoryMemoryStore,
    MemoryRecord,
    MemoryStore,
    SqliteMemoryStore,
)

__all__ = [
    "MemoryService",
    "MemoryHit",
    "MemoryRecord",
    "MemoryStore",
    "InMemoryMemoryStore",
    "SqliteMemoryStore",
    "MemoryContextAdapter",
]
