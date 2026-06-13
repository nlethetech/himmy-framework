"""Studio audit spine: a durable, process-wide :class:`SqliteEntityRegistry`.

Studio's other state (runs, memory, routines) lives in cwd-keyed SQLite sidecars
under ``.himmy/``; this is the matching home for the **entity/audit spine** — the
immutable, versioned record trail that learning signals (run feedback, memory
mutations) project onto so they survive independent of any single sidecar and are
erasure-scopable. Like :func:`himmy.api.studio_runs.get_run_store` and
:func:`himmy.api.studio_memory.get_memory_service` it is a process-wide singleton
re-opened when the project root (cwd) changes, so tests that ``chdir`` between
cases get a fresh spine.
"""

from __future__ import annotations

from pathlib import Path

from himmy.entities.sqlite_registry import SqliteEntityRegistry


def entities_db_path() -> str:
    """Path to the Studio entity spine DB (``.himmy/entities.db``), creating ``.himmy``.

    Honours ``HIMMY_ENTITIES_PATH`` for tests/airgap relocation, mirroring the
    ``HIMMY_MEMORY_PATH`` override on the memory store.
    """
    import os

    env = os.environ.get("HIMMY_ENTITIES_PATH")
    if env:
        return env
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "entities.db")


_REGISTRY: SqliteEntityRegistry | None = None
_REGISTRY_PATH: str | None = None


def get_entity_registry() -> SqliteEntityRegistry:
    """Process-wide durable entity registry, (re)opened if the cwd/env path changed."""
    global _REGISTRY, _REGISTRY_PATH
    path = entities_db_path()
    if _REGISTRY is None or _REGISTRY_PATH != path:
        if _REGISTRY is not None:
            _REGISTRY.close()
        _REGISTRY = SqliteEntityRegistry(path)
        _REGISTRY_PATH = path
    return _REGISTRY


def reset_entity_registry() -> None:
    """Drop the cached registry handle (tests change cwd between cases)."""
    global _REGISTRY, _REGISTRY_PATH
    if _REGISTRY is not None:
        _REGISTRY.close()
    _REGISTRY = None
    _REGISTRY_PATH = None


__all__ = [
    "entities_db_path",
    "get_entity_registry",
    "reset_entity_registry",
]
