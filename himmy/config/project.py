"""Project configuration: a ``himmy.toml`` for per-project defaults (no env sprawl).

A project can set defaults once in ``./himmy.toml`` (or ``~/.himmy/config.toml``) instead
of passing flags or exporting many ``HIMMY_*`` vars::

    [defaults]
    provider = "ollama"
    model = "qwen2.5:3b-instruct"
    tool_packs = ["web", "utils"]
    guardrails = ["pii"]

    [toolkit]
    embedder = "ollama"
    memory_path = ".himmy/memory.db"

Precedence is **CLI flag > environment > himmy.toml > built-in default** — the CLI applies
``[defaults]`` to the agent/runtime and feeds ``[toolkit]`` to ``ToolkitConfig`` (where the
environment still wins). Read with stdlib ``tomllib``.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

#: The per-project state directory name (durable spine, store, memory all live here).
HIMMY_DIR_NAME = ".himmy"

#: Override for the canonical durable spine path (``.himmy/spine.db`` by default). When
#: set it is honoured verbatim — the parent directory is created on first use.
HIMMY_SPINE_PATH_ENV = "HIMMY_SPINE_PATH"


def find_project_config(start: str | Path | None = None) -> Path | None:
    """Locate ``himmy.toml`` (cwd) or ``~/.himmy/config.toml``, or ``None``."""
    local = Path(start or Path.cwd()) / "himmy.toml"
    if local.is_file():
        return local
    home = Path.home() / ".himmy" / "config.toml"
    if home.is_file():
        return home
    return None


def find_project_root(start: str | Path | None = None) -> Path:
    """Resolve the project root the durable ``.himmy/`` state directory hangs off.

    Walks up from ``start`` (default: the process CWD) and returns the FIRST ancestor
    that already carries a project marker — a ``himmy.toml`` file or an existing
    ``.himmy/`` directory. Falls back to ``start`` itself when no marker is found, so a
    bare checkout still resolves to a stable, predictable location (the CWD) instead of
    creating state in an arbitrary parent. This is what lets the CLI and a server launched
    from anywhere INSIDE the same project tree share ONE ``.himmy/spine.db`` rather than
    scattering a spine per working directory.
    """
    base = Path(start or Path.cwd()).resolve()
    for candidate in (base, *base.parents):
        if (candidate / "himmy.toml").is_file() or (
            candidate / HIMMY_DIR_NAME
        ).is_dir():
            return candidate
    return base


def himmy_dir(start: str | Path | None = None) -> Path:
    """The project's durable ``.himmy/`` state directory (created on demand).

    Resolved against :func:`find_project_root` so every interface launched from inside
    the same project tree agrees on one location. The directory is created (idempotently)
    so callers can write into it without a separate ``mkdir``.
    """
    path = find_project_root(start) / HIMMY_DIR_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def spine_db_path(start: str | Path | None = None) -> str:
    """The ONE canonical durable spine path the CLI, Studio, and /v1 all share.

    Resolution: ``HIMMY_SPINE_PATH`` when set (honoured verbatim, its parent dir created),
    otherwise ``<project-root>/.himmy/spine.db`` (project root resolved by
    :func:`find_project_root`). This is the single decision point for WHERE the entity
    spine lives, so a run's provenance projected by the server is the same on-disk
    database a later ``himmy seclog`` reads — coherence across all three interfaces.
    """
    override = os.environ.get(HIMMY_SPINE_PATH_ENV)
    if override and override.strip():
        path = Path(override.strip()).expanduser()
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(himmy_dir(start) / "spine.db")


def load_project_config(start: str | Path | None = None) -> dict[str, Any]:
    """Load the project config as a dict (empty when no file is found)."""
    path = find_project_config(start)
    if path is None:
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


__all__ = [
    "HIMMY_DIR_NAME",
    "HIMMY_SPINE_PATH_ENV",
    "find_project_config",
    "find_project_root",
    "himmy_dir",
    "load_project_config",
    "spine_db_path",
]
