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

import tomllib
from pathlib import Path
from typing import Any


def find_project_config(start: str | Path | None = None) -> Path | None:
    """Locate ``himmy.toml`` (cwd) or ``~/.himmy/config.toml``, or ``None``."""
    local = Path(start or Path.cwd()) / "himmy.toml"
    if local.is_file():
        return local
    home = Path.home() / ".himmy" / "config.toml"
    if home.is_file():
        return home
    return None


def load_project_config(start: str | Path | None = None) -> dict[str, Any]:
    """Load the project config as a dict (empty when no file is found)."""
    path = find_project_config(start)
    if path is None:
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


__all__ = ["find_project_config", "load_project_config"]
