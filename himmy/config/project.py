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

#: Override for the canonical durable conversation store (``.himmy/conversations.db`` by
#: default). When set it is honoured verbatim — the parent directory is created on first use.
HIMMY_CONVERSATIONS_PATH_ENV = "HIMMY_CONVERSATIONS_PATH"

#: Override for the /v1 HITL checkpoint store (``.himmy/v1_approvals.db`` by default).
#: This is the SURFACE-OWNED durable inbox for ``/v1`` approval-gated runs (T2f), kept
#: DISTINCT from Studio's ``.himmy/approvals.db`` — the two surfaces resume from their own
#: spec source (Studio from an ``agent_path`` filesystem file, /v1 from a stored DB
#: ``AgentSpec``), so a single shared inbox is a deferred item, not this tier. When set the
#: value is honoured verbatim — the parent directory is created on first use.
HIMMY_V1_APPROVALS_PATH_ENV = "HIMMY_V1_APPROVALS_PATH"

#: Env override for the durable /v1 graph-checkpoint store (T3b). A long graph
#: workflow/team run checkpoints each completed superstep here so it resumes after a
#: restart. When set the value is honoured verbatim — the parent directory is created on
#: first use; otherwise ``<project-root>/.himmy/graph_checkpoints.db`` is used.
HIMMY_GRAPH_CHECKPOINTS_PATH_ENV = "HIMMY_GRAPH_CHECKPOINTS_PATH"

#: Override for the durable SubjectKeyVault (``.himmy/keyvault.db`` by default, S2). This is
#: the single most SECURITY-CRITICAL file in a governed deployment: it holds every subject's
#: KEK (wrapped under the configured meta-KEK provider), and **destroying it IS erasure** —
#: a lost or unbacked keyvault.db permanently crypto-shreds every governed subject. It must
#: share the same backup/residency posture as ``spine.db`` (operator warning). When set the
#: value is honoured verbatim — the parent directory is created on first use.
HIMMY_KEYVAULT_PATH_ENV = "HIMMY_KEYVAULT_PATH"


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


def conversations_db_path(start: str | Path | None = None) -> str:
    """The ONE canonical durable conversation store the CLI and Studio both share (T2.3).

    Resolution: ``HIMMY_CONVERSATIONS_PATH`` when set (honoured verbatim, its parent dir
    created), otherwise ``<project-root>/.himmy/conversations.db`` (project root resolved by
    :func:`find_project_root`). This is the single decision point for WHERE conversations
    live, so a thread started in ``himmy chat`` is the same on-disk database the Studio Chats
    list reads — coherence across the interfaces.
    """
    override = os.environ.get(HIMMY_CONVERSATIONS_PATH_ENV)
    if override and override.strip():
        path = Path(override.strip()).expanduser()
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(himmy_dir(start) / "conversations.db")


def v1_approvals_db_path(start: str | Path | None = None) -> str:
    """The durable /v1 HITL checkpoint store path (per-surface inbox, T2f).

    Resolution: ``HIMMY_V1_APPROVALS_PATH`` when set (honoured verbatim, its parent dir
    created), otherwise ``<project-root>/.himmy/v1_approvals.db`` (project root resolved by
    :func:`find_project_root`). DELIBERATELY a different file than Studio's
    ``.himmy/approvals.db``: the two surfaces rebuild their paused runtime from different
    spec sources (Studio from a filesystem ``agent_path``; /v1 from a stored DB
    ``AgentSpec`` a /v1 row carries but a Studio row does not), so a drop-in shared inbox
    is a future item — keeping the inboxes per-surface here is the reviewer must_fix.
    """
    override = os.environ.get(HIMMY_V1_APPROVALS_PATH_ENV)
    if override and override.strip():
        path = Path(override.strip()).expanduser()
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(himmy_dir(start) / "v1_approvals.db")


def graph_checkpoints_db_path(start: str | Path | None = None) -> str:
    """The durable /v1 graph-checkpoint store path for long team/workflow runs (T3b).

    Resolution: ``HIMMY_GRAPH_CHECKPOINTS_PATH`` when set (honoured verbatim, its parent dir
    created), otherwise ``<project-root>/.himmy/graph_checkpoints.db``. A ``graph`` team run
    or any workflow run persists each completed superstep here so it resumes from the last
    one after a simulated restart (the T3b durable-resume acceptance).
    """
    override = os.environ.get(HIMMY_GRAPH_CHECKPOINTS_PATH_ENV)
    if override and override.strip():
        path = Path(override.strip()).expanduser()
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(himmy_dir(start) / "graph_checkpoints.db")


def keyvault_db_path(start: str | Path | None = None) -> str:
    """The durable SubjectKeyVault path (``.himmy/keyvault.db`` by default, S2).

    Resolution: ``HIMMY_KEYVAULT_PATH`` when set (honoured verbatim, its parent dir
    created), otherwise ``<project-root>/.himmy/keyvault.db`` (project root resolved by
    :func:`find_project_root`, so it sits next to ``spine.db``). This is the durable home
    for per-subject KEKs: destroying a subject's row crypto-shreds that subject, and losing
    the whole file shreds every governed subject — give it the SAME backup/residency posture
    as ``spine.db``.
    """
    override = os.environ.get(HIMMY_KEYVAULT_PATH_ENV)
    if override and override.strip():
        path = Path(override.strip()).expanduser()
        if path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)
    return str(himmy_dir(start) / "keyvault.db")


def load_project_config(start: str | Path | None = None) -> dict[str, Any]:
    """Load the project config as a dict (empty when no file is found)."""
    path = find_project_config(start)
    if path is None:
        return {}
    with path.open("rb") as handle:
        return tomllib.load(handle)


__all__ = [
    "HIMMY_CONVERSATIONS_PATH_ENV",
    "HIMMY_DIR_NAME",
    "HIMMY_GRAPH_CHECKPOINTS_PATH_ENV",
    "HIMMY_KEYVAULT_PATH_ENV",
    "HIMMY_SPINE_PATH_ENV",
    "HIMMY_V1_APPROVALS_PATH_ENV",
    "conversations_db_path",
    "find_project_config",
    "find_project_root",
    "graph_checkpoints_db_path",
    "himmy_dir",
    "keyvault_db_path",
    "v1_approvals_db_path",
    "load_project_config",
    "spine_db_path",
]
