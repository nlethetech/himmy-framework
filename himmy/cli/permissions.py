"""Permission profiles for the himmy CLI: the ``[permissions]`` allowlist.

A project's ``himmy.toml`` can carry a ``[permissions]`` table whose
``auto_approve = ["tool_a", "tool_b"]`` list names the approval-gated tools the
human has decided to trust for this project. On runtime build the CLI flips
``requires_approval`` off for those tools in the registry, so they run without a
prompt; tools NOT on the list still pause for in-REPL / in-run approval.

The ``always`` answer at an approval prompt persists the just-approved tool's
name into that list — a read-modify-write of the toml that preserves the rest of
the file (and the rest of the ``[permissions]`` table) and only edits/creates the
``auto_approve`` array, so a hand-edited ``himmy.toml`` survives the rewrite.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from himmy.config.project import find_project_root


def load_auto_approve(start: str | Path | None = None) -> list[str]:
    """The ``[permissions] auto_approve`` tool names from ``himmy.toml`` (project root).

    Reads the project's local ``himmy.toml`` only (not the home fallback — an
    allowlist is a per-project trust decision). Resolves the file via
    :func:`~himmy.config.project.find_project_root` so a run launched from a
    SUBDIRECTORY still finds the project's ``himmy.toml`` (the same root mechanism
    ``spine.db`` / ``agent.yaml`` use), rather than silently dropping the allowlist.
    Returns ``[]`` when the file or the table/key is absent or malformed, so a
    missing/odd config never crashes a run.
    """
    path = find_project_root(start) / "himmy.toml"
    if not path.is_file():
        return []
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return []
    perms = data.get("permissions")
    if not isinstance(perms, dict):
        return []
    names = perms.get("auto_approve")
    if not isinstance(names, list):
        return []
    return [str(n) for n in names if isinstance(n, str)]


def load_session_budget(start: str | Path | None = None) -> float | None:
    """The ``[limits] session_budget`` dollar cap from ``himmy.toml`` (project root), or None.

    Same per-project read as :func:`load_auto_approve`: reads only the local
    ``himmy.toml``, resolved via :func:`~himmy.config.project.find_project_root` so a
    run launched from a SUBDIRECTORY still enforces the project's budget cap instead
    of silently falling back to unrestricted. Returns ``None`` when the file/table/key
    is absent or the value is not a positive number, so a missing/odd config simply
    means "no budget".
    """
    path = find_project_root(start) / "himmy.toml"
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    limits = data.get("limits")
    if not isinstance(limits, dict):
        return None
    value = limits.get("session_budget")
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        return None
    return float(value)


def apply_allowlist(registry: Any, allow: list[str]) -> None:
    """Flip ``requires_approval`` off for every allowlisted tool in ``registry``.

    Mutates the matching :class:`~himmy.services.tools.models.ToolDefinition` in
    place (the model is not frozen). A name that isn't registered is ignored —
    the allowlist is advisory, never a hard error. A no-op for an empty list or a
    ``None`` registry, so the zero-config path is untouched.
    """
    if registry is None or not allow:
        return
    wanted = set(allow)
    for definition in registry.list():
        if definition.name in wanted and definition.requires_approval:
            definition.requires_approval = False


def grant_all_approvals(registry: Any) -> None:
    """Flip ``requires_approval`` off for EVERY tool (``--yolo``).

    A no-op for a ``None`` registry. Used by the ``--yolo`` flag so every gated
    tool runs without a prompt for the whole session.
    """
    if registry is None:
        return
    for definition in registry.list():
        if definition.requires_approval:
            definition.requires_approval = False


def _format_str_array(names: list[str]) -> str:
    """Render a TOML string array, one entry per line, deterministic + quoted."""
    if not names:
        return "[]"
    body = ",\n".join(f'    "{n}"' for n in names)
    return f"[\n{body},\n]"


def persist_auto_approve(tool_name: str, *, start: str | Path | None = None) -> bool:
    """Add ``tool_name`` to ``[permissions] auto_approve`` in ``himmy.toml``.

    Read-modify-write that preserves the rest of the file as text wherever
    possible: it rewrites only the ``auto_approve`` array (creating the
    ``[permissions]`` table and/or the file when missing). Returns ``True`` when a
    change was written, ``False`` when the tool was already present (no write).

    The strategy is line-oriented rather than a full TOML re-serialization so a
    hand-authored ``himmy.toml`` (comments, ordering, other tables) is not
    flattened by a round-trip — only the one array is touched.

    Resolves the file via :func:`~himmy.config.project.find_project_root` so an
    ``always`` answer given from a SUBDIRECTORY persists into the SAME
    ``himmy.toml`` :func:`load_auto_approve` later reads, not a stray one in the cwd.
    """
    path = find_project_root(start) / "himmy.toml"
    existing = load_auto_approve(path.parent)
    if tool_name in existing:
        return False
    new_names = [*existing, tool_name]
    array_text = _format_str_array(new_names)

    if not path.is_file():
        path.write_text(
            f"[permissions]\nauto_approve = {array_text}\n", encoding="utf-8"
        )
        return True

    original = path.read_text(encoding="utf-8")
    lines = original.splitlines()
    out: list[str] = []
    in_permissions = False
    replaced_array = False
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            # Entering a new table. If we were in [permissions] and never found
            # an auto_approve key, inject it before leaving the table.
            if in_permissions and not replaced_array:
                out.append(f"auto_approve = {array_text}")
                replaced_array = True
            in_permissions = stripped == "[permissions]"
            out.append(line)
            i += 1
            continue
        if in_permissions and stripped.startswith("auto_approve"):
            # Replace the whole existing array (which may span multiple lines).
            out.append(f"auto_approve = {array_text}")
            replaced_array = True
            # Skip the source array lines until its closing ``]``.
            if "]" not in stripped.split("=", 1)[-1]:
                i += 1
                while i < len(lines) and "]" not in lines[i]:
                    i += 1
            i += 1
            continue
        out.append(line)
        i += 1

    if in_permissions and not replaced_array:
        out.append(f"auto_approve = {array_text}")
        replaced_array = True
    if not replaced_array:
        # No [permissions] table anywhere: append one.
        if out and out[-1].strip() != "":
            out.append("")
        out.append("[permissions]")
        out.append(f"auto_approve = {array_text}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return True


__all__ = [
    "load_auto_approve",
    "load_session_budget",
    "apply_allowlist",
    "grant_all_approvals",
    "persist_auto_approve",
]
