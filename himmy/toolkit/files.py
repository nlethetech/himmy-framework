"""Files pack: ``read_file``, ``write_file``, and ``list_dir`` under a sandbox root.

Every path the model supplies is resolved against a configured ``fs_root`` and must
stay inside it — ``../`` traversal is rejected by comparing the fully-resolved real
path against the resolved root, and symlinks anywhere below the root are rejected
outright (even ones currently pointing back inside: a link can be retargeted between
check and use). I/O re-validates the path immediately before opening and opens with
``O_NOFOLLOW`` where the platform supports it, closing the TOCTOU window. ``write_file``
is approval-gated by default (it mutates the filesystem) unless ``fs_allow_write`` is
set on the config.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig

#: ``O_NOFOLLOW`` where the platform has it (0 elsewhere — the explicit symlink
#: rejection in :func:`_safe_path` still guards those platforms).
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _safe_path(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``; raise if it escapes the jail or is a symlink.

    Two layers: the *resolved* real path must stay inside the resolved root (catches
    ``../`` traversal), and every path component below the root — including the
    final target — is rejected if it is a symlink. The symlink check runs on the
    *unresolved* path (links are never followed), so a link inside the sandbox can
    never redirect I/O outside it, not even one whose current target resolves back
    inside the root (it could be retargeted between check and use).
    """
    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ToolSecurityError(f"path {rel!r} escapes the sandbox root")
    try:
        parts = (root_resolved / rel).relative_to(root_resolved).parts
    except ValueError as exc:  # lexically outside the root (absolute detours)
        raise ToolSecurityError(f"path {rel!r} escapes the sandbox root") from exc
    probe = root_resolved
    for part in parts:
        if part == "..":
            probe = probe.parent
            continue
        probe = probe / part
        if probe.is_symlink():
            raise ToolSecurityError(
                f"path {rel!r} crosses a symlink ({probe.name!r}); "
                "symlinks are not allowed inside the sandbox root"
            )
    return candidate


def _open_fd_nofollow(root: Path, rel: str, flags: int) -> int:
    """Open ``rel`` under ``root``, re-validating immediately before the syscall.

    Re-running :func:`_safe_path` here (not only at handler entry) and OR-ing in
    ``O_NOFOLLOW`` closes the check-to-use (TOCTOU) window in which a path
    component could be swapped for a symlink.
    """
    target = _safe_path(root, rel)
    try:
        return os.open(target, flags | _O_NOFOLLOW, 0o666)
    except OSError as exc:
        if exc.errno == errno.ELOOP:  # O_NOFOLLOW hit a symlink at open time
            raise ToolSecurityError(f"path {rel!r} is a symlink") from exc
        raise


_READ_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Path relative to the sandbox root."},
        "max_bytes": {"type": "integer", "minimum": 1, "default": 200_000},
    },
    "required": ["path"],
    "additionalProperties": False,
}

_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
        "append": {"type": "boolean", "default": False},
    },
    "required": ["path", "content"],
    "additionalProperties": False,
}

_LIST_SCHEMA = {
    "type": "object",
    "properties": {"path": {"type": "string", "default": "."}},
    "additionalProperties": False,
}


def register_files_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register the filesystem tools jailed to ``config.fs_root``."""
    root = config.fs_root

    def read_file(args: dict[str, Any]) -> dict[str, Any]:
        path = str(args["path"])
        max_bytes = int(args.get("max_bytes", 200_000))
        target = _safe_path(root, path)
        if not target.is_file():
            raise FileNotFoundError(f"no such file: {path}")
        fd = _open_fd_nofollow(root, path, os.O_RDONLY)
        with os.fdopen(fd, "rb") as fh:
            data = fh.read(max_bytes)
            truncated = os.fstat(fh.fileno()).st_size > len(data)
        return {
            "path": path,
            "content": data.decode("utf-8", errors="replace"),
            "truncated": truncated,
        }

    def write_file(args: dict[str, Any]) -> dict[str, Any]:
        path = str(args["path"])
        content = str(args["content"])
        target = _safe_path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Append/truncate via open(2) flags; ``_open_fd_nofollow`` re-validates the
        # path after the mkdir, immediately before the write actually starts.
        flags = os.O_WRONLY | os.O_CREAT
        flags |= os.O_APPEND if args.get("append") else os.O_TRUNC
        fd = _open_fd_nofollow(root, path, flags)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
        return {"path": path, "bytes_written": len(content.encode("utf-8"))}

    def list_dir(args: dict[str, Any]) -> dict[str, Any]:
        path = str(args.get("path", "."))
        target = _safe_path(root, path)
        if not target.is_dir():
            raise NotADirectoryError(f"not a directory: {path}")
        entries = [
            {"name": child.name, "is_dir": child.is_dir(), "size": child.stat().st_size}
            for child in sorted(target.iterdir())
        ]
        return {"path": path, "entries": entries}

    register_local_tool(
        registry,
        name="read_file",
        read_only=True,
        handler=read_file,
        description="Read a UTF-8 text file under the sandbox root.",
        args_json_schema=_READ_SCHEMA,
        metadata={"pack": "files"},
    )
    register_local_tool(
        registry,
        name="write_file",
        read_only=False,
        handler=write_file,
        description="Write (or append to) a file under the sandbox root.",
        args_json_schema=_WRITE_SCHEMA,
        requires_approval=not config.fs_allow_write,
        metadata={"pack": "files"},
    )
    register_local_tool(
        registry,
        name="list_dir",
        read_only=True,
        handler=list_dir,
        description="List the entries of a directory under the sandbox root.",
        args_json_schema=_LIST_SCHEMA,
        metadata={"pack": "files"},
    )


__all__ = ["register_files_pack"]
