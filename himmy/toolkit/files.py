"""Files pack: ``read_file``, ``write_file``, and ``list_dir`` under a sandbox root.

Every path the model supplies is resolved against a configured ``fs_root`` and must
stay inside it — traversal (``../``) and symlink escapes are rejected by comparing the
fully-resolved real path against the resolved root. ``write_file`` is approval-gated by
default (it mutates the filesystem) unless ``fs_allow_write`` is set on the config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig


def _safe_path(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root``; raise if it escapes the jail.

    The check is on the *resolved* real paths, so both ``../`` traversal and a
    symlink pointing outside the root are caught.
    """
    root_resolved = root.resolve()
    candidate = (root_resolved / rel).resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ToolSecurityError(f"path {rel!r} escapes the sandbox root")
    return candidate


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
        data = target.read_bytes()[:max_bytes]
        return {
            "path": path,
            "content": data.decode("utf-8", errors="replace"),
            "truncated": target.stat().st_size > len(data),
        }

    def write_file(args: dict[str, Any]) -> dict[str, Any]:
        path = str(args["path"])
        content = str(args["content"])
        target = _safe_path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if args.get("append") else "w"
        with target.open(mode, encoding="utf-8") as fh:
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
        handler=read_file,
        description="Read a UTF-8 text file under the sandbox root.",
        args_json_schema=_READ_SCHEMA,
        metadata={"pack": "files"},
    )
    register_local_tool(
        registry,
        name="write_file",
        handler=write_file,
        description="Write (or append to) a file under the sandbox root.",
        args_json_schema=_WRITE_SCHEMA,
        requires_approval=not config.fs_allow_write,
        metadata={"pack": "files"},
    )
    register_local_tool(
        registry,
        name="list_dir",
        handler=list_dir,
        description="List the entries of a directory under the sandbox root.",
        args_json_schema=_LIST_SCHEMA,
        metadata={"pack": "files"},
    )


__all__ = ["register_files_pack"]
