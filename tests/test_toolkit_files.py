"""Tests for the files pack: read/write/list jailed to a sandbox root."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.files import register_files_pack


def _registry(root: Path) -> ToolRegistry:
    registry = ToolRegistry()
    register_files_pack(registry, ToolkitConfig(fs_root=root, fs_allow_write=True))
    return registry


def test_write_read_roundtrip(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.handler_for("write_file")({"path": "notes/a.txt", "content": "hello"})
    out = reg.handler_for("read_file")({"path": "notes/a.txt"})
    assert out["content"] == "hello"
    assert (tmp_path / "notes" / "a.txt").read_text() == "hello"


def test_append_mode(tmp_path: Path) -> None:
    reg = _registry(tmp_path)
    reg.handler_for("write_file")({"path": "a.txt", "content": "x"})
    reg.handler_for("write_file")({"path": "a.txt", "content": "y", "append": True})
    assert reg.handler_for("read_file")({"path": "a.txt"})["content"] == "xy"


def test_list_dir(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("1")
    (tmp_path / "sub").mkdir()
    out = _registry(tmp_path).handler_for("list_dir")({"path": "."})
    names = {e["name"]: e["is_dir"] for e in out["entries"]}
    assert names == {"one.txt": False, "sub": True}


def test_traversal_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ToolSecurityError):
        _registry(tmp_path).handler_for("read_file")({"path": "../escape.txt"})


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "secret.txt"
    outside.write_text("top secret")
    root = tmp_path / "jail"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(ToolSecurityError):
        _registry(root).handler_for("read_file")({"path": "link"})


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _registry(tmp_path).handler_for("read_file")({"path": "nope.txt"})


def test_symlink_to_outside_rejected_for_read_and_write(tmp_path: Path) -> None:
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret")
    root = tmp_path / "jail"
    root.mkdir()
    (root / "link").symlink_to(outside)
    reg = _registry(root)
    with pytest.raises(ToolSecurityError):
        reg.handler_for("read_file")({"path": "link"})
    with pytest.raises(ToolSecurityError):
        reg.handler_for("write_file")({"path": "link", "content": "clobbered"})
    assert outside.read_text() == "top secret"  # nothing written through the link


def test_symlink_rejected_even_when_target_is_inside_root(tmp_path: Path) -> None:
    # A link whose *current* target is inside the jail is still rejected: it could
    # be retargeted outside between the check and the actual open (TOCTOU).
    (tmp_path / "real.txt").write_text("data")
    (tmp_path / "link.txt").symlink_to(tmp_path / "real.txt")
    reg = _registry(tmp_path)
    with pytest.raises(ToolSecurityError):
        reg.handler_for("read_file")({"path": "link.txt"})
    with pytest.raises(ToolSecurityError):
        reg.handler_for("write_file")({"path": "link.txt", "content": "x"})
    assert (tmp_path / "real.txt").read_text() == "data"


def test_symlinked_directory_component_is_rejected(tmp_path: Path) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    (outside_dir / "f.txt").write_text("secret")
    root = tmp_path / "jail"
    root.mkdir()
    (root / "sub").symlink_to(outside_dir)
    reg = _registry(root)
    with pytest.raises(ToolSecurityError):
        reg.handler_for("read_file")({"path": "sub/f.txt"})
    with pytest.raises(ToolSecurityError):
        reg.handler_for("write_file")({"path": "sub/f.txt", "content": "x"})
