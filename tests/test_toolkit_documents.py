"""Tests for the documents pack: read_document over text/markdown files."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.documents import register_documents_pack


def _registry(root: Path) -> ToolRegistry:
    registry = ToolRegistry()
    register_documents_pack(registry, ToolkitConfig(fs_root=root))
    return registry


def test_reads_text_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello document body")
    out = _registry(tmp_path).handler_for("read_document")({"path": "a.txt"})
    assert "hello document body" in out["text"]
    assert out["chars"] == len("hello document body")


def test_reads_markdown(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Title\n\nbody text")
    out = _registry(tmp_path).handler_for("read_document")({"path": "a.md"})
    assert "body text" in out["text"]


def test_truncates(tmp_path: Path) -> None:
    (tmp_path / "big.txt").write_text("x" * 100)
    out = _registry(tmp_path).handler_for("read_document")(
        {"path": "big.txt", "max_chars": 10}
    )
    assert len(out["text"]) == 10
    assert out["truncated"] is True


def test_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(ToolSecurityError):
        _registry(tmp_path).handler_for("read_document")({"path": "../x.txt"})


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _registry(tmp_path).handler_for("read_document")({"path": "nope.txt"})
