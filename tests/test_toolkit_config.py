"""Tests for ToolkitConfig env plumbing (the newer knobs)."""

from __future__ import annotations

import pytest

from himmy.toolkit.config import ToolkitConfig


def test_from_env_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With nothing set, the durable/memory knobs default to off/in-process."""
    for key in ("HIMMY_KB_DSN", "HIMMY_MEMORY_PATH", "HIMMY_MEMORY_SUBJECT"):
        monkeypatch.delenv(key, raising=False)
    cfg = ToolkitConfig.from_env()
    assert cfg.kb_dsn is None
    assert cfg.memory_path is None
    assert cfg.memory_subject == "default"


def test_from_env_reads_durable_and_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """The env vars flow into the config."""
    monkeypatch.setenv("HIMMY_KB_DSN", "postgresql://x/y")
    monkeypatch.setenv("HIMMY_MEMORY_PATH", "/tmp/mem.db")
    monkeypatch.setenv("HIMMY_MEMORY_SUBJECT", "boss")
    cfg = ToolkitConfig.from_env()
    assert cfg.kb_dsn == "postgresql://x/y"
    assert cfg.memory_path == "/tmp/mem.db"
    assert cfg.memory_subject == "boss"
