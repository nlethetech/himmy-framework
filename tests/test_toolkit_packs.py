"""Tests for the toolkit pack catalog and resolver."""

from __future__ import annotations

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit import (
    BUILTIN_PACKS,
    ToolkitConfig,
    UnknownToolPackError,
    register_packs,
    resolve_packs,
)


def test_catalog_has_expected_packs() -> None:
    """The built-in catalog exposes all nine expected packs."""
    assert set(BUILTIN_PACKS) == {
        "web",
        "files",
        "data",
        "code",
        "utils",
        "knowledge",
        "documents",
        "comms",
        "data-sources",
        "memory",
    }


def test_register_all_packs_populates_registry() -> None:
    """Registering every pack wires all 18 tools with definitions + handlers."""
    registry = ToolRegistry()
    register_packs(registry, list(BUILTIN_PACKS), ToolkitConfig())
    names = {d.name for d in registry.list()}
    assert names == {
        "web_search",
        "web_fetch",
        "http_request",
        "read_file",
        "write_file",
        "list_dir",
        "sql_query",
        "run_python",
        "calculator",
        "current_time",
        "kb_ingest",
        "kb_search",
        "read_document",
        "send_email",
        "send_webhook",
        "weather",
        "geocode",
        "wikipedia",
        "remember",
        "recall",
    }
    for name in names:
        assert registry.handler_for(name) is not None


def test_pack_tool_names_match_registrations() -> None:
    """Each pack's advertised tool_names are exactly what it registers."""
    for pack in BUILTIN_PACKS.values():
        registry = ToolRegistry()
        pack.register(registry, ToolkitConfig())
        assert {d.name for d in registry.list()} == set(pack.tool_names)


def test_resolve_unknown_pack_raises() -> None:
    """An unknown pack name is a clear UnknownToolPackError."""
    with pytest.raises(UnknownToolPackError):
        resolve_packs(["web", "nope"])


def test_write_and_run_python_are_approval_gated() -> None:
    """Mutating/exec tools require approval by default."""
    registry = ToolRegistry()
    register_packs(registry, ["files", "code"], ToolkitConfig())
    assert registry.get("write_file").requires_approval is True
    assert registry.get("run_python").requires_approval is True


def test_fs_allow_write_disables_write_gate() -> None:
    """fs_allow_write makes write_file no longer require approval."""
    registry = ToolRegistry()
    register_packs(registry, ["files"], ToolkitConfig(fs_allow_write=True))
    assert registry.get("write_file").requires_approval is False
