"""``build_runtime_for_spec`` picks storage by the server-context flag (durable-defaults).

Outside a server context (the default — every CLI/test path) it wires the in-memory
``StorageService``; inside a server context it wires the durable file-backed
``SqliteStorageService``. ``HIMMY_STORE_PATH`` is steered into ``tmp_path`` so nothing is
written to the repo, and the server flag is always reset after each case.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.config.agent_spec import AgentSpec
from himmy.runtime.from_spec import build_runtime_for_spec
from himmy.services.storage.factory import (
    reset_server_context,
    set_server_context,
)
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService


@pytest.fixture(autouse=True)
def _store_in_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep any durable store under ``tmp_path`` and clear any exported DSN."""
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "storage.db"))


def test_cli_context_wires_in_memory_storage() -> None:
    """With no server context, the runtime's storage is the in-memory ``StorageService``."""
    runtime, _registry = build_runtime_for_spec(AgentSpec(name="t"))
    assert isinstance(runtime.memory_store, StorageService)
    assert not isinstance(runtime.memory_store, SqliteStorageService)


def test_server_context_wires_durable_storage(tmp_path: Path) -> None:
    """Inside a server context, the runtime's storage is the durable SQLite backend."""
    token = set_server_context(True)
    try:
        runtime, _registry = build_runtime_for_spec(AgentSpec(name="t"))
        assert isinstance(runtime.memory_store, SqliteStorageService)
        assert runtime.memory_store.path == str(tmp_path / "storage.db")
    finally:
        reset_server_context(token)
    # After reset the default in-memory wiring returns.
    runtime2, _ = build_runtime_for_spec(AgentSpec(name="t"))
    assert isinstance(runtime2.memory_store, StorageService)
    assert not isinstance(runtime2.memory_store, SqliteStorageService)


def test_server_context_memory_spec_reuses_durable_storage(tmp_path: Path) -> None:
    """A ``memory: true`` spec inside a server context still gets the durable backend."""
    token = set_server_context(True)
    try:
        runtime, _registry = build_runtime_for_spec(AgentSpec(name="t", memory=True))
        # The runtime memory_store (thread/event backbone) is the durable SQLite store.
        assert isinstance(runtime.memory_store, SqliteStorageService)
    finally:
        reset_server_context(token)
