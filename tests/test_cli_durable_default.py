"""``himmy run`` stays in-memory (zero-setup), even with a DSN/store path exported.

The one-shot CLI must never opt into a durable backend: a developer with a production
``HIMMY_DATABASE_URL`` exported still gets a clean ephemeral store for a quick run. This
verifies the CLI level — both the direct one-shot store builder and an end-to-end
``main(["run", ...])`` invocation against the offline stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from himmy.cli.__main__ import main
from himmy.services.storage.service import StorageService
from himmy.services.storage.sqlite import SqliteStorageService


@pytest.fixture(autouse=True)
def _hostile_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Export a DSN + store path to prove the CLI ignores them and stays in-memory."""
    monkeypatch.setenv("HIMMY_DATABASE_URL", "postgres://user@host/db")
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "should_not_exist.db"))


def test_build_storage_one_shot_is_in_memory() -> None:
    """The one-shot CLI store builder is in-memory regardless of exported env."""
    from himmy.runtime.builder import build_storage

    store = build_storage()
    assert isinstance(store, StorageService)
    assert not isinstance(store, SqliteStorageService)


def test_cli_run_wires_in_memory_storage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An end-to-end ``himmy run`` wires the in-memory store (no server context)."""
    captured: dict[str, Any] = {}

    import himmy.cli.commands as commands

    real = commands.from_spec.build_runtime_for_spec

    def _spy(spec: Any, **kwargs: Any) -> Any:
        runtime, registry = real(spec, **kwargs)
        captured["memory_store"] = runtime.memory_store
        return runtime, registry

    monkeypatch.setattr(commands.from_spec, "build_runtime_for_spec", _spy)

    code = main(["run", "--provider", "stub", "--name", "t", "-p", "hello there"])
    assert code == 0
    store = captured.get("memory_store")
    assert isinstance(store, StorageService)
    assert not isinstance(store, SqliteStorageService)
    # The CLI never created the durable store file.
    assert not (tmp_path / "should_not_exist.db").exists()
