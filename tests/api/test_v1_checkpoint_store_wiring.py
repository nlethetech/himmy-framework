"""T2f — /v1's OWN HITL checkpoint store is wired per-surface, durable on a server.

* The zero-config offline default uses an IN-MEMORY checkpoint store (no file written),
  so ``build_default`` stays a pure in-RAM, no-side-effect call.
* The durable opt-in (server context) wires a file-backed ``SqliteCheckpointStore`` at
  ``.himmy/v1_approvals.db`` — DELIBERATELY a DIFFERENT file than Studio's
  ``.himmy/approvals.db`` (per-surface inboxes; the reviewer must_fix against a drop-in
  shared inbox).
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.deps import ApiContainer
from himmy.runtime.checkpoint import InMemoryCheckpointStore, SqliteCheckpointStore


def test_offline_default_uses_in_memory_checkpoint_store(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The offline ``build_default`` wires an in-memory HITL store (no file)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.delenv("HIMMY_DURABLE_STORAGE", raising=False)
    container = ApiContainer.build_default()
    assert isinstance(container.checkpoint_store, InMemoryCheckpointStore)
    assert not (tmp_path / ".himmy" / "v1_approvals.db").exists()


def test_durable_opt_in_wires_v1_approvals_db(tmp_path: Any, monkeypatch: Any) -> None:
    """The durable server path wires a SqliteCheckpointStore at .himmy/v1_approvals.db."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    monkeypatch.setenv("HIMMY_DURABLE_STORAGE", "1")
    app = create_app()
    with TestClient(app):
        store = app.state.container.checkpoint_store
        assert isinstance(store, SqliteCheckpointStore)
    # The /v1 inbox is its OWN file, NOT Studio's approvals.db (per-surface, must_fix).
    assert (tmp_path / ".himmy" / "v1_approvals.db").exists()


def test_v1_inbox_is_distinct_from_studio_inbox(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """/v1 and Studio resume from different checkpoint files (per-surface inboxes)."""
    from himmy.api.studio_approvals import _db_path as studio_db_path
    from himmy.config.project import v1_approvals_db_path

    monkeypatch.chdir(tmp_path)
    assert v1_approvals_db_path() != studio_db_path()
    assert v1_approvals_db_path().endswith("v1_approvals.db")
    assert studio_db_path().endswith("approvals.db")
