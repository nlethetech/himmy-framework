"""Tests for ``scripts/ops_prune.py`` (the operator-driven retention entry point).

The per-store ``prune_*`` methods are unit-tested in tests/storage; these tests pin
the script that wires them together for a deployment: it must require a bound, prune
the default stores in place, keep live/unresolved work, and tolerate absent stores.
"""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType

import pytest

from himmy.agents.base_agent.thread import ChatThread
from himmy.core.events import EventType, RunEvent
from himmy.runtime.checkpoint import (
    APPROVED,
    AWAITING_APPROVAL,
    AgentCheckpoint,
    SqliteCheckpointStore,
)
from himmy.runtime.session import SqliteSessionStore
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = _REPO_ROOT / "scripts" / "ops_prune.py"
    spec = importlib.util.spec_from_file_location("ops_prune", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = _load()


def _old_iso(days: float) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def _seed(store_dir: Path) -> None:
    store_dir.mkdir(parents=True, exist_ok=True)

    events = SqliteStorageService(str(store_dir / "storage.db"))
    run_async(
        events.append_event(
            RunEvent(event_type=EventType.AGENT_RUN_STARTED, timestamp=_old_iso(120))
        )
    )
    run_async(events.append_event(RunEvent(event_type=EventType.AGENT_RUN_STARTED)))
    run_async(events.close())

    ckpt = SqliteCheckpointStore(str(store_dir / "approvals.db"))
    ckpt.save(AgentCheckpoint(status=APPROVED, created_at=_old_iso(120)))  # pruned
    ckpt.save(AgentCheckpoint(status=AWAITING_APPROVAL, created_at=_old_iso(120)))  # kept
    ckpt.close()

    sessions = SqliteSessionStore(str(store_dir / "sessions.db"))
    sessions.save("old", ChatThread(agent_id="a"))
    sessions._conn.execute(
        "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
        (_old_iso(120), "old"),
    )
    sessions._conn.commit()
    sessions.save("fresh", ChatThread(agent_id="a"))
    sessions.close()


def _args(store_dir: Path, *extra: str) -> list[str]:
    return [
        "--store-path",
        str(store_dir / "storage.db"),
        "--approvals-path",
        str(store_dir / "approvals.db"),
        "--sessions-path",
        str(store_dir / "sessions.db"),
        *extra,
    ]


def test_requires_a_bound(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        ops.main(_args(tmp_path))  # no --older-than-days / --keep-last-*


def test_prunes_old_keeps_recent_and_live(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _seed(tmp_path)
    rc = ops.main(_args(tmp_path, "--older-than-days", "90"))
    assert rc == 0
    # 1 old event + 1 resolved checkpoint + 1 old session = 3 rows removed.
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == "3"

    events = SqliteStorageService(str(tmp_path / "storage.db"))
    assert len(run_async(events.list_events())) == 1  # recent event survived
    run_async(events.close())

    ckpt = SqliteCheckpointStore(str(tmp_path / "approvals.db"))
    # The unresolved (live) checkpoint is never deleted by age.
    assert len(ckpt.list_by_status(AWAITING_APPROVAL)) == 1
    assert len(ckpt.list_by_status(APPROVED)) == 0
    ckpt.close()

    sessions = SqliteSessionStore(str(tmp_path / "sessions.db"))
    ids = {s.session_id for s in sessions.list_sessions()}
    assert ids == {"fresh"}
    sessions.close()


def test_absent_stores_are_skipped(tmp_path: Path) -> None:
    # No stores created at all — the script must no-op cleanly, not crash.
    rc = ops.main(_args(tmp_path, "--older-than-days", "90"))
    assert rc == 0
