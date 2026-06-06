"""Tests for the Studio Approvals inbox (HITL pause/list/redaction)."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_approvals as sa
from himmy.api.studio_runs import reset_run_store
from himmy.runtime.checkpoint import (
    AWAITING_APPROVAL,
    AgentCheckpoint,
    PendingToolCall,
)


@pytest.fixture
def fresh_stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    sa.reset_checkpoint_store()
    reset_run_store()
    yield
    sa.reset_checkpoint_store()


def _save_pending(**kw) -> AgentCheckpoint:
    cp = AgentCheckpoint(
        status=AWAITING_APPROVAL,
        pending_tool_calls=[
            PendingToolCall(
                tool_call_id="c1", tool_name="send_email", args=kw.get("args", {})
            )
        ],
        thread={"messages": [{"role": "user", "content": "email bob"}]},
    )
    sa.get_checkpoint_store().save(cp)
    return cp


def test_list_pending_returns_saved_checkpoint(fresh_stores: None) -> None:
    cp = _save_pending()
    pending = sa.list_pending()
    assert [p.checkpoint_id for p in pending] == [cp.checkpoint_id]
    assert pending[0].tools == ["send_email"]
    assert pending[0].status == AWAITING_APPROVAL


def test_detail_redacts_secret_args(fresh_stores: None) -> None:
    cp = _save_pending(args={"to": "bob@x.com", "api_token": "sekret", "body": "hi"})
    detail = sa.get_detail(cp.checkpoint_id)
    assert detail is not None
    call = detail.pending_tool_calls[0]
    assert call.args["to"] == "bob@x.com"  # non-secret kept
    assert call.args["api_token"] == "••••"  # secret-looking key masked
    assert detail.thread_preview[-1]["content"] == "email bob"


def test_detail_unknown_returns_none(fresh_stores: None) -> None:
    assert sa.get_detail("nope") is None


def test_resolve_unknown_checkpoint_errors(fresh_stores: None) -> None:
    from tests.conftest import run_async

    async def _collect():
        return [f async for f in sa.resolve("nope", approved=True)]

    frames = run_async(_collect())
    assert any(f["type"] == "error" for f in frames)
