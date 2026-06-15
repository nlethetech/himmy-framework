"""Tests for the Studio Approvals inbox (HITL pause/list/redaction)."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.api import studio_approvals as sa
from himmy.api.studio_runs import reset_run_store
from himmy.runtime.checkpoint import (
    APPROVED,
    AWAITING_APPROVAL,
    RESOLVING,
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


def _save_pending(*, status: str = AWAITING_APPROVAL, **kw) -> AgentCheckpoint:
    cp = AgentCheckpoint(
        status=status,
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


def test_detail_redacts_secret_nested_in_json_body(fresh_stores: None) -> None:
    """A secret buried inside a nested json_body is masked in the approver-facing view.

    Regression: the approver inbox once redacted only top-level keys, so a credential
    nested one level deep leaked in plaintext to the human (and diverged from the audit
    spine's mask). The shared canonical redactor recurses, so it is masked here too.
    """
    cp = _save_pending(
        args={
            "method": "POST",
            "json_body": {"user": "bob", "config": {"api_key": "sk-live-DEEP"}},
        }
    )
    detail = sa.get_detail(cp.checkpoint_id)
    assert detail is not None
    call = detail.pending_tool_calls[0]
    assert call.args["json_body"]["user"] == "bob"  # benign nested value kept
    # The credential sits two levels deep under a BENIGN container key, so only a
    # recursive walk reaches it — top-level-only redaction would leak it in plaintext.
    assert call.args["json_body"]["config"]["api_key"] == "••••"  # nested secret masked


def test_detail_unknown_returns_none(fresh_stores: None) -> None:
    assert sa.get_detail("nope") is None


def test_resolve_unknown_checkpoint_errors(fresh_stores: None) -> None:
    from tests.conftest import run_async

    async def _collect():
        return [f async for f in sa.resolve("nope", approved=True)]

    frames = run_async(_collect())
    assert any(f["type"] == "error" for f in frames)


def test_list_pending_surfaces_stranded_resolving_row(fresh_stores: None) -> None:
    """A 'resolving' checkpoint (resume crashed mid-flight) stays visible in the inbox.

    Regression: the new RESOLVING claim-state used to hide a crash-stranded row from
    the only human surface — list_pending listed AWAITING_APPROVAL only — so the run
    became unrecoverable. It must surface so a human can retry it.
    """
    fresh = _save_pending()
    stuck = _save_pending(status=RESOLVING)

    pending = sa.list_pending()
    ids = {p.checkpoint_id for p in pending}
    assert fresh.checkpoint_id in ids
    assert stuck.checkpoint_id in ids
    stuck_summary = next(p for p in pending if p.checkpoint_id == stuck.checkpoint_id)
    assert stuck_summary.status == RESOLVING


def test_resolve_accepts_resolving_checkpoint(fresh_stores: None) -> None:
    """A 'resolving' checkpoint is treated as resumable by resolve() (not refused).

    Regression: resolve() refused anything != AWAITING_APPROVAL with 'already
    resolving', so a crash-stranded RESOLVING row — which resume_agent_loop's atomic
    claim() explicitly re-claims — was unreachable from Studio. The pre-check must let
    it through; the atomic claim remains the real concurrency gate. (Here no paused run
    is recorded, so it proceeds PAST the status gate to the agent-lookup error — which
    is exactly the proof the 'already resolving' refusal no longer fires.)
    """
    from tests.conftest import run_async

    stuck = _save_pending(status=RESOLVING)

    async def _collect():
        return [f async for f in sa.resolve(stuck.checkpoint_id, approved=True)]

    frames = run_async(_collect())
    messages = [f.get("message", "") for f in frames if f["type"] == "error"]
    # The old code emitted 'already resolving' and stopped; the fix gets past that.
    assert not any("already resolving" in m for m in messages)
    assert any("cannot locate the paused run's agent" in m for m in messages)


def test_resolve_still_refuses_terminal_checkpoint(fresh_stores: None) -> None:
    """An already-approved/rejected checkpoint stays terminal — resolve() refuses it."""
    from tests.conftest import run_async

    done = _save_pending(status=APPROVED)

    async def _collect():
        return [f async for f in sa.resolve(done.checkpoint_id, approved=True)]

    frames = run_async(_collect())
    assert any(
        f["type"] == "error" and "already approved" in f.get("message", "")
        for f in frames
    )
