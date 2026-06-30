"""rbac-harden(tool-store-tenancy): the HITL approval-resume re-run re-scopes its packs.

``studio_approvals.resolve`` rebuilds a runtime from a paused run's spec and re-runs it. The
re-run must scope its memory + knowledge tool packs to the SAME tenant the run paused under —
otherwise an approved tool on a shared durable store reads/writes the static shared scope, a
cross-tenant leak. The owning ``workspace_id`` rides durably through the checkpoint's
``ctx['context_metadata']['workspace_id']`` (the place the original run stamped it); the
resume re-derives it via :func:`_checkpoint_owner_workspace` and threads it as ``subject=``.
"""

from __future__ import annotations

from himmy.api.studio_approvals import (
    _checkpoint_owner_subject_scope,
    _checkpoint_owner_workspace,
)
from himmy.runtime.checkpoint import AgentCheckpoint


def test_owner_workspace_recovered_from_checkpoint_ctx() -> None:
    """A tenant-scoped paused run yields its owning workspace from the checkpoint ctx."""
    cp = AgentCheckpoint(ctx={"context_metadata": {"workspace_id": "tenant-7"}})
    assert _checkpoint_owner_workspace(cp) == "tenant-7"


def test_owner_workspace_none_for_offline_checkpoint() -> None:
    """Offline / single-box invariant: a checkpoint with no workspace yields ``None``."""
    assert _checkpoint_owner_workspace(AgentCheckpoint()) is None
    assert (
        _checkpoint_owner_workspace(AgentCheckpoint(ctx={"context_metadata": {}}))
        is None
    )
    assert _checkpoint_owner_workspace(AgentCheckpoint(ctx={})) is None


def test_owner_workspace_handles_malformed_ctx() -> None:
    """A non-dict / absent context_metadata never raises — falls back to unscoped."""

    class _Bare:
        ctx = None

    assert _checkpoint_owner_workspace(_Bare()) is None
    assert _checkpoint_owner_workspace(object()) is None
    cp = AgentCheckpoint(ctx={"context_metadata": "not-a-dict"})  # type: ignore[dict-item]
    assert _checkpoint_owner_workspace(cp) is None


# ---- rbac-harden(mopup-r1): the resume must also recover the within-tenant subject axis


def test_owner_subject_scope_recovered_from_checkpoint_ctx() -> None:
    """A subject_scoped paused run yields its data subject from the checkpoint ctx.

    Regression for the dropped subject axis on resume: the resume threaded only the tenant
    (``subject=workspace``) and never the within-tenant USER axis, so a subject_scoped run's
    resumed segment collapsed two users onto one memory/KB namespace (``t:<tenant>:default``).
    The originating run now stamps ``subject_scope`` into the checkpoint and the resume
    re-derives it here.
    """
    cp = AgentCheckpoint(
        ctx={"context_metadata": {"workspace_id": "tenant-7", "subject_scope": "userA"}}
    )
    assert _checkpoint_owner_subject_scope(cp) == "userA"


def test_owner_subject_scope_none_for_offline_checkpoint() -> None:
    """Offline / non-subject-scoped invariant: no subject_scope carried yields ``None``."""
    assert _checkpoint_owner_subject_scope(AgentCheckpoint()) is None
    assert (
        _checkpoint_owner_subject_scope(AgentCheckpoint(ctx={"context_metadata": {}}))
        is None
    )
    assert _checkpoint_owner_subject_scope(AgentCheckpoint(ctx={})) is None


def test_owner_subject_scope_handles_malformed_ctx() -> None:
    """A non-dict / absent context_metadata never raises — falls back to unscoped."""

    class _Bare:
        ctx = None

    assert _checkpoint_owner_subject_scope(_Bare()) is None
    assert _checkpoint_owner_subject_scope(object()) is None
    cp = AgentCheckpoint(ctx={"context_metadata": "not-a-dict"})  # type: ignore[dict-item]
    assert _checkpoint_owner_subject_scope(cp) is None
