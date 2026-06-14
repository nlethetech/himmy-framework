"""Studio Approvals: the human-in-the-loop inbox.

When a Studio run calls an approval-gated tool (e.g. ``send_email``), the runtime
pauses into an :class:`~himmy.runtime.checkpoint.AgentCheckpoint` instead of running
it. This module lists those pending checkpoints, exposes one for review (with secret
args redacted), and resumes the run — approving or rejecting the pending call — while
streaming the continuation back to the GUI.

The durable checkpoint store lives at ``.himmy/approvals.db`` (mirrors the run store's
cwd-keyed singleton so test isolation works across directory changes).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from himmy.runtime.checkpoint import (
    AWAITING_APPROVAL,
    RESOLVING,
    SqliteCheckpointStore,
    redact_tool_args,
)

#: The checkpoint statuses a human can still act on from the inbox: a fresh
#: ``awaiting_approval`` AND a ``resolving`` row left stranded by a resume that
#: crashed mid-flight. ``claim()`` (the real concurrency gate in
#: ``resume_agent_loop``) re-claims both, so the crash-retry the new RESOLVING design
#: explicitly supports stays reachable from the only shipped human surface — without
#: it a crashed resume would be permanently stuck (hidden by ``list_pending`` and
#: refused by ``resolve``).
_RESUMABLE = (AWAITING_APPROVAL, RESOLVING)

#: Back-compat alias: the canonical redaction now lives in ``himmy.runtime.checkpoint``
#: (surface-neutral, shared with /v1). Kept so existing imports of ``_redact_args`` work.
_redact_args = redact_tool_args


# ---- store singleton (cwd-keyed, like get_run_store) --------------------

_STORE: SqliteCheckpointStore | None = None
_STORE_PATH: str | None = None


def _db_path() -> str:
    d = Path(".himmy")
    d.mkdir(exist_ok=True)
    return str(d / "approvals.db")


def get_checkpoint_store() -> SqliteCheckpointStore:
    """Process-wide durable checkpoint store, reopened if the project root changed."""
    global _STORE, _STORE_PATH
    path = _db_path()
    if _STORE is None or _STORE_PATH != path:
        if _STORE is not None:
            _STORE.close()
        # K2: route through the one aux-store selector. The tenant-scoped Postgres mirror
        # for HITL checkpoints lands in K3; until then the Postgres builder is None and
        # this resolves to the durable SQLite checkpoint store byte-for-byte.
        from himmy.services.storage.aux_store_factory import select_aux_store

        _STORE = select_aux_store(lambda: SqliteCheckpointStore(path))
        _STORE_PATH = path
    return _STORE


def reset_checkpoint_store() -> None:
    """Drop the cached store handle (tests change cwd between cases)."""
    global _STORE, _STORE_PATH
    if _STORE is not None:
        _STORE.close()
    _STORE = None
    _STORE_PATH = None


# ---- response models ----------------------------------------------------


class PendingToolView(BaseModel):
    tool_name: str
    args: dict[str, Any]


class ApprovalSummary(BaseModel):
    checkpoint_id: str
    status: str
    created_at: str
    tools: list[str]
    run_id: str | None = None
    agent: str | None = None
    prompt: str = ""


class ApprovalDetail(ApprovalSummary):
    pending_tool_calls: list[PendingToolView] = []
    thread_preview: list[dict[str, str]] = []


# ---- service ------------------------------------------------------------


def _summary(cp: Any) -> ApprovalSummary:
    from himmy.api.studio_runs import get_run_store

    run = get_run_store().get_by_checkpoint(cp.checkpoint_id)
    return ApprovalSummary(
        checkpoint_id=cp.checkpoint_id,
        status=cp.status,
        created_at=cp.created_at,
        tools=[p.tool_name for p in cp.pending_tool_calls],
        run_id=run.id if run else None,
        agent=run.agent_name if run else None,
        prompt=run.prompt if run else "",
    )


def list_pending() -> list[ApprovalSummary]:
    """All checkpoints a human can still act on (newest first).

    Includes fresh ``awaiting_approval`` checkpoints AND any ``resolving`` row left
    stranded by a resume that crashed mid-flight — those are still re-claimable by
    ``resume_agent_loop`` (so retryable via :func:`resolve`), and hiding them would
    strand the run with no human surface to recover it.
    """
    store = get_checkpoint_store()
    pending = [cp for status in _RESUMABLE for cp in store.list_by_status(status)]
    pending.sort(key=lambda cp: cp.created_at, reverse=True)
    return [_summary(cp) for cp in pending]


def get_detail(checkpoint_id: str) -> ApprovalDetail | None:
    cp = get_checkpoint_store().load(checkpoint_id)
    if cp is None:
        return None
    base = _summary(cp)
    messages = (cp.thread or {}).get("messages") or []
    preview = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))[:400]}
        for m in messages[-4:]
        if isinstance(m, dict)
    ]
    return ApprovalDetail(
        **base.model_dump(),
        pending_tool_calls=[
            PendingToolView(tool_name=p.tool_name, args=_redact_args(p.args))
            for p in cp.pending_tool_calls
        ],
        thread_preview=preview,
    )


async def resolve(
    checkpoint_id: str, *, approved: bool, actor: str = "human"
) -> AsyncIterator[dict[str, Any]]:
    """Approve/reject a pending checkpoint and stream the resumed run's frames.

    Rebuilds a runtime from the paused run's spec (so the approved tool actually
    executes), calls ``resume_agent_loop``, and persists the continuation as a new
    run linked by ``checkpoint_id``.
    """
    import asyncio
    import time

    from himmy.api import studio_service as ss
    from himmy.api.studio_canonical import resolve_canonical_storage
    from himmy.api.studio_runs import get_run_store
    from himmy.core.ids import new_uuid
    from himmy.runtime import from_spec

    store = get_checkpoint_store()
    cp = store.load(checkpoint_id)
    if cp is None:
        yield {"type": "error", "message": "unknown approval"}
        return
    # Accept ``resolving`` here too: a resume that crashed mid-flight leaves the row
    # at ``resolving``, and ``resume_agent_loop``'s atomic ``claim()`` re-claims it —
    # that claim, not this pre-check, is the real concurrency gate. Refusing it here
    # would make the crash-retry the RESOLVING design supports unreachable from the
    # only shipped human surface. An already-``approved``/``rejected`` row is terminal.
    if cp.status not in _RESUMABLE:
        yield {"type": "error", "message": f"already {cp.status}"}
        return

    paused = get_run_store().get_by_checkpoint(checkpoint_id)
    agent_path = paused.agent_path if paused else None
    provider = paused.provider if paused else None
    model = paused.model if paused else None
    if not agent_path:
        yield {"type": "error", "message": "cannot locate the paused run's agent"}
        return

    collected: list[Any] = []
    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def _on_event(event: Any) -> None:
        collected.append(event)
        await queue.put(event)

    spec = ss.load_studio_spec(agent_path, provider=provider, model=model)
    runtime, registry = await asyncio.to_thread(
        lambda: from_spec.build_runtime_for_spec(
            spec,
            provider=provider,
            model=model,
            on_event=_on_event,
            capture_io=True,
            checkpoint_store=store,
            durable_defaults=True,
        )
    )
    cog = ss._Cognition(ss._read_only_map(registry), spec.name)

    run_id = new_uuid()
    started = time.monotonic()
    output_text = ""
    status = "ok"
    error_msg: str | None = None
    next_checkpoint: str | None = None

    yield {"type": "start", "agent": spec.name, "streaming": False, "resumed": True}
    try:
        run_task = asyncio.create_task(
            runtime.resume_agent_loop(checkpoint_id, approved=approved, actor=actor)
        )
        async for frame in ss._drain_cognition(queue, run_task, cog):
            yield frame
        loop = await run_task
        if loop.stopped_reason == "awaiting_approval":
            status = "awaiting_approval"
            next_checkpoint = loop.checkpoint_id
        else:
            output_text = loop.final.output_text or ""
            yield {"type": "message", "text": output_text}
    except Exception as exc:  # noqa: BLE001
        status = "error"
        error_msg = str(exc)

    duration_ms = (time.monotonic() - started) * 1000.0
    # The studio.db cache write is synchronous (worker thread); the canonical mirror
    # (T2.2) is then upserted on the main loop so the resume's terminal/paused status
    # is reflected in the ONE store /v1 + the CLI read — the last-write-wins guard keeps
    # an authoritative AWAITING_APPROVAL from being rolled back by a stale projection.
    built: Any | None = None
    try:
        built = await asyncio.to_thread(
            ss._record_run,
            run_id=run_id,
            spec=spec,
            agent_path=agent_path,
            provider=provider,
            model=model,
            prompt=f"[resumed {'approve' if approved else 'reject'} {checkpoint_id[:8]}]",
            history=[],
            output=output_text,
            tools=cog.tools_used,
            events=collected,
            steps=cog.steps,
            usage=ss._usage_of(cog),
            status=status,
            error=error_msg,
            duration_ms=duration_ms,
            thread_id=None,
            checkpoint_id=next_checkpoint,
        )
    except Exception:  # noqa: BLE001
        pass
    await ss._record_canonical(resolve_canonical_storage(), built)

    if status == "error":
        yield {
            "type": "error",
            "message": error_msg or "resume failed",
            "run_id": run_id,
        }
    elif status == "awaiting_approval":
        yield {"type": "paused", "checkpoint_id": next_checkpoint, "run_id": run_id}
    else:
        yield {
            "type": "done",
            "output_text": output_text,
            "run_id": run_id,
            "succeeded": True,
        }


__all__ = [
    "ApprovalSummary",
    "ApprovalDetail",
    "PendingToolView",
    "get_checkpoint_store",
    "reset_checkpoint_store",
    "list_pending",
    "get_detail",
    "resolve",
]
