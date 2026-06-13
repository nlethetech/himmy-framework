"""API kernel: the ``/v1/threads`` router — agentic conversation continuation (T2g).

Today ``/v1`` can only REPLAY a thread (``GET /v1/runs/{id}/thread``, read-only). This
router makes a conversation a first-class, workspace-scoped, durable resource:

* ``POST /v1/threads`` creates a workspace-stamped thread (optionally bound to a default
  ``agent_id`` it continues with);
* ``POST /v1/threads/{id}/messages`` loads the authoritative
  :class:`~himmy.agents.base_agent.thread.ChatThread`, appends the user turn, and runs it
  on the PER-RUN AGENTIC runtime (tools + HITL/plan pause) — NOT a tool-less single-shot
  ``run_task_detailed`` masquerading as chat (reviewer must_fix). The updated thread is
  persisted back to the ONE ``.himmy/conversations.db`` the CLI + Studio read, and a new
  canonical :class:`~himmy.services.storage.models.RunRecord` is linked by thread_id;
* ``GET /v1/threads/{id}`` replays the flat (user/agent) projection;
* ``GET /v1/threads`` is omitted on purpose (a per-tenant conversation INDEX over the
  single-user-local ConversationStore is a future item; threads are reached by id here).

Plan mode: ``plan=true`` on a message pauses the run at PLAN-READY (the published plan in
``run.metadata['plan']``), resumable via the SAME ``POST /v1/runs/{id}/approve`` machinery
the HITL gate uses (T2f) — so plan mode is programmatically reachable.

Security: ``run:read`` for GET, ``run:write`` for create/continue (threads ride the
``run`` RBAC resource — a conversation is a run container, not a separate resource).
Workspace ownership is enforced on every route via the thread's stored
``metadata['workspace_id']`` (a cross-tenant ``thread_id`` is a clean 404, never a leak).
The tenant-spec sanitizer (T0.3) still bites: a continuation's stored spec runs under the
caller's operator status, so a tenant can never reach privileged tools.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from himmy.api.auth import (
    get_principal,
    require_permission,
    require_workspace,
    resolve_workspace,
)
from himmy.api.models import NOT_FOUND_RESPONSE
from himmy.api.security_audit import audit_event
from himmy.application.services import (
    HitlNotSupportedError,
    HitlRequiresAgentError,
    ThreadAgentRequiredError,
    ThreadNotFoundError,
)
from himmy.services.storage.models import RunRecord

router = APIRouter(prefix="/v1/threads", tags=["threads"])

_READ = [Depends(require_permission("run", "read"))]
_WRITE = [Depends(require_permission("run", "write"))]


class CreateThreadRequest(BaseModel):
    """The POST /v1/threads body: the owning workspace + an optional default agent."""

    workspace_id: str
    agent_id: str | None = None
    title: str | None = None


class ThreadView(BaseModel):
    """The API view of a created/continued thread (no message bodies)."""

    thread_id: str
    workspace_id: str
    agent_id: str | None = None
    message_count: int = 0


class FlatMessageView(BaseModel):
    """One row of the flat (user/agent) transcript both UIs read."""

    role: str
    text: str


class ThreadMessagesView(BaseModel):
    """The GET /v1/threads/{id} envelope: the flat transcript + ownership."""

    thread_id: str
    workspace_id: str | None = None
    agent_id: str | None = None
    messages: list[FlatMessageView] = []


class PostMessageRequest(BaseModel):
    """The POST /v1/threads/{id}/messages body: the user turn + run knobs.

    ``agent_id`` overrides the thread's default agent for this turn (else the thread's
    bound agent is used). ``hitl`` pauses the run at an approval-gated tool; ``plan``
    pauses it at PLAN-READY with the published plan in ``run.metadata['plan']``. Both
    resume via ``POST /v1/runs/{id}/approve``. ``idempotency_key`` makes a re-POST return
    the prior run instead of running the turn twice.
    """

    workspace_id: str | None = None
    subject_id: str = "default"
    prompt: str
    agent_id: str | None = None
    idempotency_key: str | None = None
    hitl: bool = False
    plan: bool = False


def _container(request: Request) -> Any:
    """Pull the wired :class:`ApiContainer` off the app state."""
    return request.app.state.container


def _thread_app(request: Request) -> Any:
    """Pull the wired :class:`ThreadAppService` off the app container."""
    return _container(request).thread_app


def _operator_provisioned(request: Request) -> bool:
    """Whether THIS request may carry operator-only spec fields (mirrors agents.py)."""
    return bool(getattr(get_principal(request), "all_tenants", False))


@router.post("", response_model=ThreadView, dependencies=_WRITE, status_code=201)
async def create_thread(body: CreateThreadRequest, request: Request) -> ThreadView:
    """Create an empty, workspace-stamped thread (T2g).

    When ``agent_id`` is given the thread records that agent as its default, so a
    later ``POST /{id}/messages`` continues it without re-supplying the agent. The agent
    is NOT resolved here (it is resolved + sanitized at message time).
    """
    workspace_id = require_workspace(request, body.workspace_id)
    thread = _thread_app(request).create_thread(
        workspace_id=workspace_id,
        agent_id=body.agent_id,
        title=body.title,
    )
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="create",
        workspace_id=workspace_id,
        detail=f"thread {thread.thread_id} created",
    )
    return ThreadView(
        thread_id=thread.thread_id,
        workspace_id=workspace_id,
        agent_id=body.agent_id,
        message_count=0,
    )


@router.get(
    "/{thread_id}",
    response_model=ThreadMessagesView,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_READ,
)
async def get_thread(
    thread_id: str, request: Request, workspace_id: str | None = None
) -> ThreadMessagesView:
    """Replay the flat (user/agent) projection of an owned thread (404 cross-tenant)."""
    workspace_id = resolve_workspace(request, workspace_id)
    thread_app = _thread_app(request)
    try:
        thread = thread_app.load_owned_thread(thread_id, workspace_id=workspace_id)
        flat = thread_app.flat_messages(thread_id, workspace_id=workspace_id)
    except ThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail="thread not found") from exc
    from himmy.application.services import THREAD_AGENT_KEY

    return ThreadMessagesView(
        thread_id=thread_id,
        workspace_id=workspace_id,
        agent_id=(thread.metadata or {}).get(THREAD_AGENT_KEY),
        messages=[FlatMessageView(role=m.role, text=m.text) for m in flat],
    )


@router.post(
    "/{thread_id}/messages",
    response_model=RunRecord,
    responses=NOT_FOUND_RESPONSE,
    dependencies=_WRITE,
)
async def post_message(
    thread_id: str, body: PostMessageRequest, request: Request
) -> RunRecord:
    """Append a user turn to an owned thread and run it agentically (T2g).

    The turn runs on the per-run tool-bearing runtime (tools available + HITL/plan
    pauses), the updated thread is persisted to the shared ConversationStore, and a new
    linked :class:`RunRecord` is returned (RUNNING/QUEUED — fire-and-forget like
    ``create_run``; poll ``GET /v1/runs/{id}`` for its outcome). A cross-tenant
    ``thread_id`` is a 404; a thread with no resolvable agent is a 422.
    """
    workspace_id = require_workspace(request, body.workspace_id or "")
    try:
        run = cast(
            RunRecord,
            await _thread_app(request).append_message(
                conversation_id=thread_id,
                workspace_id=workspace_id,
                subject_id=body.subject_id,
                prompt=body.prompt,
                agent_id=body.agent_id,
                idempotency_key=body.idempotency_key,
                actor=get_principal(request).actor_metadata(),
                operator_provisioned=_operator_provisioned(request),
                hitl=body.hitl,
                plan=body.plan,
            ),
        )
    except ThreadNotFoundError as exc:
        raise HTTPException(status_code=404, detail="thread not found") from exc
    except ThreadAgentRequiredError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "thread continuation requires a resolvable agent_id "
                "(bind one at create or supply it on the message)"
            ),
        ) from exc
    except HitlRequiresAgentError as exc:  # pragma: no cover - guarded above
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except HitlNotSupportedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    audit_event(
        request,
        event_type="access",
        outcome="allow",
        resource="run",
        action="create",
        workspace_id=workspace_id,
        detail=f"thread {thread_id} continued by run {run.run_id}",
    )
    return run


__all__ = ["router"]
