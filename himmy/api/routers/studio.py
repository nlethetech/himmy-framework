"""API kernel: the Himmy Studio router (the local web GUI's backend).

Studio is the no-code front door to himmy: chat with an agent, build/edit an
``agent.yaml``, and browse past runs/traces — all over the same FastAPI BFF. This
router is intentionally GUI-shaped (not the tenant-scoped ``/v1`` surface): it is
meant to be served on loopback by ``himmy studio`` for a single local user.

Endpoints:
  * ``GET  /api/studio/doctor``  — environment diagnostics as JSON.
  * ``GET  /api/studio/agents``  — discover agent.yaml files in the project.
  * ``POST /api/studio/run``     — run an agent for one turn (SSE token stream).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from himmy.api import (
    studio_agents,
    studio_approvals,
    studio_connections,
    studio_service,
)
from himmy.api.studio_approvals import ApprovalDetail, ApprovalSummary
from himmy.api.studio_connections import (
    ConnectionStatus,
    ConnectionTestResult,
    ReadOnlyBackendError,
    SendResult,
)
from himmy.api.studio_runs import (
    RunAnalytics,
    StudioRun,
    StudioRunListResponse,
    get_run_store,
)

router = APIRouter(prefix="/api/studio", tags=["studio"])


@router.get("/health")
async def health() -> dict[str, Any]:
    """Fast readiness probe: version, writable secrets, and provider availability."""
    import shutil

    from himmy import __version__
    from himmy.config.secrets import get_writable_provider

    return {
        "status": "ok",
        "version": __version__,
        "secrets_writable": get_writable_provider() is not None,
        "providers": {
            "claude_cli": shutil.which("claude") is not None,
            "ollama": shutil.which("ollama") is not None,
        },
    }


@router.get("/doctor")
async def doctor() -> dict:
    """Environment diagnostics: extras, providers, keys, and the next step.

    The JSON twin of ``himmy doctor`` — same
    :func:`himmy.runtime.diagnostics.collect_doctor_report` snapshot the CLI prints.
    """
    from himmy.runtime.diagnostics import collect_doctor_report

    return collect_doctor_report().to_dict()


@router.get("/benchmarks")
async def benchmarks() -> dict:
    """Cached per-model reliability scorecards (from `himmy bench` / the probe)."""
    from himmy.api import studio_bench

    return {"entries": studio_bench.list_cached()}


@router.post("/benchmarks/probe")
async def benchmarks_probe() -> dict:
    """Run a quick tool-focused reliability check against detected local models.

    Slow (real model calls) but bounded — a small suite against ≤3 local models, one
    trial each. Caches the result so Doctor shows it without re-running.
    """
    from himmy.api import studio_bench

    return await studio_bench.run_probe()


@router.get("/agents", response_model=list[studio_service.AgentSummary])
async def list_agents() -> list[studio_service.AgentSummary]:
    """Discover single-agent specs under the project root."""
    return studio_service.list_agents()


@router.get("/teams", response_model=list[studio_service.TeamSummary])
async def list_teams() -> list[studio_service.TeamSummary]:
    """Discover multi-agent team specs (manager + workers) under the project root."""
    return studio_service.list_teams()


class TurnInput(BaseModel):
    """One prior conversation turn replayed to preserve multi-turn context."""

    role: str  # "user" | "assistant"
    content: str


class RunRequest(BaseModel):
    """POST /api/studio/run body: which agent, the prompt, and optional overrides."""

    agent_path: str = Field(..., description="agent spec path, relative to the root")
    prompt: str = Field(..., max_length=100_000)
    provider: str | None = None
    model: str | None = None
    history: list[TurnInput] = Field(default=[], max_length=200)


def _sse(event: dict[str, Any]) -> str:
    """Encode one event dict as a Server-Sent-Events ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/run")
async def run(body: RunRequest) -> StreamingResponse:
    """Run an agent for one user turn, streaming GUI events over SSE.

    Loads the selected ``agent.yaml`` (with project defaults + skills applied),
    wires it exactly like ``himmy run``, and streams ``start``/``token``/``tool``/
    ``message``/``done`` frames. A load/build failure becomes a single ``error``
    frame so the client always gets a clean end to the stream.
    """
    # Resolve + load up front so a bad path is a clean 4xx, not a mid-stream error.
    try:
        spec = studio_service.load_studio_spec(
            body.agent_path, provider=body.provider, model=body.model
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def _stream() -> AsyncIterator[str]:
        try:
            async for event in studio_service.stream_agent_run(
                spec,
                body.prompt,
                history=[t.model_dump() for t in body.history],
                provider=body.provider,
                model=body.model,
                agent_path=body.agent_path,
            ):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - surface as a terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class RunTeamRequest(BaseModel):
    """POST /api/studio/run-team body: which team and the request prompt."""

    team_path: str = Field(..., description="team spec path, relative to the root")
    prompt: str = Field(..., max_length=100_000)


@router.post("/run-team")
async def run_team(body: RunTeamRequest) -> StreamingResponse:
    """Run a multi-agent team, streaming the live routing/delegate/tool trail (SSE).

    Members run on their own providers (e.g. a Claude-CLI manager delegating to local
    Ollama workers). Frames: ``start`` → ``delegate``/``handoff``/``tool`` → ``message``
    → ``done`` (or a terminal ``error``).
    """
    try:
        spec = studio_service.load_team(body.team_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    teams = {t.path: t for t in studio_service.list_teams()}
    team_name = teams[body.team_path].name if body.team_path in teams else "team"

    async def _stream() -> AsyncIterator[str]:
        try:
            async for event in studio_service.stream_team_run(
                spec, body.prompt, team_name=team_name, team_path=body.team_path
            ):
                yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/runs", response_model=StudioRunListResponse)
async def list_runs(limit: int = 50, offset: int = 0) -> StudioRunListResponse:
    """List past Studio runs (newest first), paginated."""
    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    store = get_run_store()
    items = store.list(limit=limit, offset=offset)
    total = store.count()
    return StudioRunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        next_offset=(offset + len(items)) if offset + len(items) < total else None,
    )


@router.get("/runs/analytics", response_model=RunAnalytics)
async def runs_analytics() -> RunAnalytics:
    """Aggregate cost/token/latency stats across runs (the analytics dashboard)."""
    return get_run_store().analytics()


@router.get("/runs/{run_id}", response_model=StudioRun)
async def get_run(run_id: str) -> StudioRun:
    """Fetch one run in full: transcript, tools, and the step-by-step timeline."""
    run = get_run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


# ---- Connections (connect Email/Telegram/Web so agents can act) ---------


class ConnectionSetRequest(BaseModel):
    fields: dict[str, Any]


@router.get("/connections", response_model=list[ConnectionStatus])
async def connections() -> list[ConnectionStatus]:
    """List connectable account types + their configured/writable status."""
    return studio_connections.list_connections()


@router.get("/connections/{ctype}", response_model=ConnectionStatus)
async def connection(ctype: str) -> ConnectionStatus:
    status = studio_connections.get_connection(ctype)
    if status is None:
        raise HTTPException(status_code=404, detail="unknown connection type")
    return status


@router.put("/connections/{ctype}", response_model=ConnectionStatus)
async def set_connection(ctype: str, body: ConnectionSetRequest) -> ConnectionStatus:
    """Store a connection's fields (secrets → the writable backend; never echoed)."""
    try:
        return studio_connections.set_connection(ctype, body.fields)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc
    except ReadOnlyBackendError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete("/connections/{ctype}", response_model=ConnectionStatus)
async def delete_connection(ctype: str) -> ConnectionStatus:
    try:
        return studio_connections.delete_connection(ctype)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc
    except ReadOnlyBackendError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/connections/{ctype}/test", response_model=ConnectionTestResult)
async def test_connection(ctype: str) -> ConnectionTestResult:
    """Live-validate a connection (SMTP login / Telegram getMe / search ping)."""
    try:
        return await studio_connections.test_connection(ctype)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc


class SendRequest(BaseModel):
    payload: dict[str, Any]


@router.post("/connections/{ctype}/send", response_model=SendResult)
async def send_via_connection(ctype: str, body: SendRequest) -> SendResult:
    """Send a user-composed message directly (a Home quick action)."""
    try:
        return await studio_connections.send_via_connection(ctype, body.payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc


# ---- Approvals (human-in-the-loop inbox) --------------------------------


@router.get("/approvals", response_model=list[ApprovalSummary])
async def approvals() -> list[ApprovalSummary]:
    """List runs paused awaiting a human decision."""
    return studio_approvals.list_pending()


@router.get("/approvals/{checkpoint_id}", response_model=ApprovalDetail)
async def approval(checkpoint_id: str) -> ApprovalDetail:
    detail = studio_approvals.get_detail(checkpoint_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="approval not found")
    return detail


def _resolve_stream(checkpoint_id: str, approved: bool) -> StreamingResponse:
    async def _gen() -> AsyncIterator[str]:
        async for event in studio_approvals.resolve(checkpoint_id, approved=approved):
            yield _sse(event)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.post("/approvals/{checkpoint_id}/approve")
async def approve(checkpoint_id: str) -> StreamingResponse:
    """Approve the pending tool call and stream the resumed run."""
    return _resolve_stream(checkpoint_id, True)


@router.post("/approvals/{checkpoint_id}/reject")
async def reject(checkpoint_id: str) -> StreamingResponse:
    """Reject the pending tool call and stream the resumed run."""
    return _resolve_stream(checkpoint_id, False)


# ---- Calendar -----------------------------------------------------------


class CalendarAddRequest(BaseModel):
    date: str = Field(..., min_length=8, max_length=10)  # YYYY-MM-DD
    title: str = Field(..., min_length=1, max_length=500)
    time: str | None = Field(None, max_length=5)  # HH:MM
    notes: str = Field("", max_length=4000)


@router.get("/calendar")
async def calendar_list(month: str | None = None) -> list[Any]:
    from himmy.api.studio_calendar import get_calendar_store

    return get_calendar_store().list(month=month)


@router.post("/calendar")
async def calendar_add(body: CalendarAddRequest) -> Any:
    from himmy.api.studio_calendar import CalendarEvent, get_calendar_store

    ev = CalendarEvent(
        date=body.date, title=body.title, time=body.time or None, notes=body.notes
    )
    return get_calendar_store().add(ev)


@router.delete("/calendar/{event_id}")
async def calendar_delete(event_id: str) -> dict[str, bool]:
    from himmy.api.studio_calendar import get_calendar_store

    return {"ok": get_calendar_store().delete(event_id)}


# ---- Memory (long-term recall browser) ----------------------------------


class MemoryAddRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20_000)
    subject_id: str = Field("default", max_length=200)
    kind: str = Field("semantic", max_length=40)


class MemoryRecallRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2_000)
    subject_id: str = Field("default", max_length=200)
    top_k: int = Field(5, ge=1, le=50)


@router.get("/memory/subjects", response_model=list[str])
async def memory_subjects() -> list[str]:
    from himmy.api import studio_memory

    return studio_memory.list_subjects()


@router.get("/memory")
async def memory_list(subject: str = "default") -> list[Any]:
    from himmy.api import studio_memory

    return studio_memory.list_memories(subject)


@router.post("/memory")
async def memory_add(body: MemoryAddRequest) -> Any:
    from himmy.api import studio_memory

    return studio_memory.add_memory(
        body.text, subject_id=body.subject_id, kind=body.kind
    )


@router.delete("/memory/{memory_id}")
async def memory_forget(memory_id: str) -> dict[str, bool]:
    from himmy.api import studio_memory

    return {"ok": studio_memory.forget(memory_id)}


@router.post("/memory/recall")
async def memory_recall(body: MemoryRecallRequest) -> list[Any]:
    from himmy.api import studio_memory

    return await studio_memory.recall(
        body.query, subject_id=body.subject_id, top_k=body.top_k
    )


# ---- Knowledge (RAG: ingest + retrieval tester) -------------------------


class KbCreateRequest(BaseModel):
    name: str


class KbIngestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1_000_000)
    title: str | None = Field(None, max_length=300)


class KbSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(5, ge=1, le=50)


@router.get("/knowledge")
async def kb_list() -> list[Any]:
    from himmy.api import studio_knowledge

    return studio_knowledge.list_kbs()


@router.post("/knowledge")
async def kb_create(body: KbCreateRequest) -> Any:
    from himmy.api import studio_knowledge

    return await studio_knowledge.create_kb(body.name)


@router.post("/knowledge/{kb_id}/ingest")
async def kb_ingest(kb_id: str, body: KbIngestRequest) -> Any:
    from himmy.api import studio_knowledge

    return await studio_knowledge.ingest_text(kb_id, body.text, title=body.title)


@router.post("/knowledge/{kb_id}/search")
async def kb_search(kb_id: str, body: KbSearchRequest) -> list[Any]:
    from himmy.api import studio_knowledge

    return await studio_knowledge.search(kb_id, body.query, top_k=body.top_k)


@router.delete("/knowledge/{kb_id}")
async def kb_delete(kb_id: str) -> dict[str, bool]:
    from himmy.api import studio_knowledge

    return {"ok": await studio_knowledge.delete_kb(kb_id)}


# ---- Evaluation (suites → run → scorecard) ------------------------------


class EvalRunRequest(BaseModel):
    suite_path: str
    agent_path: str
    provider: str | None = None
    model: str | None = None


@router.get("/evals")
async def eval_suites() -> list[Any]:
    from himmy.api import studio_eval

    return studio_eval.discover_suites()


@router.post("/evals/run")
async def eval_run(body: EvalRunRequest) -> Any:
    import asyncio

    from himmy.api import studio_eval

    try:
        return await asyncio.wait_for(
            studio_eval.run_eval(
                body.suite_path,
                body.agent_path,
                provider=body.provider,
                model=body.model,
            ),
            timeout=900,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="eval timed out") from exc


# ---- Workflows (declarative multi-step) ---------------------------------


class WorkflowRunRequest(BaseModel):
    workflow_path: str
    agent_path: str
    provider: str | None = None
    model: str | None = None
    initial_state: dict[str, Any] = {}


@router.get("/workflows")
async def workflows() -> list[Any]:
    from himmy.api import studio_workflows

    return studio_workflows.discover_workflows()


@router.post("/workflows/run")
async def workflow_run(body: WorkflowRunRequest) -> Any:
    import asyncio

    from himmy.api import studio_workflows

    try:
        return await asyncio.wait_for(
            studio_workflows.run_workflow(
                body.workflow_path,
                body.agent_path,
                provider=body.provider,
                model=body.model,
                initial_state=body.initial_state,
            ),
            timeout=900,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="workflow timed out") from exc


# ---- Lineage (provenance of a run's answer) -----------------------------


@router.get("/runs/{run_id}/lineage")
async def run_lineage(run_id: str) -> Any:
    from himmy.api import studio_lineage

    view = studio_lineage.run_lineage(run_id)
    if view is None:
        raise HTTPException(status_code=404, detail="run not found")
    return view


# ---- Agent authoring (the no-code builder) ------------------------------


@router.get("/tools", response_model=list[studio_agents.PackInfo])
async def tool_packs() -> list[studio_agents.PackInfo]:
    """The built-in tool packs an agent can switch on."""
    return studio_agents.list_tool_packs()


@router.get("/skills", response_model=list[studio_agents.SkillInfo])
async def skills() -> list[studio_agents.SkillInfo]:
    """Available skills (built-in + project-local)."""
    return studio_agents.list_skill_infos()


@router.get("/agent", response_model=studio_agents.AgentDetail)
async def get_agent(path: str) -> studio_agents.AgentDetail:
    """Load one agent's full editable spec (by project-relative path)."""
    try:
        return studio_agents.load_agent_detail(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agents/validate", response_model=studio_agents.ValidationResult)
async def validate_agent(spec: dict) -> studio_agents.ValidationResult:
    """Validate a proposed spec without saving (live form feedback)."""
    errors = studio_agents.validate_spec(spec)
    return studio_agents.ValidationResult(ok=not errors, errors=errors)


@router.put("/agents", response_model=studio_service.AgentSummary)
async def save_agent(
    body: studio_agents.SaveAgentRequest,
) -> studio_service.AgentSummary:
    """Create or update an agent.yaml (validated; merges onto any existing file).

    Returns 409 when creating would overwrite an existing file without ``overwrite``.
    """
    try:
        return studio_agents.save_agent(body.path, body.spec, overwrite=body.overwrite)
    except studio_agents.AgentExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except studio_agents.SpecInvalid as exc:
        raise HTTPException(status_code=422, detail={"errors": exc.errors}) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


__all__ = ["router"]
