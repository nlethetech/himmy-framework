"""API kernel: Studio team builder + streaming workflow runner.

Two guarded route families share this file (both built via
:func:`himmy.api.routers.studio_common.build_studio_router`, so they carry the
same ``studio:use`` guard and live under the ``/api/studio`` prefix the
DNS-rebinding guard matches on):

* ``/api/studio/teams`` — author a multi-agent ``*.team.yaml`` from the GUI:
  ``POST`` saves a validated spec (path-safe name, written next to existing
  team files), ``POST /validate`` runs the same checks without writing, and
  ``GET /detail`` loads an existing team's parsed spec for editing.
* ``/api/studio/workflows/run-stream`` — run a declarative workflow with a
  chosen agent, streaming node-level progress (started / completed / failed
  per step, then a terminal ``done`` with the final state) over SSE, bounded
  by a hard timeout.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from himmy.api import studio_service
from himmy.api.routers.studio_common import build_studio_router, studio_permission
from himmy.core.events import EventType

router = APIRouter()
_teams = build_studio_router("teams", tag="studio-teams")
_workflows = build_studio_router("workflows", tag="studio-workflows")

#: Saving a team spec writes a ``.team.yaml`` to disk; running a workflow executes
#: agents. Both are privileged mutations gated by the respective surface's ``:write``
#: permission (admin-only by default), additively on top of each router's ``:read``
#: baseline so a read-only role can browse/validate but never persist or execute.
_teams_write = Depends(studio_permission("studio.teams", "write"))
_workflows_write = Depends(studio_permission("studio.workflows", "write"))


# ---- Team builder: request/response shapes -------------------------------

# Path-safe team names only: a leading alphanumeric, then a bounded run of
# alphanumerics / dot / underscore / hyphen. No slashes, no leading dot — the
# name becomes the ``<name>.team.yaml`` filename.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

# Providers a member may pin (mirrors the team_spec docs). A custom value is
# not rejected at save time (dispatch falls back to the default backend), but
# the GUI offers these.
KNOWN_PROVIDERS = ("claude-cli", "ollama", "pydantic-ai", "openrouter", "stub")


class TeamMemberForm(BaseModel):
    """One member of the team as the GUI form submits it."""

    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field("", max_length=4000)
    instructions: list[str] = Field(default=[], max_length=50)
    role: str | None = Field(None, max_length=120)
    provider: str | None = Field(None, max_length=40)
    model: str = Field("default", max_length=120)
    tools: list[str] = Field(default=[], max_length=64)
    tool_packs: list[str] = Field(default=[], max_length=32)
    handoffs: list[str] = Field(default=[], max_length=32)
    delegates: list[str] = Field(default=[], max_length=32)


class TeamForm(BaseModel):
    """The proposed team spec, shared by save + validate."""

    name: str = Field(..., min_length=1, max_length=64)
    entry: str = Field(..., min_length=1, max_length=80)
    members: list[TeamMemberForm] = Field(..., max_length=24)


class SaveTeamRequest(TeamForm):
    """``POST /api/studio/teams``: write ``<name>.team.yaml``.

    ``path`` (optional) targets an existing team file being edited, so a team
    discovered outside the default directory round-trips to the same file.
    ``overwrite`` must be true to write over an existing file (else 409).
    """

    path: str | None = Field(None, max_length=512)
    overwrite: bool = False


class TeamValidationResult(BaseModel):
    ok: bool
    errors: list[str] = []


class TeamDetail(BaseModel):
    """An existing team's parsed spec, shaped for the builder form."""

    path: str
    name: str
    spec: dict[str, Any]
    has_advanced: bool = False  # carries fields the form doesn't expose


def _validate_team(form: TeamForm) -> list[str]:
    """Human-readable validation errors for a proposed team (empty = valid)."""
    errors: list[str] = []

    if not _NAME_RE.match(form.name):
        errors.append(
            "Team name must start with a letter or digit and use only "
            "letters, digits, dots, underscores or hyphens (max 64 chars)."
        )

    if not form.members:
        errors.append("At least one member is required.")

    names = [m.name for m in form.members]
    seen: set[str] = set()
    for n in names:
        if n in seen:
            errors.append(f"Duplicate member name: {n!r}.")
        seen.add(n)

    if form.entry and form.entry not in seen:
        errors.append(f"Entry member {form.entry!r} is not in the member list.")

    from himmy.toolkit import BUILTIN_PACKS

    known_packs = set(BUILTIN_PACKS)
    for m in form.members:
        for kind in ("handoffs", "delegates"):
            for target in getattr(m, kind):
                if target == m.name:
                    errors.append(f"Member {m.name!r} cannot {kind[:-1]} to itself.")
                elif target not in seen:
                    errors.append(
                        f"Member {m.name!r} has an unknown {kind[:-1]} "
                        f"target: {target!r}."
                    )
        for pack in m.tool_packs:
            if pack not in known_packs:
                errors.append(f"Member {m.name!r} uses an unknown tool pack: {pack!r}.")

    # The real loader's validation (same model load_team_spec uses).
    if not errors:
        from himmy.config.team_spec import TeamSpec

        try:
            TeamSpec.model_validate(_spec_payload(form))
        except Exception as exc:  # noqa: BLE001 - surface pydantic messages
            errors.append(f"Spec error: {exc}")

    return errors


def _member_payload(m: TeamMemberForm) -> dict[str, Any]:
    """One member as a tidy YAML mapping (defaults/empties dropped)."""
    out: dict[str, Any] = {"name": m.name, "description": m.description}
    if m.instructions:
        out["instructions"] = list(m.instructions)
    if m.role:
        out["role"] = m.role
    if m.provider:
        out["provider"] = m.provider
    if m.model and m.model != "default":
        out["model"] = m.model
    if m.tools:
        out["tools"] = list(m.tools)
    if m.tool_packs:
        out["tool_packs"] = list(m.tool_packs)
    if m.handoffs:
        out["handoffs"] = list(m.handoffs)
    if m.delegates:
        out["delegates"] = list(m.delegates)
    return out


def _spec_payload(form: TeamForm) -> dict[str, Any]:
    """The team spec mapping exactly as it will be serialized."""
    return {
        "entry": form.entry,
        "members": [_member_payload(m) for m in form.members],
    }


def _writable_team_path(body: SaveTeamRequest, root: Path) -> Path:
    """Resolve where this team gets written, safely under the project root.

    With ``body.path``: the exact (existing-team) file, traversal-guarded.
    Without: ``<name>.team.yaml`` next to the first discovered team file, or
    at the project root when no team exists yet.
    """
    if body.path is not None:
        rel = body.path
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise ValueError(f"invalid team path: {rel!r}")
        if not rel.endswith((".yaml", ".yml")):
            raise ValueError("team path must end in .yaml")
        target = (root / rel).resolve()
        if root != target.parent and root not in target.parents:
            raise ValueError(f"path escapes project root: {rel!r}")
        return target

    teams = studio_service.list_teams(root)
    directory = (root / teams[0].path).parent if teams else root
    return (directory / f"{body.name}.team.yaml").resolve()


@_teams.post("/validate", response_model=TeamValidationResult)
async def validate_team(form: TeamForm) -> TeamValidationResult:
    """Validate a proposed team without saving (live form feedback)."""
    errors = _validate_team(form)
    return TeamValidationResult(ok=not errors, errors=errors)


@_teams.post(
    "", response_model=studio_service.TeamSummary, dependencies=[_teams_write]
)
async def save_team(body: SaveTeamRequest) -> studio_service.TeamSummary:
    """Create or update a ``<name>.team.yaml`` (validated via the team loader).

    Returns 409 when the target exists and ``overwrite`` is false, 422 with the
    validation errors when the spec is invalid. Keys the form doesn't expose
    (e.g. ``mcp_servers``) are preserved when overwriting an existing file.
    """
    import yaml

    errors = _validate_team(body)
    if errors:
        raise HTTPException(status_code=422, detail="; ".join(errors))

    root = studio_service.project_root().resolve()
    try:
        target = _writable_team_path(body, root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    existing: dict[str, Any] = {}
    if target.is_file():
        if not body.overwrite:
            raise HTTPException(
                status_code=409,
                detail=f"{target.relative_to(root)} already exists",
            )
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing = loaded

    merged = dict(existing)
    merged.update(_spec_payload(body))

    # Round-trip the artifact through the real loader path before any write hits
    # disk (yaml.safe_load + TeamSpec is what load_team_spec does). TeamSpec now
    # forbids extras, so validate only the spec-defined keys here: opaque advanced/
    # forward-compat keys carried over from an existing file (e.g. ``x_custom``) are
    # preserved on disk but not fed to the strict model.
    from himmy.config.team_spec import TeamSpec

    text = yaml.safe_dump(merged, sort_keys=False, allow_unicode=True)
    spec_keys = set(TeamSpec.model_fields)
    to_validate = {k: v for k, v in yaml.safe_load(text).items() if k in spec_keys}
    try:
        TeamSpec.model_validate(to_validate)
    except Exception as exc:  # noqa: BLE001 - surface as a validation failure
        raise HTTPException(status_code=422, detail=f"Spec error: {exc}") from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")

    rel = str(target.relative_to(root))
    for summary in studio_service.list_teams(root):
        if summary.path == rel:
            return summary
    # Written somewhere discovery doesn't scan (edit of a nested file): still
    # return an accurate summary built from the spec we just validated.
    return studio_service.TeamSummary(
        name=body.name,
        path=rel,
        entry=body.entry,
        members=[
            studio_service.TeamMemberInfo(
                name=m.name,
                role=m.role,
                provider=m.provider,
                model=m.model,
                delegates=list(m.delegates),
                handoffs=list(m.handoffs),
            )
            for m in body.members
        ],
    )


@_teams.get("/detail", response_model=TeamDetail)
async def team_detail(path: str) -> TeamDetail:
    """Load an existing team's parsed spec (by project-relative path) for editing."""
    if len(path) > 512:
        raise HTTPException(status_code=400, detail="path too long")
    try:
        spec = studio_service.load_team(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stem = Path(path).name
    for suffix in (".yaml", ".yml"):
        stem = stem.removesuffix(suffix)
    name = stem.removesuffix(".team").replace("-team", "") or stem

    members = [
        {
            "name": m.name,
            "description": m.description,
            "instructions": list(m.instructions),
            "role": m.role,
            "provider": m.provider,
            "model": m.model,
            "tools": list(m.tools),
            "tool_packs": list(m.tool_packs),
            "handoffs": list(m.handoffs),
            "delegates": list(m.delegates),
        }
        for m in spec.members
    ]
    has_advanced = bool(getattr(spec, "mcp_servers", None)) or any(
        m.tools_module for m in spec.members
    )
    return TeamDetail(
        path=path,
        name=name,
        spec={"entry": spec.entry, "members": members},
        has_advanced=has_advanced,
    )


# ---- Workflow runner: stream node-level progress over SSE -----------------


class WorkflowStreamRequest(BaseModel):
    """``POST /api/studio/workflows/run-stream``: run a workflow with an agent."""

    workflow_path: str = Field(..., min_length=1, max_length=512)
    agent_path: str = Field(..., min_length=1, max_length=512)
    provider: str | None = Field(None, max_length=64)
    model: str | None = Field(None, max_length=128)
    initial_state: dict[str, Any] = Field(default={})
    timeout_seconds: float = Field(600.0, ge=5.0, le=900.0)


def _sse(event: dict[str, Any]) -> str:
    """Encode one event dict as a Server-Sent-Events ``data:`` frame."""
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


def _cap(text: str | None, n: int = 20_000) -> str | None:
    if text is None or len(text) <= n:
        return text
    return text[: n - 1] + "…"


def _jsonable_state(state: dict[str, Any]) -> dict[str, Any]:
    """The final state as JSON-safe data, bounded so a frame can't explode."""
    try:
        text = json.dumps(state, default=str, ensure_ascii=False)
    except (TypeError, ValueError):  # pragma: no cover - default=str covers most
        return {"_unserializable": True}
    if len(text) > 100_000:
        return {"_truncated": True}
    return dict(json.loads(text))


async def _workflow_stream(
    wf: Any, spec: Any, body: WorkflowStreamRequest, tool_authorizer: Any = None
) -> AsyncIterator[str]:
    """Bind the request principal's gate ambiently, then drive the workflow stream.

    centralize-tool-gate: ``tool_authorizer`` is the request principal's tool-capability
    gate, bound AMBIENTLY for the whole drive so every workflow step's tool dispatch is
    enforced even though this surface builds the runtime without threading the authorizer
    explicitly. ``None`` offline (no authenticator) → inert, byte-unchanged. The actual
    drive lives in :func:`_workflow_stream_body`; this wrapper just scopes the ambient
    binding around the entire async iteration.
    """
    from himmy.services.tools.ambient import use_tool_authorizer

    with use_tool_authorizer(tool_authorizer):
        async for frame in _workflow_stream_body(wf, spec, body):
            yield frame


async def _workflow_stream_body(
    wf: Any, spec: Any, body: WorkflowStreamRequest
) -> AsyncIterator[str]:
    """Drive one workflow run, yielding SSE frames per node transition.

    Frames: ``workflow_start`` → per step ``node``(running) … ``node``
    (completed|failed) → ``done`` (status + final state + step results), or a
    terminal ``error``. The run is bounded by the orchestrator's hard timeout
    plus a belt-and-braces outer deadline.
    """
    from himmy.orchestrators.workflow import WorkflowOrchestrator
    from himmy.runtime import from_spec

    queue: asyncio.Queue[Any] = asyncio.Queue()

    async def _push(event: Any) -> None:
        await queue.put(event)

    # Build off the event loop (tool-pack registration + ingest are sync).
    try:
        runtime, _registry = await asyncio.to_thread(
            lambda: from_spec.build_runtime_for_spec(
                spec,
                provider=body.provider,
                model=body.model,
                durable_defaults=True,
            )
        )
    except Exception as exc:  # noqa: BLE001 - a missing provider is a clean frame
        yield _sse(
            {
                "type": "error",
                "message": (
                    f"could not build the agent runtime: {exc}. Check that the "
                    "selected provider is installed and configured (Doctor shows "
                    "what's available)."
                ),
            }
        )
        return

    orch = WorkflowOrchestrator(runtime, on_event=_push)
    step_names: list[str] = [s.name for s in wf.steps]
    yield _sse(
        {
            "type": "workflow_start",
            "workflow": wf.name,
            "steps": step_names,
            "timeout_seconds": body.timeout_seconds,
        }
    )

    run_task = asyncio.create_task(
        orch.run(
            wf,
            spec.to_persona(),
            initial_state=dict(body.initial_state),
            timeout_seconds=body.timeout_seconds,
        )
    )
    started = time.monotonic()
    index = 0

    try:
        while not (run_task.done() and queue.empty()):
            # Belt-and-braces outer deadline on top of the orchestrator timeout.
            if time.monotonic() - started > body.timeout_seconds + 30:
                run_task.cancel()
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.25)
            except TimeoutError:
                continue
            etype = getattr(event, "event_type", None)
            payload = getattr(event, "payload", None) or {}
            if etype == EventType.WORKFLOW_STARTED:
                index = int(payload.get("start_index", 0) or 0)
                if 0 <= index < len(step_names):
                    yield _sse(
                        {"type": "node", "name": step_names[index], "state": "running"}
                    )
            elif etype == EventType.WORKFLOW_STEP_COMPLETED:
                status = str(payload.get("status", "completed"))
                yield _sse(
                    {
                        "type": "node",
                        "name": str(payload.get("step_name", "?")),
                        "state": "failed" if status != "completed" else "completed",
                        "error": _cap(getattr(event, "error", None), 2000),
                    }
                )
                index += 1
                if status == "completed" and index < len(step_names):
                    yield _sse(
                        {"type": "node", "name": step_names[index], "state": "running"}
                    )
        result = await run_task
        yield _sse(
            {
                "type": "done",
                "workflow_name": result.workflow_name,
                "status": result.status,
                "final_state": _jsonable_state(result.final_state),
                "step_results": [
                    {
                        "step_name": r.step_name,
                        "status": r.status,
                        "output_text": _cap(r.output_text),
                        "error": _cap(r.error, 2000),
                        "attempts": r.attempts,
                    }
                    for r in result.step_results
                ],
            }
        )
    except asyncio.CancelledError:
        if not run_task.done():
            # OUR generator was cancelled (client went away) — stop the run too.
            run_task.cancel()
            raise
        yield _sse(
            {
                "type": "error",
                "message": (
                    "workflow run was cancelled "
                    f"(hard timeout {body.timeout_seconds:.0f}s)"
                ),
            }
        )
    except Exception as exc:  # noqa: BLE001 - terminal error frame, clean stream end
        yield _sse({"type": "error", "message": str(exc)})


@_workflows.post("/run-stream", dependencies=[_workflows_write])
async def run_workflow_stream(
    body: WorkflowStreamRequest, request: Request
) -> StreamingResponse:
    """Run a workflow with the chosen agent, streaming node-level events (SSE).

    Node frames carry ``state``: ``running`` → ``completed`` | ``failed``; the
    terminal ``done`` frame carries the final status, state, and step results.
    Path/spec problems are a clean 4xx before the stream starts.
    """
    try:
        from himmy.config.workflow_spec import load_workflow_spec

        wf = load_workflow_spec(
            str(studio_service.resolve_spec_path(body.workflow_path))
        )
        spec = studio_service.load_studio_spec(
            body.agent_path, provider=body.provider, model=body.model
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not wf.steps:
        raise HTTPException(status_code=400, detail="workflow has no steps")

    # centralize-tool-gate: thread the request principal's tool-capability gate so workflow
    # step tools are enforced (inert offline). Bound ambiently inside ``_workflow_stream``.
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    tool_authorizer = ToolCapabilityAuthorizer.from_request(request)
    return StreamingResponse(
        _workflow_stream(wf, spec, body, tool_authorizer),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


router.include_router(_teams)
router.include_router(_workflows)

__all__ = ["router"]
