"""API kernel: the Himmy Studio router (the local web GUI's backend).

Studio is the no-code front door to himmy: chat with an agent, build/edit an
``agent.yaml``, and browse past runs/traces — all over the same FastAPI BFF. This
router is intentionally GUI-shaped (not the tenant-scoped ``/v1`` surface): it is
meant to be served on loopback by ``himmy studio`` for a single local user.

When an authenticator IS configured (API keys / OIDC), every Studio route requires a
PER-SURFACE permission (the coarse ``studio:use`` is gone): the main router carries a
``studio.console:read`` baseline ("may open Studio"), and each read route on this main
router ADDITIONALLY declares its own per-surface ``studio.<surface>:read`` (a higher-value
read — connections/google/approvals/models — therefore needs both the console baseline AND
that surface's read grant, NOT just the baseline), while each mutating route declares its
``studio.<surface>:write`` / ``studio.mcp:manage`` grant — so a least-privilege role that
holds only the console baseline cannot read a surface it was not granted, and only ``admin``
mutates. (red-team r6: before, these reads collapsed to the coarse console baseline, so any
console-read role reached connections/google/approvals/models — fixed by the per-route read
guards above + withholding the operator connections/google surfaces from default browse
roles in :data:`himmy.api.auth.rbac._STUDIO_GLOBAL_STORE_RESOURCES`.)
Studio holds credentials (SMTP/Telegram), approvals, and run triggers, so a
tenant-scoped key without those grants must not reach the write surface once the app
is proxied beyond loopback. The zero-config (no-auth) loopback default is unchanged.
A tenant-bound principal additionally sees only its own workspace's runs (the global
run readers are tenant-filtered via :func:`himmy.api.auth.context.studio_tenant_filter`).

Endpoints:
  * ``GET  /api/studio/doctor``  — environment diagnostics as JSON.
  * ``GET  /api/studio/agents``  — discover agent.yaml files in the project.
  * ``POST /api/studio/run``     — run an agent for one turn (SSE token stream).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from himmy.api import (
    studio_agents,
    studio_approvals,
    studio_connections,
    studio_feedback,
    studio_service,
)
from himmy.api.auth import (
    authorize_object,
    authorize_studio_object,
    enforce_subject_write,
    get_principal,
    narrow_subject,
    scoped_read,
    studio_subject_filter,
    studio_tenant_filter,
    subject_write,
)
from himmy.api.routers.studio_common import studio_permission
from himmy.api.studio_approvals import ApprovalDetail, ApprovalSummary
from himmy.api.studio_connections import (
    ConnectionStatus,
    ConnectionTestResult,
    ReadOnlyBackendError,
    SendResult,
)
from himmy.api.studio_feedback import RunFeedback
from himmy.api.studio_runs import (
    RunAnalytics,
    StudioRun,
    StudioRunListResponse,
    get_run_store,
)

# Granular Studio RBAC resources (mirroring the ``/v1`` per-surface pattern). The main
# router carries a low-privilege ``studio.console:read`` BASELINE — "may open Studio at
# all" — so a read-only role can browse the console; each MUTATING route below declares
# its stronger ``:write`` / ``:manage`` permission additively (FastAPI runs router- and
# route-level dependencies together, so a mutator requires the baseline read AND the
# write/manage grant, both of which ``admin`` holds via ``*:*``). The granular surfaces:
_RES_CONSOLE = "studio.console"  # baseline: open Studio + read diagnostics/health
_RES_RUNS = "studio.runs"  # run history, analytics, feedback, lineage
_RES_CONNECTIONS = "studio.connections"  # email/telegram/web connection secrets
_RES_GOOGLE = "studio.google"  # Gmail/Calendar OAuth credentials + send
_RES_APPROVALS = "studio.approvals"  # HITL checkpoint approve/reject
_RES_MODELS = "studio.models"  # provider API keys, Ollama pulls, model compare
# red-team reattack-r6: the BENIGN model catalog (the tenant-facing picker) is split off
# ``studio.models`` so it stays tenant-grantable while ``studio.models`` (the operator
# provider-credential posture surface) is withheld to admin-only.
_RES_MODEL_CATALOG = "studio.modelcatalog"  # read-only model picker (no key posture)
_RES_AGENTS = "studio.agents"  # agent.yaml edit/validate + project-root spec discovery
_RES_TEAMS = "studio.teams"  # team.yaml project-root spec discovery
_RES_TASKS = "studio.tasks"  # task board CRUD
_RES_CHATS = "studio.chats"  # chat session CRUD
_RES_COOKBOOK = "studio.cookbook"  # saved recipes CRUD
_RES_NOTES = "studio.notes"  # notes CRUD
_RES_CALENDAR = "studio.calendar"  # local calendar CRUD
_RES_MEMORY = "studio.memory"  # subject memory CRUD/recall
_RES_KNOWLEDGE = "studio.knowledge"  # KB create/ingest/search/delete
_RES_EVALS = "studio.evals"  # eval suite runs
_RES_WORKFLOWS = "studio.workflows"  # workflow runs


router = APIRouter(
    prefix="/api/studio",
    tags=["studio"],
    dependencies=[Depends(studio_permission(_RES_CONSOLE, "read"))],
)


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Fast readiness probe: status + version, plus operator deployment posture for admins.

    red-team reattack-r6: the ``secrets_writable`` flag and the ``providers`` presence map
    (which LLM binaries — claude/ollama — are installed) are OPERATOR deployment posture,
    the same reconnaissance class withheld from tenant browse roles for ``/doctor`` and the
    provider credential-status surface. The bare readiness fields (``status``/``version``)
    stay readable by every browse role (and load balancers) under the ``studio.console:read``
    baseline; the posture fields are included ONLY when the caller also passes
    ``studio.console:write`` (admin-only), so a tenant browse role gets a clean probe without
    the operator topology. OFFLINE is unaffected — :func:`_caller_holds_console_write`
    returns True when no authenticator is configured, so the single-box console sees the full
    payload byte-unchanged.
    """
    import shutil

    from himmy import __version__

    payload: dict[str, Any] = {"status": "ok", "version": __version__}
    if _caller_holds_console_write(request):
        from himmy.config.secrets import get_writable_provider

        payload["secrets_writable"] = get_writable_provider() is not None
        payload["providers"] = {
            "claude_cli": shutil.which("claude") is not None,
            "ollama": shutil.which("ollama") is not None,
        }
    return payload


def _caller_holds_console_write(request: Request) -> bool:
    """Whether the request may see operator deployment posture (``studio.console:write``).

    Mirrors the offline bypass + escape-hatch of :func:`studio_permission`/
    :func:`require_permission` so the posture-field gate can never diverge from the route
    guards: returns True when no authenticator is configured (offline-first — the zero-config
    console is byte-unchanged), when the ``HIMMY_STUDIO_AUTH`` kill-switch is off, and
    otherwise iff the principal's roles grant ``studio.console:write`` (admin holds it via
    ``*:*``). Best-effort: any error fails CLOSED (posture withheld).
    """
    from himmy.api.routers.studio_common import _studio_auth_off

    if _studio_auth_off():
        return True
    if getattr(request.app.state, "authenticator", None) is None:
        return True  # offline-first → no RBAC → full posture (byte-unchanged)
    try:
        from himmy.api.auth.rbac import DEFAULT_POLICY

        policy = getattr(request.app.state, "access_policy", None) or DEFAULT_POLICY
        return policy.authorize(get_principal(request), "studio.console", "write")
    except Exception:  # noqa: BLE001 - posture is non-essential; fail closed
        return False


@router.get(
    "/doctor",
    dependencies=[Depends(studio_permission(_RES_CONSOLE, "write"))],
)
async def doctor() -> dict[str, Any]:
    """Environment diagnostics: extras, providers, keys, and the next step.

    The JSON twin of ``himmy doctor`` — same
    :func:`himmy.runtime.diagnostics.collect_doctor_report` snapshot the CLI prints.

    red-team r6: this leaks DEPLOYMENT TOPOLOGY (installed extras, which provider keys are
    configured, the storage backend + a password-masked DSN host) — an OPERATOR concern,
    not a tenant one. Inheriting only the ``studio.console:read`` baseline made it reachable
    by every tenant-facing browse role; it now additionally requires ``studio.console:write``
    (admin-only, like the ``benchmarks/probe`` operator action), so a tenant browse role is
    403'd. OFFLINE is unaffected (``studio_permission`` no-ops without an authenticator).
    """
    from himmy.runtime.diagnostics import collect_doctor_report

    return collect_doctor_report().to_dict()


@router.get(
    "/benchmarks",
    dependencies=[Depends(studio_permission(_RES_CONSOLE, "write"))],
)
async def benchmarks() -> dict[str, Any]:
    """Cached per-model reliability scorecards (from `himmy bench` / the probe).

    red-team reattack-r7: this returns the raw cached scorecards
    (``studio_bench.list_cached()``) — finer-grained operator-topology reliability detail
    (``model_id``, ``suite``, run timestamp, accuracy CIs, tool-call accuracy, p50/p95
    latency, error rate, trial counts) about the deployment's LOCAL models. Its
    operator-topology siblings on this router (``GET /doctor`` and ``POST
    /benchmarks/probe``) were raised to ``studio.console:write`` (admin-only) in r6 to keep
    deployment reconnaissance away from tenant browse roles, but this read was left at the
    ``studio.console:read`` baseline. Raised to ``studio.console:write`` for consistency so
    a tenant browse role is 403'd here too. (The BENIGN per-model accuracy/latency summary
    a tenant model picker needs stays tenant-readable via ``GET /api/studio/models`` ->
    ``build_model_catalog``, gated by the separate ``studio.modelcatalog:read``.) OFFLINE is
    unaffected (``studio_permission`` no-ops without an authenticator).
    """
    from himmy.api import studio_bench

    return {"entries": studio_bench.list_cached()}


@router.post(
    "/benchmarks/probe",
    dependencies=[Depends(studio_permission(_RES_CONSOLE, "write"))],
)
async def benchmarks_probe() -> dict[str, Any]:
    """Run a quick tool-focused reliability check against detected local models.

    Slow (real model calls) but bounded — a small suite against ≤3 local models, one
    trial each. Caches the result so Doctor shows it without re-running.
    """
    from himmy.api import studio_bench

    return await studio_bench.run_probe()


@router.get(
    "/agents",
    response_model=list[studio_service.AgentSummary],
    dependencies=[Depends(studio_permission(_RES_AGENTS, "read"))],
)
async def list_agents() -> list[studio_service.AgentSummary]:
    """Discover single-agent specs under the project root.

    red-team reattack-r10: this list globs the single operator ``project_root()`` and
    returns each spec's project-relative server filesystem ``path`` (infrastructure recon),
    ``name``, ``provider``/``model`` and tool/skill presence — operator-local filesystem
    inventory with NO per-tenant axis to intersect on, the SAME class withheld for
    ``studio.eval``/``studio.evals``/``studio.workflows``/``studio.routines`` in r7/r8.
    Previously it carried only the coarse ``studio.console:read`` baseline (held by every
    browse role), so a tenant-facing viewer/operator/auditor could read the operator's
    agent inventory. It now requires the per-surface ``studio.agents:read``, which is
    withheld from the default browse roles (see :data:`himmy.api.auth.rbac._STUDIO_GLOBAL_STORE_RESOURCES`)
    and remains admin-only. OFFLINE is unaffected (``studio_permission`` no-ops without an
    authenticator).
    """
    return studio_service.list_agents()


@router.get(
    "/teams",
    response_model=list[studio_service.TeamSummary],
    dependencies=[Depends(studio_permission(_RES_TEAMS, "read"))],
)
async def list_teams() -> list[studio_service.TeamSummary]:
    """Discover multi-agent team specs (manager + workers) under the project root.

    red-team reattack-r10: like :func:`list_agents`, this globs the single operator
    ``project_root()`` and returns each team's project-relative filesystem ``path``,
    ``name``, ``entry`` and the member graph (providers/models/delegates/handoffs =
    orchestration topology) — operator-local inventory with NO tenant axis, the same class
    closed for the sibling discovery surfaces in r7/r8. It now requires the per-surface
    ``studio.teams:read`` (withheld from browse roles, admin-only) instead of collapsing to
    ``studio.console:read``. OFFLINE is unaffected (``studio_permission`` no-ops without an
    authenticator).
    """
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


def _canonical_storage(request: Request) -> Any | None:
    """The ONE canonical run store every surface reads (T2.2), or None.

    A Studio run mirrors into this store so it is browsable from ``/v1`` + the CLI.
    Resolved from the request-bound container's storage (the same backend the
    ``RunAppService`` writes), so a SQLite *or* Postgres deployment mirrors through the
    same async store on the main loop. Returns None when no container is attached (a
    bare test app) so the streamer degrades to studio.db-only, never erroring.
    """
    container = getattr(request.app.state, "container", None)
    return getattr(container, "storage", None) if container is not None else None


async def _authorize_run(request: Request, run_id: str) -> bool:
    """May this request's principal read run ``run_id`` (BOLA gate for by-id readers)?

    The by-id companion the sibling run readers (``analytics``/``feedback``/``lineage``)
    were missing: ``get_studio_run_unified`` already 404s a cross-tenant fetch, but the
    feedback and lineage readers hit the studio.db presentation cache directly with no
    tenant check, so a tenant-bound principal could pull another tenant's run
    feedback/lineage by id whenever a cache row existed. This resolves the run's
    **canonical** ``workspace_id`` (the authoritative tenant stamp — the presentation
    cache carries none) and applies :func:`authorize_studio_object`.

    Returns True — i.e. allow — for an unrestricted principal (offline / ``all_tenants``,
    where the tenant stamp is irrelevant) and when no canonical storage is attached (the
    bare/single-box test app). So the zero-config path is a NO-OP and byte-unchanged.

    For a TENANT-BOUND or ``subject_scoped`` principal, enforcement engages on BOTH axes: a
    run whose canonical record is absent (cache-only — erased/crypto-shredded or aged out
    while the process-wide studio.db presentation cache row lingers) is REFUSED (returns False
    → 404), because the cache carries no ``workspace_id``/``subject_id`` to attribute it and
    serving it would leak another tenant's/subject's feedback/lineage by id (an IDOR/BOLA).
    This mirrors the sibling ``get_studio_run_unified``, which also folds a cache-only row to
    not-found for a scoped reader. A canonically-known run is allowed only when its
    ``workspace_id`` passes :func:`authorize_studio_object` (tenant axis) AND its
    ``subject_id`` passes :func:`authorize_object` (subject/BOLA axis) — closing the
    alternate-surface BOLA where a subject-scoped caller could read/poison another subject's
    run feedback through Studio. Callers fold a False verdict into a **404** so existence is
    never leaked.
    """
    principal = get_principal(request)
    if principal.all_tenants:
        return True
    storage = _canonical_storage(request)
    if storage is None:
        return True
    rec = await storage.get_run(run_id)
    if rec is None:
        # Cache-only run (the canonical RunRecord is absent — erased/crypto-shredded or
        # aged out — while the process-wide studio.db presentation cache row lingers).
        # A scoped reader cannot be served it: the cache carries NO workspace_id/subject_id
        # to attribute it, so allowing it would leak another tenant's/subject's
        # feedback/lineage by id (an IDOR/BOLA). A purely tenant-bound but NOT subject-scoped
        # principal still refuses (the workspace stamp is missing); a subject-scoped one
        # refuses on the same grounds. The ``all_tenants`` short-circuit above keeps the
        # offline/admin path a no-op, so the zero-config surface is byte-unchanged.
        return False
    return authorize_studio_object(
        request, getattr(rec, "workspace_id", None)
    ) and authorize_object(request, getattr(rec, "subject_id", None))


@router.post("/run", dependencies=[Depends(studio_permission(_RES_RUNS, "write"))])
async def run(body: RunRequest, request: Request) -> StreamingResponse:
    """Run an agent for one user turn, streaming GUI events over SSE.

    Loads the selected ``agent.yaml`` (with project defaults + skills applied),
    wires it exactly like ``himmy run``, and streams ``start``/``token``/``tool``/
    ``message``/``done`` frames. A load/build failure becomes a single ``error``
    frame so the client always gets a clean end to the stream. The run is mirrored
    into the canonical run store (T2.2) so it is browsable from ``/v1`` + the CLI.
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

    canonical = _canonical_storage(request)
    # centralize-tool-gate: bind the request principal's tool-capability gate ambiently for
    # the whole stream so the Studio chat run's tools are enforced (this surface builds its
    # runtime without an explicit authorizer). ``None`` offline (no authenticator) → inert,
    # byte-unchanged loopback default.
    from himmy.services.tools.ambient import use_tool_authorizer
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    tool_authorizer = ToolCapabilityAuthorizer.from_request(request)

    async def _stream() -> AsyncIterator[str]:
        try:
            with use_tool_authorizer(tool_authorizer):
                async for event in studio_service.stream_agent_run(
                    spec,
                    body.prompt,
                    history=[t.model_dump() for t in body.history],
                    provider=body.provider,
                    model=body.model,
                    agent_path=body.agent_path,
                    canonical_storage=canonical,
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


@router.post(
    "/run-team", dependencies=[Depends(studio_permission(_RES_RUNS, "write"))]
)
async def run_team(body: RunTeamRequest, request: Request) -> StreamingResponse:
    """Run a multi-agent team, streaming the live routing/delegate/tool trail (SSE).

    Members run on their own providers (e.g. a Claude-CLI manager delegating to local
    Ollama workers). Frames: ``start`` → ``delegate``/``handoff``/``tool`` → ``message``
    → ``done`` (or a terminal ``error``). The team run is mirrored into the canonical
    run store (T2.2) so it is browsable from ``/v1`` + the CLI.
    """
    try:
        spec = studio_service.load_team(body.team_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    teams = {t.path: t for t in studio_service.list_teams()}
    team_name = teams[body.team_path].name if body.team_path in teams else "team"
    canonical = _canonical_storage(request)
    # centralize-tool-gate: bind the request principal's gate ambiently for the team
    # stream (inert offline). Members dispatch tools through runtimes this surface builds
    # without an explicit authorizer; the ambient binding makes the chokepoint enforce.
    from himmy.services.tools.ambient import use_tool_authorizer
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    tool_authorizer = ToolCapabilityAuthorizer.from_request(request)

    async def _stream() -> AsyncIterator[str]:
        try:
            with use_tool_authorizer(tool_authorizer):
                async for event in studio_service.stream_team_run(
                    spec,
                    body.prompt,
                    team_name=team_name,
                    team_path=body.team_path,
                    canonical_storage=canonical,
                ):
                    yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---- Deep Research (plan → search → synthesize) -------------------------

RESEARCH_INSTRUCTIONS = [
    "You are a thorough research analyst. Investigate the user's question and "
    "produce a well-structured, source-grounded report.",
    "Method: (1) briefly plan the angles worth investigating; (2) use web_search "
    "to find sources across several distinct angles, and web_fetch to read the most "
    "promising ones; (3) cross-check claims across sources before stating them.",
    "Write the final answer in Markdown with: a one-paragraph **Summary**, then "
    "**Key findings** as bullets (each with the fact and which source supports it), "
    "then **Sources** as a list of the URLs you actually used. Prefer primary "
    "sources. Be explicit about uncertainty and note where sources disagree.",
]


class ResearchRequest(BaseModel):
    """POST /api/studio/research: a question to investigate across the web."""

    query: str = Field(..., min_length=1, max_length=4000)
    provider: str | None = None
    model: str | None = None
    deep: bool = True  # deep → also allow fetching pages, not just search


@router.post(
    "/research", dependencies=[Depends(studio_permission(_RES_RUNS, "write"))]
)
async def research(body: ResearchRequest, request: Request) -> StreamingResponse:
    """Run a deep-research agent (web search + fetch) for one question over SSE.

    Builds an ephemeral research-configured :class:`AgentSpec` (web tools + a
    plan→search→synthesize methodology) and streams the same cognition/tool/message
    frames as a normal run — so the client can show the live search trail and the
    final report. Reuses all existing run machinery; nothing is persisted as an agent.
    The run is mirrored into the canonical run store (T2.2) like any other Studio run.
    """
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec(
        name="Deep Research",
        description="Investigates a question across the web and writes a report.",
        instructions=list(RESEARCH_INSTRUCTIONS),
        tool_packs=["web"],
        tool_router=False,
        temperature=0.2,
        provider=body.provider,
        model=body.model or "default",
    )
    canonical = _canonical_storage(request)
    # centralize-tool-gate: bind the request principal's gate ambiently (inert offline).
    from himmy.services.tools.ambient import use_tool_authorizer
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    tool_authorizer = ToolCapabilityAuthorizer.from_request(request)

    async def _stream() -> AsyncIterator[str]:
        try:
            with use_tool_authorizer(tool_authorizer):
                async for event in studio_service.stream_agent_run(
                    spec,
                    body.query,
                    provider=body.provider,
                    model=body.model,
                    agent_path="(deep-research)",
                    canonical_storage=canonical,
                ):
                    yield _sse(event)
        except Exception as exc:  # noqa: BLE001 - terminal error frame
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get(
    "/runs",
    response_model=StudioRunListResponse,
    dependencies=[
        Depends(studio_permission(_RES_RUNS, "read")),
        Depends(scoped_read),
    ],
)
async def list_runs(
    request: Request, limit: int = 50, offset: int = 0
) -> StudioRunListResponse:
    """List past runs (newest first), paginated, from the ONE canonical store (T2.2).

    Reads the canonical run store every surface shares, so runs created by ``/v1`` and
    the CLI ``--persist`` path appear here too (Studio is the single-user-local browse
    over the unified store). studio.db is a presentation cache that enriches matching
    rows + supplies human feedback.

    red-team r6: requires the per-surface ``studio.runs:read`` rather than collapsing to
    the coarse ``studio.console:read`` baseline, so a role holding only the console
    baseline (but not the runs grant) cannot browse run history. The rows are still
    tenant-filtered (``studio_tenant_filter``) so a tenant-bound principal sees only its
    own workspace's runs — this guard is the additional surface gate, NOT the tenant
    boundary. OFFLINE is unaffected (``studio_permission`` no-ops without an authenticator).
    """
    from himmy.api.studio_canonical import list_studio_runs_unified

    limit = max(1, min(limit, 200))
    offset = max(0, offset)
    items, total = await list_studio_runs_unified(
        _canonical_storage(request),
        limit=limit,
        offset=offset,
        accessible_workspaces=studio_tenant_filter(request),
        accessible_subject=studio_subject_filter(request),
    )
    return StudioRunListResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
        next_offset=(offset + len(items)) if offset + len(items) < total else None,
    )


@router.get(
    "/runs/analytics",
    response_model=RunAnalytics,
    dependencies=[
        Depends(studio_permission(_RES_RUNS, "read")),
        Depends(scoped_read),
    ],
)
async def runs_analytics(request: Request) -> RunAnalytics:
    """Aggregate cost/token/latency stats across runs (the analytics dashboard).

    The studio.db presentation cache these aggregates are computed over carries NO
    per-run tenant stamp (only the canonical store does), so a global aggregate would
    sum another tenant's cost/tokens into a tenant-bound principal's dashboard. For an
    unrestricted principal (offline / ``all_tenants``) the full aggregate is returned —
    byte-unchanged single-box behavior — while a tenant-bound principal gets EMPTY
    analytics rather than a cross-tenant total (the cache cannot be safely attributed to
    a workspace; per-tenant analytics is a follow-up that requires a tenant-stamped
    aggregate over the canonical store).
    """
    if studio_tenant_filter(request) is not None:
        return RunAnalytics()
    return get_run_store().analytics()


@router.get(
    "/runs/{run_id}",
    response_model=StudioRun,
    dependencies=[
        Depends(studio_permission(_RES_RUNS, "read")),
        Depends(scoped_read),
    ],
)
async def get_run(run_id: str, request: Request) -> StudioRun:
    """Fetch one run in full from the canonical store: transcript, tools, timeline.

    Resolves through the ONE canonical run store (T2.2) so a ``/v1``- or CLI-authored
    run is viewable in the Studio detail panel, with the studio.db cache supplying the
    richest fields when present.
    """
    from himmy.api.studio_canonical import get_studio_run_unified

    run = await get_studio_run_unified(
        _canonical_storage(request),
        run_id,
        accessible_workspaces=studio_tenant_filter(request),
        accessible_subject=studio_subject_filter(request),
    )
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return run


class RunFeedbackRequest(BaseModel):
    """A human's thumbs up/down on a run (with an optional note)."""

    verdict: Literal["up", "down"]
    note: str | None = Field(default=None, max_length=4000)


@router.post(
    "/runs/{run_id}/feedback",
    response_model=RunFeedback,
    dependencies=[Depends(studio_permission(_RES_RUNS, "write"))],
)
async def set_run_feedback(
    run_id: str, body: RunFeedbackRequest, request: Request
) -> RunFeedback:
    """Record human feedback on a run (the cheapest, least-noisy learning signal).

    Writes the verdict to the run row and appends an immutable, versioned
    ``run_feedback`` record to the entity spine. Re-rating is allowed and produces
    a new spine version (a full audit of every verdict change).
    """
    if not await _authorize_run(request, run_id):
        raise HTTPException(status_code=404, detail="run not found")
    try:
        fb = studio_feedback.record_feedback(
            run_id, verdict=body.verdict, note=body.note
        )
    except ValueError as exc:  # unknown verdict (belt-and-braces vs the Literal)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if fb is None:
        raise HTTPException(status_code=404, detail="run not found")
    # Attribute the verdict to the tools this run used (positive-signal learning). The run's
    # thread_id locates its tool events; best-effort, off-by-default for behaviour change.
    run = get_run_store().get(run_id)
    await studio_feedback.attribute_feedback_outcomes(
        verdict=body.verdict,
        run_id=run_id,
        thread_id=getattr(run, "thread_id", None),
        workspace_id=getattr(run, "workspace_id", None),
    )
    return fb


@router.get(
    "/runs/{run_id}/feedback",
    response_model=RunFeedback | None,
    dependencies=[
        Depends(studio_permission(_RES_RUNS, "read")),
        Depends(scoped_read),
    ],
)
async def get_run_feedback(run_id: str, request: Request) -> RunFeedback | None:
    """Current feedback on a run (latest verdict), or ``null`` if none yet."""
    if get_run_store().get(run_id) is None or not await _authorize_run(
        request, run_id
    ):
        raise HTTPException(status_code=404, detail="run not found")
    return studio_feedback.get_feedback(run_id)


# ---- Connections (connect Email/Telegram/Web so agents can act) ---------


class ConnectionSetRequest(BaseModel):
    fields: dict[str, Any]


@router.get(
    "/connections",
    response_model=list[ConnectionStatus],
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "read"))],
)
async def connections() -> list[ConnectionStatus]:
    """List connectable account types + their configured/writable status.

    red-team r6: requires the per-surface ``studio.connections:read`` (not just the coarse
    ``studio.console:read`` baseline), so a tenant browse role that does NOT hold the
    connections grant cannot read the operator's connection status + masked field hints.
    """
    return studio_connections.list_connections()


@router.get(
    "/connections/{ctype}",
    response_model=ConnectionStatus,
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "read"))],
)
async def connection(ctype: str) -> ConnectionStatus:
    status = studio_connections.get_connection(ctype)
    if status is None:
        raise HTTPException(status_code=404, detail="unknown connection type")
    return status


@router.put(
    "/connections/{ctype}",
    response_model=ConnectionStatus,
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "write"))],
)
async def set_connection(ctype: str, body: ConnectionSetRequest) -> ConnectionStatus:
    """Store a connection's fields (secrets → the writable backend; never echoed)."""
    try:
        return studio_connections.set_connection(ctype, body.fields)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc
    except ReadOnlyBackendError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete(
    "/connections/{ctype}",
    response_model=ConnectionStatus,
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "write"))],
)
async def delete_connection(ctype: str) -> ConnectionStatus:
    try:
        return studio_connections.delete_connection(ctype)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc
    except ReadOnlyBackendError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put(
    "/connections/{ctype}/enable",
    response_model=ConnectionStatus,
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "write"))],
)
async def enable_connection(ctype: str) -> ConnectionStatus:
    """Enable a connector-managed connection for its surface (outbound tool / inbound mount).

    Delegates to the connector-management service so the flag the runtime + CLI honor is the
    SAME store key. 404 for an unknown or non-enableable connection (email/telegram/web_search
    are always-on once configured); 409 on a read-only secrets backend.
    """
    try:
        return studio_connections.set_enabled(ctype, True)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="unknown or non-enableable connection"
        ) from exc
    except ReadOnlyBackendError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.put(
    "/connections/{ctype}/disable",
    response_model=ConnectionStatus,
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "write"))],
)
async def disable_connection(ctype: str) -> ConnectionStatus:
    """Disable a connector-managed connection (its tool/mount stops being wired)."""
    try:
        return studio_connections.set_enabled(ctype, False)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="unknown or non-enableable connection"
        ) from exc
    except ReadOnlyBackendError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post(
    "/connections/{ctype}/test",
    response_model=ConnectionTestResult,
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "write"))],
)
async def test_connection(ctype: str) -> ConnectionTestResult:
    """Live-validate a connection (SMTP login / Telegram getMe / search ping)."""
    try:
        return await studio_connections.test_connection(ctype)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc


class SendRequest(BaseModel):
    payload: dict[str, Any]


@router.post(
    "/connections/{ctype}/send",
    response_model=SendResult,
    dependencies=[Depends(studio_permission(_RES_CONNECTIONS, "write"))],
)
async def send_via_connection(ctype: str, body: SendRequest) -> SendResult:
    """Send a user-composed message directly (a Home quick action)."""
    try:
        return await studio_connections.send_via_connection(ctype, body.payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown connection type") from exc


# ---- Google (Gmail + Calendar via OAuth) --------------------------------


class GoogleClientRequest(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=300)
    client_secret: str = Field(..., min_length=1, max_length=300)


class GmailSendRequest(BaseModel):
    to: str = Field(..., min_length=1, max_length=320)
    subject: str = Field("", max_length=998)
    body: str = Field("", max_length=20000)


class CalendarCreateRequest(BaseModel):
    summary: str = Field(..., min_length=1, max_length=500)
    start: str = Field(..., max_length=40)
    end: str = Field(..., max_length=40)
    all_day: bool = False


def _google_redirect_uri(request: Any) -> str:
    """Derive the loopback OAuth callback URL from the incoming request."""
    base = str(request.base_url).rstrip("/")
    return f"{base}/api/studio/google/callback"


@router.get(
    "/google",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "read"))],
)
async def google_status() -> Any:
    """The operator's Google OAuth connection state (connected account, client status).

    red-team r6: requires the per-surface ``studio.google:read`` rather than collapsing to
    the ``studio.console:read`` baseline, so a tenant browse role without the Google grant
    cannot read the operator's connected-account state.
    """
    from himmy.api import studio_google

    return studio_google.status()


@router.put(
    "/google/client",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "write"))],
)
async def google_set_client(body: GoogleClientRequest) -> Any:
    from himmy.api import studio_google

    try:
        return studio_google.set_client(body.client_id, body.client_secret)
    except studio_google.GoogleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/google/auth-url",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "write"))],
)
async def google_auth_url(request: Request) -> dict[str, str]:
    """Mint a Google OAuth authorization URL (and a pending CSRF ``state``).

    red-team r6: this is a STATE-CHANGING step of the OAuth connect lifecycle (it registers
    a pending ``state``), so it is gated identically to its PUT/DELETE mutator siblings with
    ``studio.google:write`` (admin-only) — NOT the read baseline. A read-only browse role can
    no longer initiate the operator deployment's Google connect flow.
    """
    from himmy.api import studio_google

    try:
        url = studio_google.auth_url(_google_redirect_uri(request))
    except studio_google.GoogleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"url": url}


@router.get(
    "/google/callback",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "write"))],
)
async def google_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> HTMLResponse:
    """Complete the Google OAuth exchange and PERSIST the operator's tokens.

    red-team r6: exchanging the ``code`` writes ACCESS/REFRESH tokens into the deployment's
    writable secrets backend — a connection-state mutation. Gated with ``studio.google:write``
    (admin-only) like the rest of the OAuth lifecycle, so a read-only browse role cannot
    complete a token exchange via this GET.
    """
    from himmy.api import studio_google

    if error or not code:
        return HTMLResponse(_oauth_page("Connection cancelled", error or "no code"))
    try:
        st = await studio_google.exchange_code(
            code, _google_redirect_uri(request), state
        )
    except studio_google.GoogleError as exc:
        return HTMLResponse(_oauth_page("Could not connect", str(exc)))
    who = st.email or "your Google account"
    return HTMLResponse(
        _oauth_page("Connected ✓", f"{who} is now connected. You can close this tab.")
    )


def _oauth_page(title: str, detail: str) -> str:
    """A tiny self-closing page shown in the OAuth popup after the redirect."""
    import html as _html

    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Himmy · Google</title>"
        "<style>body{font-family:system-ui,sans-serif;background:#111;color:#eee;"
        "display:grid;place-items:center;height:100vh;margin:0;text-align:center}"
        "h1{font-size:20px;margin:0 0 8px}p{color:#9aa;max-width:30rem}</style>"
        # Security: post to the concrete Studio origin (this popup is served
        # same-origin, so window.location.origin IS the Studio origin) instead of
        # the wildcard "*", so the connected signal can't leak to other windows.
        "<script>try{if(window.opener){window.opener.postMessage("
        "'himmy-google-connected',window.location.origin);"
        "setTimeout(function(){window.close()},1200)}}"
        "catch(e){}</script></head><body><div>"
        f"<h1>{_html.escape(title)}</h1><p>{_html.escape(detail)}</p>"
        "</div></body></html>"
    )


@router.delete(
    "/google", dependencies=[Depends(studio_permission(_RES_GOOGLE, "write"))]
)
async def google_disconnect() -> Any:
    from himmy.api import studio_google

    try:
        return studio_google.disconnect()
    except studio_google.GoogleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.delete(
    "/google/client",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "write"))],
)
async def google_forget_client() -> Any:
    from himmy.api import studio_google

    try:
        return studio_google.forget_client()
    except studio_google.GoogleError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/google/gmail",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "read"))],
)
async def google_gmail_list(max_results: int = 20) -> Any:
    """Read the operator's connected Gmail inbox.

    red-team r6: requires the per-surface ``studio.google:read`` rather than the coarse
    console baseline, so a tenant browse role without the Google grant cannot read the
    operator's inbox.
    """
    from himmy.api import studio_google

    try:
        return await studio_google.gmail_list(max(1, min(max_results, 50)))
    except studio_google.GoogleNotConnectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except studio_google.GoogleError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/google/gmail/send",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "write"))],
)
async def google_gmail_send(body: GmailSendRequest) -> Any:
    from himmy.api import studio_google

    try:
        return await studio_google.gmail_send(body.to, body.subject, body.body)
    except studio_google.GoogleNotConnectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get(
    "/google/calendar",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "read"))],
)
async def google_calendar_list(max_results: int = 20) -> Any:
    """Read the operator's connected Google Calendar.

    red-team r6: requires ``studio.google:read`` rather than the coarse console baseline,
    so a tenant browse role without the Google grant cannot read the operator's calendar.
    """
    from himmy.api import studio_google

    try:
        return await studio_google.calendar_list(max(1, min(max_results, 50)))
    except studio_google.GoogleNotConnectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except studio_google.GoogleError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post(
    "/google/calendar",
    dependencies=[Depends(studio_permission(_RES_GOOGLE, "write"))],
)
async def google_calendar_create(body: CalendarCreateRequest) -> Any:
    from himmy.api import studio_google

    try:
        return await studio_google.calendar_create(
            body.summary, body.start, body.end, all_day=body.all_day
        )
    except studio_google.GoogleNotConnectedError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except studio_google.GoogleError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---- Approvals (human-in-the-loop inbox) --------------------------------


@router.get(
    "/approvals",
    response_model=list[ApprovalSummary],
    dependencies=[Depends(studio_permission(_RES_APPROVALS, "read"))],
)
async def approvals() -> list[ApprovalSummary]:
    """List runs paused awaiting a human decision.

    red-team r6: requires the per-surface ``studio.approvals:read`` rather than collapsing
    to the coarse ``studio.console:read`` baseline, so the HITL inbox honors its own grant.
    """
    return studio_approvals.list_pending()


@router.get(
    "/approvals/{checkpoint_id}",
    response_model=ApprovalDetail,
    dependencies=[Depends(studio_permission(_RES_APPROVALS, "read"))],
)
async def approval(checkpoint_id: str) -> ApprovalDetail:
    detail = studio_approvals.get_detail(checkpoint_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="approval not found")
    return detail


def _approval_actor(request: Request) -> str:
    """Who resolved this approval: the authenticated subject, else ``human``.

    On the zero-config single-user loopback default the principal is anonymous,
    so the meaningful label is the local human; with auth on, the real subject is
    stamped onto the APPROVAL_GRANTED/REJECTED event for who-approved-what mining.
    """
    subject = (getattr(get_principal(request), "subject", "") or "").strip()
    return subject if subject and subject != "anonymous" else "human"


def _resolve_stream(
    checkpoint_id: str, approved: bool, actor: str, tool_authorizer: Any = None
) -> StreamingResponse:
    # centralize-tool-gate: bind the request principal's gate ambiently for the resumed
    # run so the re-executed (approved) tool — and any further tools in the resumed loop —
    # are enforced. Inert offline (``tool_authorizer`` None). The resume rebuilds the
    # runtime via build_runtime_for_spec without an explicit authorizer, so the ambient
    # binding is what reaches the chokepoint.
    from himmy.services.tools.ambient import use_tool_authorizer

    async def _gen() -> AsyncIterator[str]:
        with use_tool_authorizer(tool_authorizer):
            async for event in studio_approvals.resolve(
                checkpoint_id, approved=approved, actor=actor
            ):
                yield _sse(event)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@router.post(
    "/approvals/{checkpoint_id}/approve",
    dependencies=[Depends(studio_permission(_RES_APPROVALS, "write"))],
)
async def approve(checkpoint_id: str, request: Request) -> StreamingResponse:
    """Approve the pending tool call and stream the resumed run."""
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    return _resolve_stream(
        checkpoint_id,
        True,
        _approval_actor(request),
        ToolCapabilityAuthorizer.from_request(request),
    )


@router.post(
    "/approvals/{checkpoint_id}/reject",
    dependencies=[Depends(studio_permission(_RES_APPROVALS, "write"))],
)
async def reject(checkpoint_id: str, request: Request) -> StreamingResponse:
    """Reject the pending tool call and stream the resumed run."""
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    return _resolve_stream(
        checkpoint_id,
        False,
        _approval_actor(request),
        ToolCapabilityAuthorizer.from_request(request),
    )


# ---- Models (available providers + models) ------------------------------


@router.get(
    "/models",
    dependencies=[Depends(studio_permission(_RES_MODEL_CATALOG, "read"))],
)
async def models(request: Request) -> list[dict[str, Any]]:
    """Available providers + their models, with any cached benchmark stats.

    red-team r6: requires the per-surface ``studio.modelcatalog:read`` rather than
    collapsing to the coarse ``studio.console:read`` baseline, so the catalog honors its
    own grant.

    red-team reattack-r6: this BENIGN model picker is gated by ``studio.modelcatalog:read``
    (a tenant-grantable browse read), split off ``studio.models`` — which now guards ONLY
    the operator provider-credential posture surface (GET /api/studio/models/providers) and
    is admin-only. The catalog carries NO provider-key configured/detected_via posture, so a
    tenant browse role keeps the picker while losing the credential reconnaissance.

    red-team reattack-r9: the catalog's host-derived ``available`` booleans (which LLM
    binaries are installed/running) and live LOCAL Ollama model inventory are the SAME
    operator deployment posture r6 withheld from tenant browse roles on ``GET /health``.
    They are now disclosed ONLY to a caller that also holds ``studio.console:write``
    (admin) — :func:`_caller_holds_console_write`, the same gate ``/health`` uses. A tenant
    browse role gets a POSTURE-FREE picker (providers advertised as SUPPORTED, no live
    inventory); OFFLINE is byte-unchanged (``_caller_holds_console_write`` returns True with
    no authenticator) so the single-box console keeps the full catalog.

    Delegates to the shared :func:`himmy.services.inference.compare.build_model_catalog`
    seam (T3d) — the SAME catalog ``GET /v1/models`` and ``himmy models`` render, so the
    three surfaces never drift. The response shape is unchanged.
    """
    from himmy.services.inference.compare import build_model_catalog

    posture = _caller_holds_console_write(request)
    return cast(
        list[dict[str, Any]],
        await build_model_catalog(reveal_host_posture=posture),
    )


# ---- Compare (one prompt, N models, side-by-side) -----------------------


class CompareTarget(BaseModel):
    """One provider+model cell in a comparison."""

    provider: str = Field(..., max_length=40)
    model: str = Field(..., max_length=80)


class CompareRequest(BaseModel):
    """A single prompt fanned out across several models."""

    prompt: str = Field(..., min_length=1, max_length=8000)
    system: str | None = Field(None, max_length=4000)
    targets: list[CompareTarget] = Field(..., min_length=1, max_length=6)
    timeout_seconds: float = Field(120.0, ge=1, le=600)


class CompareResult(BaseModel):
    """The outcome of running the prompt against one target."""

    provider: str
    model: str
    ok: bool
    output: str = ""
    error: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float | None = None
    latency_ms: float | None = None


@router.post(
    "/compare",
    response_model=list[CompareResult],
    dependencies=[Depends(studio_permission(_RES_MODELS, "write"))],
)
async def compare(body: CompareRequest) -> list[CompareResult]:
    """Run one prompt across several models concurrently; return outputs + usage.

    Each target is an isolated model call (no tools, no run history) so the cells
    are directly comparable on output, tokens, cost, and latency.

    Delegates to the shared :func:`himmy.services.inference.compare.run_compare` seam
    (T3d) — the SAME per-target loop ``POST /v1/models/compare`` and ``himmy compare``
    drive. The response shape is unchanged.
    """
    from himmy.services.inference.compare import CompareTargetSpec, run_compare

    cells = await run_compare(
        [CompareTargetSpec(provider=t.provider, model=t.model) for t in body.targets],
        prompt=body.prompt,
        system=body.system,
        timeout_seconds=body.timeout_seconds,
    )
    return [
        CompareResult(
            provider=cell.provider,
            model=cell.model,
            ok=cell.ok,
            output=cell.output,
            error=cell.error,
            input_tokens=cell.input_tokens,
            output_tokens=cell.output_tokens,
            cost=cell.cost,
            latency_ms=cell.latency_ms,
        )
        for cell in cells
    ]


# ---- Tasks --------------------------------------------------------------


class TaskAddRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    due: str | None = Field(None, max_length=10)


class TaskDoneRequest(BaseModel):
    done: bool = True


@router.get("/tasks", dependencies=[Depends(studio_permission(_RES_TASKS, "read"))])
async def tasks_list() -> list[Any]:
    from himmy.api.studio_tasks import get_tasks_store

    return get_tasks_store().list()


@router.post("/tasks", dependencies=[Depends(studio_permission(_RES_TASKS, "write"))])
async def tasks_add(body: TaskAddRequest) -> Any:
    from himmy.api.studio_tasks import get_tasks_store

    return get_tasks_store().add(body.title, due=body.due)


@router.patch(
    "/tasks/{task_id}", dependencies=[Depends(studio_permission(_RES_TASKS, "write"))]
)
async def tasks_done(task_id: str, body: TaskDoneRequest) -> dict[str, bool]:
    from himmy.api.studio_tasks import get_tasks_store

    return {"ok": get_tasks_store().set_done(task_id, body.done)}


@router.delete(
    "/tasks/{task_id}", dependencies=[Depends(studio_permission(_RES_TASKS, "write"))]
)
async def tasks_delete(task_id: str) -> dict[str, bool]:
    from himmy.api.studio_tasks import get_tasks_store

    return {"ok": get_tasks_store().delete(task_id)}


# ---- Chats (saved, resumable conversations) -----------------------------


class ChatMessageIn(BaseModel):
    role: str = Field(..., pattern="^(user|agent)$")
    text: str = Field("", max_length=100000)


class ChatSaveRequest(BaseModel):
    id: str | None = None
    title: str | None = Field(None, max_length=200)
    agent_path: str | None = Field(None, max_length=500)
    provider: str | None = Field(None, max_length=40)
    project_id: str | None = Field(None, max_length=64)
    messages: list[ChatMessageIn] = Field(default_factory=list)


class ChatRenameRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


@router.get("/chats", dependencies=[Depends(studio_permission(_RES_CHATS, "read"))])
async def chats_list() -> Any:
    from himmy.api.studio_chats import get_chats_store

    return get_chats_store().list()


@router.get(
    "/chats/{session_id}",
    dependencies=[Depends(studio_permission(_RES_CHATS, "read"))],
)
async def chats_get(session_id: str) -> Any:
    from himmy.api.studio_chats import get_chats_store

    detail = get_chats_store().get(session_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="unknown chat session")
    return detail


@router.post("/chats", dependencies=[Depends(studio_permission(_RES_CHATS, "write"))])
async def chats_save(body: ChatSaveRequest) -> Any:
    from himmy.api.studio_chats import ChatMessage, get_chats_store

    return get_chats_store().save(
        session_id=body.id,
        title=body.title,
        agent_path=body.agent_path,
        provider=body.provider,
        project_id=body.project_id,
        messages=[ChatMessage(role=m.role, text=m.text) for m in body.messages],
    )


@router.patch(
    "/chats/{session_id}",
    dependencies=[Depends(studio_permission(_RES_CHATS, "write"))],
)
async def chats_rename(session_id: str, body: ChatRenameRequest) -> dict[str, bool]:
    from himmy.api.studio_chats import get_chats_store

    return {"ok": get_chats_store().rename(session_id, body.title)}


@router.delete(
    "/chats/{session_id}",
    dependencies=[Depends(studio_permission(_RES_CHATS, "write"))],
)
async def chats_delete(session_id: str) -> dict[str, bool]:
    from himmy.api.studio_chats import get_chats_store

    return {"ok": get_chats_store().delete(session_id)}


# ---- Cookbook (saved agent + prompt recipes) ----------------------------


class RecipeUpsertRequest(BaseModel):
    id: str | None = None
    name: str = Field(..., min_length=1, max_length=200)
    agent_path: str = Field("", max_length=500)
    prompt: str = Field("", max_length=20_000)
    notes: str = Field("", max_length=4000)


@router.get(
    "/cookbook", dependencies=[Depends(studio_permission(_RES_COOKBOOK, "read"))]
)
async def cookbook_list() -> list[Any]:
    from himmy.api.studio_cookbook import get_cookbook_store

    return get_cookbook_store().list()


@router.put(
    "/cookbook", dependencies=[Depends(studio_permission(_RES_COOKBOOK, "write"))]
)
async def cookbook_upsert(body: RecipeUpsertRequest) -> Any:
    from himmy.api.studio_cookbook import Recipe, get_cookbook_store

    r = Recipe(
        name=body.name,
        agent_path=body.agent_path,
        prompt=body.prompt,
        notes=body.notes,
    )
    if body.id:
        r.id = body.id
    return get_cookbook_store().upsert(r)


@router.delete(
    "/cookbook/{recipe_id}",
    dependencies=[Depends(studio_permission(_RES_COOKBOOK, "write"))],
)
async def cookbook_delete(recipe_id: str) -> dict[str, bool]:
    from himmy.api.studio_cookbook import get_cookbook_store

    return {"ok": get_cookbook_store().delete(recipe_id)}


# ---- Notes --------------------------------------------------------------


class NoteUpsertRequest(BaseModel):
    id: str | None = None
    title: str = Field("", max_length=300)
    body: str = Field("", max_length=200_000)


@router.get("/notes", dependencies=[Depends(studio_permission(_RES_NOTES, "read"))])
async def notes_list() -> list[Any]:
    from himmy.api.studio_notes import get_notes_store

    return get_notes_store().list()


@router.get(
    "/notes/{note_id}",
    dependencies=[Depends(studio_permission(_RES_NOTES, "read"))],
)
async def notes_get(note_id: str) -> Any:
    from himmy.api.studio_notes import get_notes_store

    note = get_notes_store().get(note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="note not found")
    return note


@router.put("/notes", dependencies=[Depends(studio_permission(_RES_NOTES, "write"))])
async def notes_upsert(body: NoteUpsertRequest) -> Any:
    from himmy.api.studio_notes import Note, get_notes_store

    note = Note(title=body.title, body=body.body)
    if body.id:
        note.id = body.id
    return get_notes_store().upsert(note)


@router.delete(
    "/notes/{note_id}", dependencies=[Depends(studio_permission(_RES_NOTES, "write"))]
)
async def notes_delete(note_id: str) -> dict[str, bool]:
    from himmy.api.studio_notes import get_notes_store

    return {"ok": get_notes_store().delete(note_id)}


# ---- Calendar -----------------------------------------------------------


class CalendarAddRequest(BaseModel):
    date: str = Field(..., min_length=8, max_length=10)  # YYYY-MM-DD
    title: str = Field(..., min_length=1, max_length=500)
    time: str | None = Field(None, max_length=5)  # HH:MM
    notes: str = Field("", max_length=4000)


@router.get(
    "/calendar", dependencies=[Depends(studio_permission(_RES_CALENDAR, "read"))]
)
async def calendar_list(month: str | None = None) -> list[Any]:
    from himmy.api.studio_calendar import get_calendar_store

    return get_calendar_store().list(month=month)


@router.post(
    "/calendar", dependencies=[Depends(studio_permission(_RES_CALENDAR, "write"))]
)
async def calendar_add(body: CalendarAddRequest) -> Any:
    from himmy.api.studio_calendar import CalendarEvent, get_calendar_store

    ev = CalendarEvent(
        date=body.date, title=body.title, time=body.time or None, notes=body.notes
    )
    return get_calendar_store().add(ev)


@router.delete(
    "/calendar/{event_id}",
    dependencies=[Depends(studio_permission(_RES_CALENDAR, "write"))],
)
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
    # Optional cutoff: below it a memory is not recalled (None = always return the top).
    similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)


def _scoped_memory_subject(request: Request, requested: str | None) -> str | None:
    """The data subject a memory read may target, derived from the verified PRINCIPAL.

    The centralized subject-axis (BOLA) gate for the subject-keyed memory surface — the
    by-construction fix for the cross-subject recall/list leak (a tenant-bound principal
    holding ``studio.memory:read`` recalling ANOTHER data subject's semantic memories from
    the shared process store). For a ``subject_scoped`` principal WITHOUT ``tenant_admin``
    the effective subject is FORCED to its own (a client-supplied ``subject_id`` for another
    subject is ignored, never honored — mirroring :func:`narrow_subject` on the runs reader).
    Every other principal — ``all_tenants`` / offline / the historical multi-user-workspace
    default / a ``tenant_admin`` — keeps ``requested`` as-is, so the zero-config path is
    byte-unchanged.
    """
    return narrow_subject(request, requested)


@router.get(
    "/memory/subjects",
    response_model=list[str],
    dependencies=[
        Depends(studio_permission(_RES_MEMORY, "read")),
        Depends(scoped_read),
    ],
)
async def memory_subjects(request: Request) -> list[str]:
    """Distinct subject ids with memories, NARROWED to the principal's own subject.

    A ``subject_scoped`` principal sees ONLY its own subject in the enumeration (the
    cross-subject enumeration oracle is closed); every other principal sees all. The
    tenant axis (the process-global store carries no ``workspace_id`` column) remains the
    documented data-model gap drained separately. NO-OP offline / ``all_tenants``.
    """
    from himmy.api import studio_memory

    pinned = _scoped_memory_subject(request, None)
    return studio_memory.list_subjects(only_subject=pinned)


@router.get(
    "/memory",
    dependencies=[
        Depends(studio_permission(_RES_MEMORY, "read")),
        Depends(scoped_read),
    ],
)
async def memory_list(request: Request, subject: str = "default") -> list[Any]:
    """A subject's memories — the ``subject`` query param is never honored over the principal.

    For a ``subject_scoped`` caller the effective subject is pinned to its own (a request
    for another subject is silently narrowed, so it can only ever read its own memories).
    NO-OP offline / ``all_tenants`` — the caller-supplied ``subject`` is used verbatim.
    """
    from himmy.api import studio_memory

    return studio_memory.list_memories(_scoped_memory_subject(request, subject))


@router.post(
    "/memory",
    dependencies=[
        Depends(studio_permission(_RES_MEMORY, "write")),
        Depends(subject_write),
    ],
)
async def memory_add(body: MemoryAddRequest, request: Request) -> Any:
    """Persist a memory — STAMPED under the principal's own subject when subject-scoped.

    The write-path BOLA companion: a ``subject_scoped`` principal may only write under its
    own subject (``enforce_subject_write`` → 403 on a foreign ``subject_id`` in the body),
    so it cannot poison another data subject's memory store. NO-OP offline / ``all_tenants``.
    """
    from himmy.api import studio_memory

    enforce_subject_write(request, body.subject_id)
    return studio_memory.add_memory(
        body.text, subject_id=body.subject_id, kind=body.kind
    )


@router.delete(
    "/memory/{memory_id}",
    dependencies=[
        Depends(studio_permission(_RES_MEMORY, "write")),
        Depends(subject_write),
    ],
)
async def memory_forget(memory_id: str, request: Request) -> dict[str, bool]:
    """Forget one memory — a ``subject_scoped`` caller may only forget its OWN subject's.

    The record's owning ``subject_id`` is checked against the principal: a cross-subject
    delete folds to a uniform 404 (existence never leaked), mirroring the runs by-id
    reader. NO-OP offline / ``all_tenants``.
    """
    from himmy.api import studio_memory

    rec = studio_memory.get_memory(memory_id)
    if rec is not None and not authorize_object(request, rec.subject_id):
        raise HTTPException(status_code=404, detail=f"unknown memory {memory_id!r}")
    return {"ok": studio_memory.forget(memory_id)}


@router.post(
    "/memory/recall",
    dependencies=[
        Depends(studio_permission(_RES_MEMORY, "read")),
        Depends(scoped_read),
    ],
)
async def memory_recall(body: MemoryRecallRequest, request: Request) -> list[Any]:
    """Semantic recall — ``subject_id`` from the body is NEVER honored over the principal.

    This POST performs a READ: a tenant-bound / ``subject_scoped`` principal could
    otherwise recall another data subject's semantic memories from the shared process
    store by passing their ``subject_id`` in the body (a cross-subject BOLA the GET-only
    coverage gate could not see). The subject is now derived from the verified principal
    via :func:`_scoped_memory_subject` — a ``subject_scoped`` caller is pinned to its own
    subject, the body value ignored. NO-OP offline / ``all_tenants``.
    """
    from himmy.api import studio_memory

    return await studio_memory.recall(
        body.query,
        subject_id=_scoped_memory_subject(request, body.subject_id),
        top_k=body.top_k,
        similarity_threshold=body.similarity_threshold,
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


@router.get(
    "/knowledge", dependencies=[Depends(studio_permission(_RES_KNOWLEDGE, "read"))]
)
async def kb_list() -> list[Any]:
    from himmy.api import studio_knowledge

    return studio_knowledge.list_kbs()


@router.post(
    "/knowledge", dependencies=[Depends(studio_permission(_RES_KNOWLEDGE, "write"))]
)
async def kb_create(body: KbCreateRequest) -> Any:
    from himmy.api import studio_knowledge

    return await studio_knowledge.create_kb(body.name)


@router.post(
    "/knowledge/{kb_id}/ingest",
    dependencies=[Depends(studio_permission(_RES_KNOWLEDGE, "write"))],
)
async def kb_ingest(kb_id: str, body: KbIngestRequest) -> Any:
    from himmy.api import studio_knowledge

    return await studio_knowledge.ingest_text(kb_id, body.text, title=body.title)


@router.post(
    "/knowledge/{kb_id}/search",
    dependencies=[Depends(studio_permission(_RES_KNOWLEDGE, "read"))],
)
async def kb_search(kb_id: str, body: KbSearchRequest) -> list[Any]:
    from himmy.api import studio_knowledge

    return await studio_knowledge.search(kb_id, body.query, top_k=body.top_k)


@router.delete(
    "/knowledge/{kb_id}",
    dependencies=[Depends(studio_permission(_RES_KNOWLEDGE, "write"))],
)
async def kb_delete(kb_id: str) -> dict[str, bool]:
    from himmy.api import studio_knowledge

    return {"ok": await studio_knowledge.delete_kb(kb_id)}


# ---- Evaluation (suites → run → scorecard) ------------------------------


class EvalRunRequest(BaseModel):
    suite_path: str
    agent_path: str
    provider: str | None = None
    model: str | None = None


@router.get(
    "/evals", dependencies=[Depends(studio_permission(_RES_EVALS, "read"))]
)
async def eval_suites() -> list[Any]:
    # red-team reattack-r8: this is the un-hardened twin of GET /api/studio/eval/suites
    # (studio_eval.list_suites). Both call ``studio_eval.discover_suites()`` — which
    # enumerates the operator's local-filesystem eval-suite names/paths/case-counts — but
    # this route previously carried no route-level guard, so it inherited only the
    # router-level ``studio.console:read`` baseline that every tenant browse role holds,
    # bypassing the r7 lockdown that withheld ``studio.eval`` from those roles. We now gate
    # it on ``studio.evals:read`` AND move ``studio.evals`` into
    # :data:`~himmy.api.auth.rbac._STUDIO_GLOBAL_STORE_RESOURCES` (admin-only) so this
    # operator-local FS reconnaissance surface is withheld from tenant-facing browse roles,
    # exactly like its r7-locked twin. The OFFLINE path is unaffected (``studio_permission``
    # /``require_permission`` no-op without an authenticator).
    from himmy.api import studio_eval

    return studio_eval.discover_suites()


@router.post(
    "/evals/run", dependencies=[Depends(studio_permission(_RES_EVALS, "write"))]
)
async def eval_run(body: EvalRunRequest, request: Request) -> Any:
    import asyncio

    from himmy.api import studio_eval
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    # centralize-tool-gate: thread the request principal's gate so eval-run tools are
    # enforced (inert offline). Bound ambiently inside ``run_eval``.
    tool_authorizer = ToolCapabilityAuthorizer.from_request(request)
    try:
        return await asyncio.wait_for(
            studio_eval.run_eval(
                body.suite_path,
                body.agent_path,
                provider=body.provider,
                model=body.model,
                tool_authorizer=tool_authorizer,
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


@router.get(
    "/workflows",
    dependencies=[Depends(studio_permission(_RES_WORKFLOWS, "read"))],
)
async def workflows() -> list[Any]:
    # red-team reattack-r8: ``discover_workflows()`` enumerates the operator's local
    # workflow specs (project-relative path + name + step/tool graph) — operator
    # orchestration topology a tenant must not see. This read previously carried no
    # route-level dependency, so it was gated only by the router-level ``studio.console:read``
    # baseline every tenant browse role holds (the same forgotten-read-guard pattern as the
    # ``/evals`` twin above). We gate it on ``studio.workflows:read`` AND move
    # ``studio.workflows`` into
    # :data:`~himmy.api.auth.rbac._STUDIO_GLOBAL_STORE_RESOURCES` (admin-only) — the
    # discovery globs ``project_root()`` with no tenant axis to intersect on, so the fix is
    # to withhold the resource (mirroring r7's eval/routines lockdown) rather than retrofit a
    # filter. Write/manage stay ``studio.workflows:write`` = admin. The OFFLINE path is
    # unaffected (``studio_permission`` no-ops without an authenticator).
    from himmy.api import studio_workflows

    return studio_workflows.discover_workflows()


@router.post(
    "/workflows/run",
    dependencies=[Depends(studio_permission(_RES_WORKFLOWS, "write"))],
)
async def workflow_run(body: WorkflowRunRequest, request: Request) -> Any:
    import asyncio

    from himmy.api import studio_workflows
    from himmy.services.tools.capability import ToolCapabilityAuthorizer

    # centralize-tool-gate: thread the request principal's gate so workflow step tools are
    # enforced (inert offline). Bound ambiently inside ``run_workflow``.
    tool_authorizer = ToolCapabilityAuthorizer.from_request(request)
    try:
        return await asyncio.wait_for(
            studio_workflows.run_workflow(
                body.workflow_path,
                body.agent_path,
                provider=body.provider,
                model=body.model,
                initial_state=body.initial_state,
                tool_authorizer=tool_authorizer,
            ),
            timeout=900,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="workflow timed out") from exc


# ---- Lineage (provenance of a run's answer) -----------------------------


@router.get(
    "/runs/{run_id}/lineage",
    dependencies=[
        Depends(studio_permission(_RES_RUNS, "read")),
        Depends(scoped_read),
    ],
)
async def run_lineage(run_id: str, request: Request) -> Any:
    from himmy.api import studio_lineage

    view = studio_lineage.run_lineage(run_id)
    if view is None or not await _authorize_run(request, run_id):
        raise HTTPException(status_code=404, detail="run not found")
    return view


# ---- Agent authoring (the no-code builder) ------------------------------


@router.get(
    "/tools",
    response_model=list[studio_agents.PackInfo],
    dependencies=[Depends(studio_permission(_RES_AGENTS, "read"))],
)
async def tool_packs() -> list[studio_agents.PackInfo]:
    """The built-in tool packs an agent can switch on.

    Part of the agent-authoring surface, so gated with ``studio.agents:read``
    (admin-only by default) rather than the open ``studio.console:read``
    baseline — the same r10 bar that locked the agent/team inventory. NO-OP
    offline / ``all_tenants``.
    """
    return studio_agents.list_tool_packs()


@router.get(
    "/skills",
    response_model=list[studio_agents.SkillInfo],
    dependencies=[Depends(studio_permission(_RES_AGENTS, "read"))],
)
async def skills() -> list[studio_agents.SkillInfo]:
    """Available skills (built-in + project-local).

    Authoring-surface inventory, gated with ``studio.agents:read`` (admin-only
    by default) like the tool packs and the agent/team lists. NO-OP offline /
    ``all_tenants``.
    """
    return studio_agents.list_skill_infos()


@router.get(
    "/agent",
    response_model=studio_agents.AgentDetail,
    dependencies=[Depends(studio_permission(_RES_AGENTS, "read"))],
)
async def get_agent(path: str) -> studio_agents.AgentDetail:
    """Load one agent's full editable spec (by project-relative path).

    The detail view discloses strictly more than the (admin-locked) ``/agents``
    list — the full system-prompt body, provider/model, tool packs, skills and
    the project-relative spec path — so it carries the same ``studio.agents:read``
    bar (admin-only by default) the r10 round applied to the inventory list,
    rather than the open ``studio.console:read`` baseline a tenant browse role
    holds. NO-OP offline / ``all_tenants``.
    """
    try:
        return studio_agents.load_agent_detail(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agents/validate", response_model=studio_agents.ValidationResult)
async def validate_agent(spec: dict[str, Any]) -> studio_agents.ValidationResult:
    """Validate a proposed spec without saving (live form feedback)."""
    errors = studio_agents.validate_spec(spec)
    return studio_agents.ValidationResult(ok=not errors, errors=errors)


@router.put(
    "/agents",
    response_model=studio_service.AgentSummary,
    dependencies=[Depends(studio_permission(_RES_AGENTS, "write"))],
)
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
