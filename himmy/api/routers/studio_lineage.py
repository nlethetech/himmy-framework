"""API kernel: Studio lineage router — the provenance-graph read surface.

Mounted under ``/api/studio/lineage`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). Two endpoints:

* ``GET /graph``  — a bounded subgraph of the entity registry around an entity
  (``entity_id`` = record id, stable id, or a raw thread id) or a run
  (``run_id`` — resolved through the ``/v1`` run service first, then the local
  Studio run store's ``thread_id``). Depth- and node-bounded so a large
  registry can never melt the browser.
* ``GET /entity/{record_id}`` — one record in full (payload + metadata +
  incident links) for the side detail panel.

Offline-first: when nothing was projected into the registry the endpoints
answer with a clear 404 (the Studio GUI then falls back to the cognition-trace
provenance view), never a 500.

Tenant scoping (red-team scope-r5). ``studio.lineage:read`` is withheld to admin-only, but a
TENANT-BOUND ``admin`` key (``roles:["admin"], tenant_ids:["A"], all_tenants:false``) holds it
— so the lineage readers MUST refuse cross-tenant payloads, not trust the admin-only gate. The
run path threads the principal's :func:`~himmy.api.auth.context.resolve_workspace` scope into
:meth:`RunAppService.get_run_lineage` (run-level tenant scope, exactly the ``/v1``
``/runs/{id}/lineage`` reader). The entity path derives a traced subgraph's OWNING workspace
from its ``chat_thread`` hub (the only reliably workspace-stamped node) and refuses it (clean
404) unless the principal's :func:`~himmy.api.auth.context.studio_tenant_filter` allows it —
**failing closed** for a tenant-bound caller when no owning workspace can be established, so an
unstamped record can never leak. Every check is a strict NO-OP for an unrestricted
(``all_tenants`` / offline) principal (filter ``None``), so the single-box path is byte-unchanged.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from fastapi import Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from himmy.api.auth import resolve_workspace, scoped_read, studio_tenant_filter
from himmy.api.routers.studio_common import build_studio_router
from himmy.entities.lineage import LineageGraph
from himmy.entities.records import EntityRecord, stable_id_for

router = build_studio_router("lineage", tag="studio-lineage")

#: The entity kind whose record reliably carries the owning ``workspace_id`` — the lineage hub
#: a run/persona/prompt/context snapshot all hang off (stamped by ``RunAppService._persist`` /
#: ``create_thread`` under ``THREAD_WORKSPACE_KEY``). The thread's own ``metadata`` rides into
#: its projected record's ``payload['metadata']`` via ``model_dump``, so the owning tenant of a
#: traced subgraph is read off any ``chat_thread`` node here.
_THREAD_KIND = "chat_thread"

# ---- bounds ---------------------------------------------------------------

#: Hard cap on nodes returned by ``/graph`` (BFS order, root first) — keeps the
#: SVG renderer responsive no matter how dense the registry is.
MAX_GRAPH_NODES = 80
#: Depth (hop) budget bounds for ``/graph``.
MAX_GRAPH_DEPTH = 6
DEFAULT_GRAPH_DEPTH = 3
#: Ids are UUIDs (36 chars) or short semantic keys — 200 is generous.
_MAX_ID_LEN = 200
#: Node labels are a one-line whisper, not the payload.
_MAX_LABEL_LEN = 120
#: Per-string cap inside the detail payload/metadata (the side panel is a
#: read surface, not an export path).
_MAX_DETAIL_STR = 4000
#: Per-collection cap inside the detail payload/metadata.
_MAX_DETAIL_ITEMS = 100
#: Incident links listed per direction in the detail view.
_MAX_DETAIL_LINKS = 50

#: Payload keys tried (in order) when deriving a human label for a node.
_LABEL_KEYS = (
    "name",
    "title",
    "label",
    "prompt",
    "summary",
    "query",
    "text",
    "content",
)


# ---- response shapes ------------------------------------------------------


class GraphNode(BaseModel):
    """One node of the lineage subgraph (a registry record, summarised)."""

    id: str
    kind: str
    label: str
    created_at: str
    stable_id: str
    version: int


class GraphEdge(BaseModel):
    """One typed directed edge of the lineage subgraph."""

    from_: str = Field(serialization_alias="from")
    to: str
    relation: str


class GraphResponse(BaseModel):
    """A bounded lineage subgraph: BFS from the root, depth- and node-capped."""

    root_id: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    truncated: bool
    node_count: int
    edge_count: int


class LinkSummary(BaseModel):
    """An incident link, with the far record summarised for display."""

    relation: str
    other_id: str
    other_kind: str | None = None
    other_label: str | None = None


class EntityDetail(BaseModel):
    """The full record behind a node — the side panel's data."""

    id: str
    stable_id: str
    kind: str
    version: int
    created_at: str
    label: str
    payload: dict[str, Any]
    metadata: dict[str, Any]
    links_in: list[LinkSummary]
    links_out: list[LinkSummary]


# ---- helpers ---------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    """Await ``value`` when awaitable — drives sync and async registries alike."""
    if inspect.isawaitable(value):
        return await value
    return value


def _registry(request: Request) -> Any:
    """The wired entity registry, or a clear 404 when none exists."""
    container = getattr(request.app.state, "container", None)
    registry = getattr(container, "entity_registry", None)
    if registry is None:
        raise HTTPException(
            status_code=404,
            detail="no entity registry is wired on this server — "
            "lineage graphs are unavailable",
        )
    return registry


def _label_of(record: EntityRecord) -> str:
    """A one-line human label for a record, derived from its payload."""
    for key in _LABEL_KEYS:
        raw = record.payload.get(key)
        if isinstance(raw, str) and raw.strip():
            return re.sub(r"\s+", " ", raw.strip())[:_MAX_LABEL_LEN]
    return f"{record.kind} {record.stable_id[:8]}"


def _bounded(value: Any, *, depth: int = 0) -> Any:
    """A display-safe copy: long strings clipped, huge collections capped.

    The side panel is a read surface; clipping keeps a pathological payload
    (megabyte strings, ten-thousand-element lists) from stalling the UI while
    leaving normal records byte-identical.
    """
    if isinstance(value, str):
        if len(value) > _MAX_DETAIL_STR:
            return value[:_MAX_DETAIL_STR] + "… [truncated]"
        return value
    if depth >= 6:  # nested beyond display usefulness
        return "… [nested]"
    if isinstance(value, dict):
        items = list(value.items())[:_MAX_DETAIL_ITEMS]
        out: dict[str, Any] = {str(k): _bounded(v, depth=depth + 1) for k, v in items}
        if len(value) > _MAX_DETAIL_ITEMS:
            out["…"] = f"[{len(value) - _MAX_DETAIL_ITEMS} more keys truncated]"
        return out
    if isinstance(value, list):
        clipped = [_bounded(v, depth=depth + 1) for v in value[:_MAX_DETAIL_ITEMS]]
        if len(value) > _MAX_DETAIL_ITEMS:
            clipped.append(f"… [{len(value) - _MAX_DETAIL_ITEMS} more truncated]")
        return clipped
    return value


async def _resolve_entity(registry: Any, entity_id: str) -> EntityRecord | None:
    """Resolve an id to a record: record id → stable id → raw thread id."""
    record = await _maybe_await(registry.get(entity_id))
    if record is not None:
        return record  # type: ignore[no-any-return]
    record = await _maybe_await(registry.get_latest(entity_id))
    if record is not None:
        return record  # type: ignore[no-any-return]
    # Convenience: a non-UUID conversation id (e.g. "cid:desk") hashes into the
    # chat_thread namespace — the lineage hub a run hangs off.
    derived = stable_id_for(entity_id, namespace="chat_thread")
    if derived != entity_id:
        record = await _maybe_await(registry.get_latest(derived))
    return record  # type: ignore[no-any-return]


async def _graph_for_run(
    request: Request, registry: Any, run_id: str, depth: int
) -> LineageGraph:
    """Trace the lineage for a run id, or raise a helpful 404.

    Tries the entity-projected ``/v1`` run service first, then the local
    Studio run store (whose runs carry a ``thread_id`` but are only present in
    the registry when the runtime projected the thread).

    Tenant scoping (red-team scope-r5): the run is resolved tenant-scoped — the
    principal's :func:`resolve_workspace` is threaded into
    :meth:`RunAppService.get_run_lineage` (a run owned by another tenant resolves to
    ``None`` → 404, exactly as the ``/v1`` ``/runs/{id}/lineage`` reader), and the
    Studio run-store fallback's resolved subgraph is run through the SAME entity tenant
    gate. For an unrestricted principal ``resolve_workspace`` returns ``None`` ⇒ the
    legacy cross-workspace resolution, byte-unchanged.
    """
    workspace_id = resolve_workspace(request, None)
    container = getattr(request.app.state, "container", None)
    run_app = getattr(container, "run_app", None)
    if run_app is not None:
        graph = await _maybe_await(
            run_app.get_run_lineage(
                run_id, workspace_id=workspace_id, max_depth=depth
            )
        )
        if graph is not None:
            return graph  # type: ignore[no-any-return]

    from himmy.api.studio_runs import get_run_store

    run = get_run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
    if not run.thread_id:
        raise HTTPException(
            status_code=404,
            detail="this run has no thread — nothing was projected into the "
            "lineage registry",
        )
    root = await _resolve_entity(registry, run.thread_id)
    if root is None:
        raise HTTPException(
            status_code=404,
            detail="this run was not projected into the entity registry — "
            "see its cognition-trace provenance instead",
        )
    # The Studio run-store fallback resolves the thread WITHOUT a tenant axis, so gate the
    # traced subgraph the same way the entity path does (404 cross-tenant / unanchored).
    return await _authorize_entity_subgraph(request, registry, root.record_id, depth)


async def _trace(registry: Any, record_id: str, depth: int) -> LineageGraph:
    """Run the registry's bounded BFS trace (sync or async backend)."""
    return await _maybe_await(  # type: ignore[no-any-return]
        registry.trace(record_id, max_depth=depth)
    )


# ---- tenant scoping (red-team scope-r5) -----------------------------------


def _record_workspace(record: EntityRecord) -> str | None:
    """The owning ``workspace_id`` stamped on a record, or ``None`` when unstamped.

    Checks the record's own ``metadata['workspace_id']`` first, then — for a
    ``chat_thread`` hub, whose own ``metadata`` rides into its projected
    ``payload['metadata']`` — the payload stamp. Returns ``None`` for any record carrying
    no tenant stamp (the caller fails closed for a tenant-bound principal on ``None``).
    """
    stamped = record.metadata.get("workspace_id")
    if isinstance(stamped, str) and stamped:
        return stamped
    payload_meta = record.payload.get("metadata")
    if isinstance(payload_meta, dict):
        ws = payload_meta.get("workspace_id")
        if isinstance(ws, str) and ws:
            return ws
    return None


def _graph_owning_workspace(graph: LineageGraph) -> str | None:
    """The owning ``workspace_id`` of a traced subgraph, or ``None`` when undeterminable.

    Reads the stamp off the subgraph's ``chat_thread`` hub (the reliably workspace-stamped
    node) — and, as a fallback, off any node carrying a direct ``metadata['workspace_id']``
    — so the tenant gate has a tenant to compare against. ``None`` means the subgraph has no
    workspace anchor, which a tenant-bound caller treats as not-found (fail closed).
    """
    for rec in graph.nodes.values():
        if rec.kind == _THREAD_KIND:
            ws = _record_workspace(rec)
            if ws is not None:
                return ws
    for rec in graph.nodes.values():
        ws = _record_workspace(rec)
        if ws is not None:
            return ws
    return None


async def _authorize_entity_subgraph(
    request: Request, registry: Any, record_id: str, depth: int
) -> LineageGraph:
    """Trace an entity AND tenant-gate the result (the centralized entity-path scope).

    The single choke point both ``/graph?entity_id=`` and the detail walk route through, so
    a tenant-bound caller can never read a subgraph (and its payloads) outside its tenant
    allow-list. For an unrestricted (``all_tenants`` / offline) principal
    :func:`studio_tenant_filter` is ``None`` ⇒ a strict NO-OP (byte-unchanged). For a
    tenant-bound principal the subgraph's owning workspace (its ``chat_thread`` hub stamp)
    must be in the allow-list, else **404** — failing closed when the subgraph carries no
    workspace anchor, so an unstamped record cannot leak across tenants.
    """
    graph = await _trace(registry, record_id, depth)
    tenants = studio_tenant_filter(request)
    if tenants is None:
        return graph
    owner = _graph_owning_workspace(graph)
    if owner is None or owner not in tenants:
        raise HTTPException(status_code=404, detail="no entity found for that id")
    return graph


def _shape_graph(graph: LineageGraph) -> GraphResponse:
    """Project a traced graph into the wire shape, node-capped in BFS order."""
    ids = list(graph.nodes)
    pruned = len(ids) > MAX_GRAPH_NODES
    keep = set(ids[:MAX_GRAPH_NODES])
    nodes = [
        GraphNode(
            id=rec.record_id,
            kind=rec.kind,
            label=_label_of(rec),
            created_at=rec.created_at,
            stable_id=rec.stable_id,
            version=rec.version,
        )
        for rid, rec in graph.nodes.items()
        if rid in keep
    ]
    edges = [
        GraphEdge(from_=e.from_record_id, to=e.to_record_id, relation=e.relation)
        for e in graph.edges
        if e.from_record_id in keep and e.to_record_id in keep
    ]
    return GraphResponse(
        root_id=graph.root_id,
        nodes=nodes,
        edges=edges,
        truncated=graph.truncated or pruned,
        node_count=len(nodes),
        edge_count=len(edges),
    )


async def _link_summaries(
    registry: Any, links: list[Any], *, other_side: str
) -> list[LinkSummary]:
    """Summarise incident links, resolving the far record for display."""
    out: list[LinkSummary] = []
    for link in links[:_MAX_DETAIL_LINKS]:
        other_id = getattr(link, other_side)
        other = await _maybe_await(registry.get(other_id))
        out.append(
            LinkSummary(
                relation=link.relation,
                other_id=other_id,
                other_kind=other.kind if other is not None else None,
                other_label=_label_of(other) if other is not None else None,
            )
        )
    return out


# ---- endpoints --------------------------------------------------------------


@router.get(
    "/graph", response_model=GraphResponse, dependencies=[Depends(scoped_read)]
)
async def lineage_graph(
    request: Request,
    entity_id: str | None = Query(None, min_length=1, max_length=_MAX_ID_LEN),
    run_id: str | None = Query(None, min_length=1, max_length=_MAX_ID_LEN),
    depth: int = Query(DEFAULT_GRAPH_DEPTH, ge=1, le=MAX_GRAPH_DEPTH),
) -> GraphResponse:
    """A bounded provenance subgraph around an entity or a run.

    Provide exactly one of ``entity_id`` (record id / stable id / thread id)
    or ``run_id``. The walk is breadth-first, ``depth`` hops, and the response
    is capped at :data:`MAX_GRAPH_NODES` nodes (``truncated`` says when either
    bound cut the story short).
    """
    if (entity_id is None) == (run_id is None):
        raise HTTPException(
            status_code=400,
            detail="provide exactly one of entity_id or run_id",
        )
    registry = _registry(request)
    if run_id is not None:
        graph = await _graph_for_run(request, registry, run_id, depth)
    else:
        assert entity_id is not None  # narrowed by the exactly-one guard above
        root = await _resolve_entity(registry, entity_id)
        if root is None:
            raise HTTPException(
                status_code=404,
                detail=f"no entity found for id {entity_id!r}",
            )
        # Tenant-gate the traced subgraph (404 cross-tenant / unanchored for a
        # tenant-bound caller; NO-OP for an unrestricted principal).
        graph = await _authorize_entity_subgraph(
            request, registry, root.record_id, depth
        )
    if not graph.nodes:
        raise HTTPException(
            status_code=404, detail="the traced entity has no lineage records"
        )
    return _shape_graph(graph)


@router.get(
    "/entity/{record_id}",
    response_model=EntityDetail,
    dependencies=[Depends(scoped_read)],
)
async def lineage_entity(
    request: Request,
    record_id: str = Path(min_length=1, max_length=_MAX_ID_LEN),
) -> EntityDetail:
    """The full record behind a graph node (payload, metadata, incident links).

    Tenant scoping (red-team scope-r5): a tenant-bound caller may only read a record whose
    OWNING workspace (its ``chat_thread`` hub) is in its allow-list — the record is traced
    and run through :func:`_authorize_entity_subgraph`, so a foreign-tenant (or
    workspace-unanchored) record is a clean 404, never a payload leak. An unrestricted
    principal takes the legacy path byte-unchanged.
    """
    registry = _registry(request)
    record = await _resolve_entity(registry, record_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"no entity found for id {record_id!r}"
        )
    # Gate the resolved record by its owning workspace (NO-OP for an unrestricted principal).
    await _authorize_entity_subgraph(
        request, registry, record.record_id, MAX_GRAPH_DEPTH
    )
    links_in = await _maybe_await(registry.links_to(record.record_id))
    links_out = await _maybe_await(registry.links_from(record.record_id))
    return EntityDetail(
        id=record.record_id,
        stable_id=record.stable_id,
        kind=record.kind,
        version=record.version,
        created_at=record.created_at,
        label=_label_of(record),
        payload=_bounded(record.payload),
        metadata=_bounded(record.metadata),
        links_in=await _link_summaries(
            registry, list(links_in), other_side="from_record_id"
        ),
        links_out=await _link_summaries(
            registry, list(links_out), other_side="to_record_id"
        ),
    )


__all__ = ["router"]
