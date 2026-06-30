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
unstamped record can never leak.

Subject scoping (red-team scope-r7). Lineage payloads expose a run's persona, prompt text
and context/evidence snapshot, so — like the ``/v1`` ``/runs/{id}/lineage`` reader — both
endpoints ALSO enforce the SUBJECT (BOLA) axis for a ``subject_scoped`` caller: the run path
gates the run's ``subject_id`` (:func:`_run_subject_blocked`, the Studio twin of
``runs._bola_blocked``), the entity-detail path gates the TARGET record's OWN subject (NOT
merely the traced subgraph's anchor — closing the cross-subject read a shared persona/prompt
edge could otherwise bridge), and the per-node/per-link filters drop any foreign-subject node
a wide trace pulled in. Every check is a strict NO-OP for an unrestricted (``all_tenants`` /
offline) / non-``subject_scoped`` principal (filters ``None``), so the single-box path is
byte-unchanged.
"""

from __future__ import annotations

import inspect
import re
from typing import Any

from fastapi import Depends, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from himmy.api.auth import (
    authorize_object,
    get_principal,
    resolve_workspace,
    scoped_read,
    studio_subject_filter,
    studio_tenant_filter,
)
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


async def _run_subject_blocked(
    request: Request, run_app: Any, run_id: str, workspace_id: str | None
) -> bool:
    """Whether subject-axis (BOLA) narrowing hides this run's lineage from the caller.

    The Studio twin of the ``/v1`` :func:`himmy.api.routers.runs._bola_blocked` gate. Returns
    True ONLY for an opt-in ``subject_scoped`` principal tracing a run attributed to ANOTHER
    data subject — so the lineage route renders a clean 404 instead of disclosing the run's
    persona/prompt/context payloads. A strict NO-OP (short-circuits before any run lookup) for
    every other caller — offline / ``all_tenants`` / ``tenant_admin`` / non-``subject_scoped``
    — so the legacy behavior is byte-unchanged.
    """
    if not get_principal(request).subject_scoped:
        return False
    run = await _maybe_await(run_app.get_run(run_id, workspace_id=workspace_id))
    if run is None:
        return False  # unknown / out-of-workspace handled by the route's own lookup
    return not authorize_object(request, getattr(run, "subject_id", None))


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

    Subject scoping (red-team scope-r7): the run lineage exposes the run's persona, prompt
    text and context/evidence snapshot, so — like the ``/v1`` ``/runs/{id}/lineage`` reader
    (which applies :func:`_bola_blocked`) — a ``subject_scoped`` caller may trace ONLY a run
    attributed to its OWN data subject. A run owned by another subject WITHIN the same tenant
    is a clean 404 (existence never leaked). A strict NO-OP for every other principal
    (offline / ``all_tenants`` / ``tenant_admin`` / non-``subject_scoped``).
    """
    workspace_id = resolve_workspace(request, None)
    container = getattr(request.app.state, "container", None)
    run_app = getattr(container, "run_app", None)
    if run_app is not None:
        if await _run_subject_blocked(request, run_app, run_id, workspace_id):
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
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
    # Subject-axis gate on the run-store fallback too (scope-r7): a ``subject_scoped`` caller
    # may trace only its OWN subject's run (404 cross-subject). NO-OP for every other caller.
    if not authorize_object(request, getattr(run, "subject_id", None)):
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


def _record_subject(record: EntityRecord) -> str | None:
    """The data subject a record is attributed to (metadata first, payload fallback).

    Used to apply the subject (BOLA) axis to the by-id entity-detail read: a
    ``subject_scoped`` caller may only read a record attributed to its OWN subject. ``None``
    (a subject-less / legacy record) is allowed by :func:`authorize_object`.
    """
    subject = record.metadata.get("subject_id")
    if isinstance(subject, str) and subject:
        return subject
    payload_subject = record.payload.get("subject_id")
    if isinstance(payload_subject, str) and payload_subject:
        return payload_subject
    payload_meta = record.payload.get("metadata")
    if isinstance(payload_meta, dict):
        nested = payload_meta.get("subject_id")
        if isinstance(nested, str) and nested:
            return nested
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


def _node_visible(
    record: EntityRecord,
    tenants: frozenset[str] | None,
    subject: str | None = None,
) -> bool:
    """Whether a single graph node may be disclosed to the caller (per-NODE scope-r6/r7).

    The subgraph-level gate (:func:`_authorize_entity_subgraph`) authorizes the WHOLE
    traced neighborhood on one matching anchor; but a cross-tenant/cross-subject edge can
    pull a foreign record into that neighborhood, whose label/kind/payload would then leak.
    This per-node check closes that on BOTH axes:

    * **Tenant.** For a tenant-bound caller (``tenants`` is a set) a node is visible only
      when its OWN owning workspace is in the allow-list — a node carrying NO workspace stamp
      is allowed (it predates tenant binding / belongs to no tenant, matching the by-id
      ``authorize_studio_object`` ``None``-is-allowed convention), and any node stamped to a
      workspace OUTSIDE the allow-list is dropped.
    * **Subject (scope-r7).** For a ``subject_scoped`` caller (``subject`` is its own
      subject) a node attributed to ANOTHER subject is dropped — a node with NO recorded
      subject is allowed (``None``-is-allowed, mirroring ``authorize_object``).

    For an unrestricted principal (``tenants is None`` and ``subject is None``) every node is
    visible — byte-unchanged.
    """
    if tenants is not None:
        ws = _record_workspace(record)
        if ws is not None and ws not in tenants:
            return False
    if subject is not None:
        rec_subject = _record_subject(record)
        if rec_subject is not None and rec_subject != subject:
            return False
    return True


def _shape_graph(
    graph: LineageGraph,
    tenants: frozenset[str] | None = None,
    subject: str | None = None,
) -> GraphResponse:
    """Project a traced graph into the wire shape, node-capped in BFS order.

    ``tenants`` (red-team scope-r6) is the caller's :func:`studio_tenant_filter` allow-list
    and ``subject`` (scope-r7) its :func:`studio_subject_filter`: both ``None`` (unrestricted
    / offline) keeps every node — byte-unchanged — while a set/value drops any node stamped to
    a workspace outside the allow-list OR attributed to another subject, so a
    cross-tenant/cross-subject edge cannot disclose a foreign record's kind/label even though
    the subgraph as a whole was anchor-authorized.
    """
    ids = [
        rid
        for rid, rec in graph.nodes.items()
        if _node_visible(rec, tenants, subject)
    ]
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
    registry: Any,
    links: list[Any],
    *,
    other_side: str,
    tenants: frozenset[str] | None = None,
    subject: str | None = None,
) -> list[LinkSummary]:
    """Summarise incident links, resolving the far record for display.

    ``tenants`` (red-team scope-r6) is the caller's :func:`studio_tenant_filter` allow-list.
    When set (a tenant-bound caller), a far-side record stamped to a workspace OUTSIDE the
    allow-list has its kind/label WITHHELD (the link is still listed by id + relation, but
    the foreign record's content is not disclosed) — closing the cross-tenant edge that let
    a caller read the label of a record it was never authorized for. ``None`` (unrestricted /
    offline) discloses every far record — byte-unchanged.
    """
    out: list[LinkSummary] = []
    for link in links[:_MAX_DETAIL_LINKS]:
        other_id = getattr(link, other_side)
        other = await _maybe_await(registry.get(other_id))
        disclose = other is not None and _node_visible(other, tenants, subject)
        out.append(
            LinkSummary(
                relation=link.relation,
                other_id=other_id,
                other_kind=other.kind if disclose else None,
                other_label=_label_of(other) if disclose else None,
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
    # Per-node tenant + subject scope (scope-r6/r7): drop any cross-tenant or cross-subject
    # node a wide trace pulled in, so a cross-tenant/cross-subject edge cannot disclose a
    # foreign record's kind/label. NO-OP offline / ``all_tenants`` / non-``subject_scoped``.
    return _shape_graph(
        graph, studio_tenant_filter(request), studio_subject_filter(request)
    )


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

    Anchor-confusion fix (red-team scope-r7): the subgraph anchor check authorizes the whole
    traced neighborhood on ANY in-tenant ``chat_thread`` hub, which a SHARED node (a reused
    persona/prompt/context snapshot bridging the caller's own thread into a FOREIGN record's
    6-hop neighborhood) can satisfy — so it is necessary but NOT sufficient. The disclosed
    TARGET record must ALSO pass the per-node gate on its OWN owning workspace
    (:func:`_node_visible`) AND, for a ``subject_scoped`` caller, its OWN subject
    (:func:`authorize_object`), else a clean 404 — closing the cross-tenant/cross-subject
    read via a shared-entity edge.
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
    # The subgraph anchor check above is satisfied by ANY in-tenant thread in the
    # neighborhood; the disclosed TARGET record must itself be visible by its OWN workspace
    # and subject, else a shared-entity edge would leak a foreign record's full payload.
    if not _node_visible(record, studio_tenant_filter(request)) or not authorize_object(
        request, _record_subject(record)
    ):
        raise HTTPException(
            status_code=404, detail=f"no entity found for id {record_id!r}"
        )
    links_in = await _maybe_await(registry.links_to(record.record_id))
    links_out = await _maybe_await(registry.links_from(record.record_id))
    # Per-link tenant scope (scope-r6): withhold a far-side record's kind/label when it is
    # stamped to a workspace outside the caller's allow-list, so a cross-tenant edge can't
    # disclose a foreign record's label. NO-OP for an unrestricted principal (filter None).
    tenants = studio_tenant_filter(request)
    subject = studio_subject_filter(request)
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
            registry,
            list(links_in),
            other_side="from_record_id",
            tenants=tenants,
            subject=subject,
        ),
        links_out=await _link_summaries(
            registry,
            list(links_out),
            other_side="to_record_id",
            tenants=tenants,
            subject=subject,
        ),
    )


__all__ = ["router"]
