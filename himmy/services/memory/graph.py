"""Graph memory: the relational READ layer (bounded multi-hop traversal).

The store faithfully *persists* a typed graph of memories (``relates_to`` /
``caused_by`` / ``contradicts`` / ``supersedes`` …) via :class:`MemoryLink`, but
writing is only half of relational recall — "pull in the facts related to this hit"
needs a *read* side. This module supplies a deterministic, bounded breadth-first
walk over those links.

It mirrors the entities lineage traversal (:meth:`EntityRegistry.trace`) but is kept
*memory-local*: it deliberately does NOT import :mod:`himmy.entities.lineage`, because
memory links are synchronous, store-local, and carry their own relation vocabulary,
whereas the entities graph is async-projected onto the content-addressed spine. The
walk is *pure*: it takes plain callables (neighbour lookup + endpoint acceptance) and
applies no I/O policy of its own — the caller (the memory service) decides bi-temporal
validity, tier and subject scoping and passes that in as ``accept``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

from himmy.services.memory.store import MemoryGraph, MemoryLink, MemoryRecord

#: Default hop budget for a memory graph walk. Memory neighbourhoods are typically
#: shallow (a hit plus its directly-related facts), so a small default covers the
#: common case while bounding a pathological walk over a densely-linked store.
DEFAULT_TRAVERSAL_DEPTH = 2


def traverse_memory_graph(
    seed_memory_id: str,
    *,
    neighbors: Callable[[str], list[MemoryLink]],
    get: Callable[[str], MemoryRecord | None],
    accept: Callable[[MemoryRecord], bool],
    max_depth: int = DEFAULT_TRAVERSAL_DEPTH,
    relations: Collection[str] | None = None,
) -> MemoryGraph:
    """Walk the memory link graph from ``seed_memory_id`` and return the subgraph.

    Breadth-first up to ``max_depth`` hops. For each node, ``neighbors(memory_id)``
    yields the incident :class:`MemoryLink`s (both directions); ``get(memory_id)``
    resolves a memory record; ``accept(record)`` is the endpoint filter the caller
    supplies (e.g. bi-temporal ``as_of`` validity, tier and subject scoping). An
    endpoint that fails ``accept`` is *not added as a node and not expanded* — its
    edges are pruned — so traversal honours endpoint validity. ``relations`` (optional)
    restricts the walk to those edge relations.

    The returned :class:`MemoryGraph` carries the records reached (``nodes``) and the
    links traversed between accepted endpoints (``edges``); ``truncated`` is set when
    the depth budget stopped the walk while an accepted, reachable, unexpanded node
    still had an edge to an unvisited accepted endpoint — so the caller can tell "this
    is the whole story" apart from "there is more, deeper".

    The walk is deterministic: at each level neighbours are visited in link order, and
    the seed is only included when it itself passes ``accept``.
    """
    rel_set = set(relations) if relations is not None else None
    nodes: dict[str, MemoryRecord] = {}
    edges: list[MemoryLink] = []
    seen_edges: set[str] = set()
    # ``visited`` tracks ids we have decided about (accepted or rejected) so we never
    # re-expand; ``accepted`` is the subset that passed ``accept`` (expandable nodes).
    visited: set[str] = {seed_memory_id}

    seed = get(seed_memory_id)
    seed_accepted = seed is not None and accept(seed)
    if seed_accepted and seed is not None:
        nodes[seed_memory_id] = seed

    # If the seed itself is invalid/out-of-scope there is nothing to expand from.
    if not seed_accepted:
        return MemoryGraph(
            root_memory_id=seed_memory_id, nodes=nodes, edges=edges, truncated=False
        )

    frontier = [seed_memory_id]
    for _ in range(max(0, max_depth)):
        if not frontier:
            break
        next_frontier: list[str] = []
        for mid in frontier:
            for link in _incident(mid, neighbors, rel_set):
                other = _other_endpoint(link, mid)
                # Resolve + accept-test the other endpoint once; reject prunes the
                # edge (an edge to an invalid/out-of-scope fact is not surfaced).
                if other not in visited:
                    visited.add(other)
                    record = get(other)
                    if record is not None and accept(record):
                        nodes[other] = record
                        next_frontier.append(other)
                # Record the edge only when both endpoints are accepted nodes.
                if other in nodes and link.link_id not in seen_edges:
                    seen_edges.add(link.link_id)
                    edges.append(link)
        frontier = next_frontier

    truncated = _is_truncated(frontier, neighbors, get, accept, visited, rel_set)
    return MemoryGraph(
        root_memory_id=seed_memory_id,
        nodes=nodes,
        edges=edges,
        truncated=truncated,
    )


def _incident(
    memory_id: str,
    neighbors: Callable[[str], list[MemoryLink]],
    rel_set: set[str] | None,
) -> list[MemoryLink]:
    """Return the relation-filtered links incident to ``memory_id``."""
    links = neighbors(memory_id)
    if rel_set is None:
        return links
    return [link for link in links if link.relation in rel_set]


def _other_endpoint(link: MemoryLink, memory_id: str) -> str:
    """Return the endpoint of ``link`` that is not ``memory_id``."""
    return (
        link.to_memory_id if link.from_memory_id == memory_id else link.from_memory_id
    )


def _is_truncated(
    frontier: list[str],
    neighbors: Callable[[str], list[MemoryLink]],
    get: Callable[[str], MemoryRecord | None],
    accept: Callable[[MemoryRecord], bool],
    visited: set[str],
    rel_set: set[str] | None,
) -> bool:
    """True iff a leftover frontier node still reaches an unvisited accepted endpoint.

    A pruned (rejected) neighbour does not count as "more story" — only an accepted,
    reachable endpoint we never got to expand marks the walk as truncated.
    """
    for mid in frontier:
        for link in _incident(mid, neighbors, rel_set):
            other = _other_endpoint(link, mid)
            if other in visited:
                continue
            record = get(other)
            if record is not None and accept(record):
                return True
    return False


__all__ = ["DEFAULT_TRAVERSAL_DEPTH", "traverse_memory_graph"]
