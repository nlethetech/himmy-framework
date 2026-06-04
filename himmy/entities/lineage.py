"""Entities kernel: the lineage READ layer (reverse + multi-hop traversal).

The registry faithfully *writes* a typed provenance graph (``uses_persona`` /
``in_thread`` / ``built_from`` / ``observed_in_run`` …), but writing is only half
of the headline promise — "trace any recommendation back to its persona +
evidence" needs a *read* side. This module supplies the shared pieces:

* :class:`LineageDirection` — which way to walk edges (out / in / both);
* :class:`LineageGraph` — a typed, JSON- and DOT-projectable subgraph returned by
  ``EntityRegistry.trace`` / ``PostgresEntityRegistry.trace``.

The traversal methods themselves live on each registry (sync in-memory,
``WITH RECURSIVE`` on Postgres) so they can use the most efficient primitive for
their backend; both return one of these graphs.
"""

from __future__ import annotations

from collections.abc import Collection
from enum import Enum

from pydantic import BaseModel, Field

from himmy.entities.records import EntityLink, EntityRecord

#: Default hop budget for :meth:`trace`. The lineage graph around a run is shallow
#: (thread -> persona/prompt/snapshot), so a small default covers it while bounding
#: pathological walks.
DEFAULT_TRACE_DEPTH = 6


class LineageDirection(str, Enum):
    """Which direction to follow edges when traversing the lineage graph.

    ``OUT`` follows edges *from* a node (a thread -> the persona it used);
    ``IN`` follows edges *into* a node (which threads observed this snapshot);
    ``BOTH`` walks the full connected neighbourhood regardless of edge direction.
    """

    OUT = "out"
    IN = "in"
    BOTH = "both"


class LineageGraph(BaseModel):
    """A traced subgraph of the lineage: the records reached and the edges between.

    ``nodes`` maps ``record_id -> EntityRecord`` (the root is included when it
    exists); ``edges`` are the typed links traversed. ``truncated`` is True when a
    ``max_depth`` cutoff stopped the walk while reachable nodes remained — so a
    caller can tell "this is the whole story" apart from "there is more, deeper".
    """

    root_id: str
    nodes: dict[str, EntityRecord] = Field(default_factory=dict)
    edges: list[EntityLink] = Field(default_factory=list)
    truncated: bool = False

    @property
    def node_count(self) -> int:
        """Number of distinct records in the graph."""
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Number of distinct links in the graph."""
        return len(self.edges)

    def record_ids(self) -> set[str]:
        """The set of record ids present as nodes."""
        return set(self.nodes)

    def relations(self) -> set[str]:
        """The distinct relation names present among the edges."""
        return {edge.relation for edge in self.edges}

    def filter_relations(self, relations: Collection[str]) -> LineageGraph:
        """Return a copy keeping only edges whose relation is in ``relations``.

        Nodes that become unreferenced are dropped, except the root is always
        retained so the result still anchors on the queried record.
        """
        keep = set(relations)
        edges = [edge for edge in self.edges if edge.relation in keep]
        referenced = {self.root_id}
        for edge in edges:
            referenced.add(edge.from_record_id)
            referenced.add(edge.to_record_id)
        nodes = {rid: rec for rid, rec in self.nodes.items() if rid in referenced}
        return LineageGraph(
            root_id=self.root_id,
            nodes=nodes,
            edges=edges,
            truncated=self.truncated,
        )

    def to_dot(self) -> str:
        """Render the graph as Graphviz DOT (node = kind/version, edge = relation)."""
        lines = ["digraph lineage {", "  rankdir=LR;", "  node [shape=box];"]
        for rid, rec in self.nodes.items():
            short = rec.stable_id[:8]
            label = f"{rec.kind}\\n{short} v{rec.version}"
            marker = (
                ", style=filled, fillcolor=lightyellow" if rid == self.root_id else ""
            )
            lines.append(f'  "{rid}" [label="{label}"{marker}];')
        for edge in self.edges:
            lines.append(
                f'  "{edge.from_record_id}" -> "{edge.to_record_id}" '
                f'[label="{edge.relation}"];'
            )
        lines.append("}")
        return "\n".join(lines)


__all__ = ["LineageDirection", "LineageGraph", "DEFAULT_TRACE_DEPTH"]
