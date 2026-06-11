"""Lineage / provenance for the ``himmy`` CLI: trace a run back to what produced it.

``himmy lineage <run_id>`` (or the latest run when omitted) reads the persisted
event trace (``.himmy/trace.db`` via :class:`SqliteEventStore`) and renders an
indented provenance tree — run → inferences → tool calls → results — in the desk
palette (gold tools, green results, faint timing). ``--dot`` emits the same tree
as Graphviz DOT for piping into ``dot``/Studio.

This mirrors :func:`himmy.api.studio_lineage.run_lineage`: a Studio run isn't in
the entity-registry lineage graph, but its full cognition trace *is* persisted, so
we reshape that ordered trace into a provenance view ("why it said that"). Here the
source is the durable trace log a ``--trace`` run wrote, keyed by ``thread_id``
(the same id :func:`himmy.cli.commands.cmd_trace` lists and shows).

Everything renders to **stdout** so it can be piped; styling collapses to plain
text on non-TTY/pipes/CI (see :func:`himmy.cli.ui.styles`). DOT is always plain.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from himmy.cli.commands import _eprint, _trace_db
from himmy.cli.ui import styles


class _LineageNode:
    """One node in the provenance tree: a label, a style colour, and children."""

    __slots__ = ("label", "color", "detail", "children")

    def __init__(self, label: str, *, color: str = "ink", detail: str = "") -> None:
        self.label = label
        self.color = color
        self.detail = detail
        self.children: list[_LineageNode] = []

    def add(self, child: _LineageNode) -> _LineageNode:
        self.children.append(child)
        return child


def _payload(event: Any) -> dict[str, Any]:
    """The event payload as a plain dict (events may carry ``None``)."""
    return getattr(event, "payload", None) or {}


def _event_type(event: Any) -> str:
    """The event type as a bare string, tolerating enum or str."""
    et = getattr(event, "event_type", "")
    return getattr(et, "value", et)


def _short(value: Any, *, limit: int = 60) -> str:
    """A compact one-line rendering of a value, truncated for tree display."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_lineage_tree(events: list[Any], run_id: str) -> _LineageNode:
    """Reshape an ordered event list into a provenance tree rooted at the run.

    The shape mirrors :func:`himmy.api.studio_lineage.run_lineage`: the run is the
    root; inferences and tool calls hang under it in trace order; a tool's result
    (``TOOL_COMPLETED``/``TOOL_FAILED``) is attached as a child of its call so the
    "call → result" provenance edge is visible. Events with no thread are ignored
    by the caller (which filters by ``thread_id``); here we just order by timestamp.
    """
    ordered = sorted(events, key=lambda e: getattr(e, "timestamp", ""))
    # Surface the model + tool/cost summary on the root, like the trace footer.
    models: list[str] = []
    tool_count = 0
    total_cost = 0.0
    for e in ordered:
        p = _payload(e)
        mk = p.get("model_key")
        if mk and mk not in models:
            models.append(str(mk))
        if _event_type(e) == "TOOL_CALLED":
            tool_count += 1
        total_cost += getattr(e, "cost", None) or 0.0

    summary_bits = []
    if models:
        summary_bits.append(" · ".join(models))
    summary_bits.append(f"{tool_count} tool call(s)")
    summary_bits.append(f"${total_cost:.4f}")
    root = _LineageNode(f"run {run_id}", color="snow", detail="  ".join(summary_bits))

    # Map a tool_call_id → its call node so the paired result attaches under it.
    calls_by_id: dict[str, _LineageNode] = {}
    for e in ordered:
        et = _event_type(e)
        p = _payload(e)
        if et == "INFERENCE_REQUESTED":
            mk = p.get("model_key", "")
            root.add(
                _LineageNode("inference", color="ink", detail=f"→ {mk}" if mk else "")
            )
        elif et == "INFERENCE_SUCCEEDED":
            it, ot = p.get("input_tokens"), p.get("output_tokens")
            ms = getattr(e, "latency_ms", None)
            bits = []
            if ms:
                bits.append(f"{ms:.0f}ms")
            if it is not None or ot is not None:
                bits.append(f"{it or 0}→{ot or 0} tok")
            cost = getattr(e, "cost", None)
            if cost:
                bits.append(f"${cost:.4f}")
            root.add(
                _LineageNode("inference ok", color="green", detail=" · ".join(bits))
            )
        elif et == "INFERENCE_FAILED":
            root.add(
                _LineageNode(
                    "inference FAILED",
                    color="crimson",
                    detail=_short(getattr(e, "error", "")),
                )
            )
        elif et == "TOOL_CALLED":
            name = p.get("tool_name", "tool")
            # TOOL_CALLED payloads carry ``args``; tolerate ``tool_args`` too.
            args = p.get("args", p.get("tool_args"))
            node = root.add(
                _LineageNode(f"tool {name}", color="gold", detail=_short(args))
            )
            call_id = getattr(e, "tool_call_id", None) or p.get("tool_call_id")
            if call_id:
                calls_by_id[call_id] = node
        elif et == "TOOL_COMPLETED":
            name = p.get("tool_name", "tool")
            outcome = p.get("tool_outcome", "success")
            ms = getattr(e, "latency_ms", None)
            detail = outcome + (f" · {ms:.0f}ms" if ms else "")
            leaf = _LineageNode("result", color="green", detail=detail)
            _attach_result(e, p, leaf, calls_by_id, root)
        elif et == "TOOL_FAILED":
            err = getattr(e, "error", "") or p.get("error", "")
            leaf = _LineageNode("result FAILED", color="crimson", detail=_short(err))
            _attach_result(e, p, leaf, calls_by_id, root)
    return root


def _attach_result(
    event: Any,
    payload: dict[str, Any],
    leaf: _LineageNode,
    calls_by_id: dict[str, _LineageNode],
    root: _LineageNode,
) -> None:
    """Attach a tool result under its matching call, or under the run as a fallback."""
    call_id = getattr(event, "tool_call_id", None) or payload.get("tool_call_id")
    parent = calls_by_id.get(call_id) if call_id else None
    (parent or root).add(leaf)


def render_tree(root: _LineageNode, c: dict[str, str]) -> str:
    """Render a provenance tree as indented ASCII-art with the desk palette."""
    lines: list[str] = []
    reset = c["reset"]
    head = f"{c[root.color]}{root.label}{reset}"
    if root.detail:
        head += f"  {c['faint']}{root.detail}{reset}"
    lines.append(head)
    kids = root.children
    for i, child in enumerate(kids):
        _render_child(child, "", i == len(kids) - 1, c, lines)
    return "\n".join(lines)


def _render_child(
    node: _LineageNode,
    prefix: str,
    last: bool,
    c: dict[str, str],
    lines: list[str],
) -> None:
    reset = c["reset"]
    connector = "└─ " if last else "├─ "
    body = f"{c[node.color]}{node.label}{reset}"
    if node.detail:
        body += f"  {c['faint']}{node.detail}{reset}"
    lines.append(f"{c['faint']}{prefix}{connector}{reset}{body}")
    child_prefix = prefix + ("   " if last else "│  ")
    for i, kid in enumerate(node.children):
        _render_child(kid, child_prefix, i == len(node.children) - 1, c, lines)


def lineage_to_dot(root: _LineageNode, run_id: str) -> str:
    """Render the provenance tree as Graphviz DOT (one node per step, edges = order).

    Falls back to formatting the trace tree directly (per the task): the trace isn't
    an entity ``LineageGraph``, so we emit an equivalent provenance digraph by hand,
    matching the look of :meth:`LineageGraph.to_dot` (rankdir=LR, box nodes, root
    highlighted).
    """
    lines = ["digraph lineage {", "  rankdir=LR;", "  node [shape=box];"]
    counter = [0]
    ids: dict[int, str] = {}

    def _walk(node: _LineageNode, parent_id: str | None) -> None:
        nid = f"n{counter[0]}"
        counter[0] += 1
        ids[id(node)] = nid
        label = node.label.replace('"', "'")
        if node.detail:
            label += "\\n" + _short(node.detail, limit=40).replace('"', "'")
        marker = ", style=filled, fillcolor=lightyellow" if parent_id is None else ""
        lines.append(f'  "{nid}" [label="{label}"{marker}];')
        if parent_id is not None:
            lines.append(f'  "{parent_id}" -> "{nid}";')
        for kid in node.children:
            _walk(kid, nid)

    _walk(root, None)
    lines.append("}")
    return "\n".join(lines)


def latest_run_id(store: Any) -> str | None:
    """The thread_id of the most recently traced run, or None when none exist."""
    runs = store.recent_threads(limit=1)
    return runs[0]["thread_id"] if runs else None


def render_run_lineage(
    events: list[Any], run_id: str, *, as_dot: bool, c: dict[str, str]
) -> str:
    """Build + render one run's provenance, as a styled tree or as DOT."""
    tree = build_lineage_tree(events, run_id)
    if as_dot:
        return lineage_to_dot(tree, run_id)
    return render_tree(tree, c)


def cmd_lineage(args: argparse.Namespace) -> int:
    """``himmy lineage [run_id]``: provenance tree of a run (or the latest), --dot to DOT.

    Reads the durable trace log written by ``himmy run --trace`` / chat. With no
    ``run_id`` the most recent traced run is used. ``--dot`` emits Graphviz instead
    of the indented tree. Exit 1 when there are no traces or the run is unknown.
    """
    from himmy.services.observability.trace import SqliteEventStore

    db = _trace_db()
    if not Path(db).exists():
        _eprint("no traces yet — run with `himmy run --trace ...` first")
        return 1

    store = SqliteEventStore(db)
    try:
        run_id = getattr(args, "run_id", None)
        if not run_id:
            run_id = latest_run_id(store)
            if not run_id:
                _eprint("no traced runs found")
                return 1
        events = store.list_events(thread_id=run_id)
        if not events:
            _eprint(
                f"no events for run {run_id!r} — list runs with `himmy lineage` "
                f"(no id) or `himmy trace`"
            )
            return 1
        as_dot = bool(getattr(args, "dot", False))
        c = styles(sys.stdout)
        print(render_run_lineage(events, run_id, as_dot=as_dot, c=c))
        return 0
    finally:
        store.close()


__all__ = [
    "cmd_lineage",
    "build_lineage_tree",
    "render_tree",
    "lineage_to_dot",
    "render_run_lineage",
    "latest_run_id",
]
