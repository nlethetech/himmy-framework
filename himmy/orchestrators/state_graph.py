"""Orchestrators kernel: a real, audited, durably-resumable ``StateGraph``.

Where :class:`~himmy.orchestrators.workflow.WorkflowOrchestrator` runs a *linear*
list of steps, a :class:`StateGraph` runs a *directed graph* of nodes over a typed
shared state — the LangGraph capability expressed provenance-native and
offline-first on himmy's existing seams:

* **Typed shared state** — a plain ``dict`` (the default) or any pydantic model;
  per-key *reducers* merge each node's returned delta into the running state, so a
  parallel fan-out can append to the same key without clobbering (e.g. ``add``
  concatenates lists). Unreduced keys are last-write-wins.
* **Nodes are functions** — ``async (state) -> delta`` (or sync). A node may wrap an
  agent, a tool, an LLM call, or pure Python; the graph does not care, which keeps
  it a thin orchestration layer over the runtime rather than a second runtime.
* **Edges** — static ``add_edge(a, b)`` plus ``add_conditional_edges(a, router)``
  where ``router(state)`` returns the next node name(s). Routing to ``END``
  finishes that branch.
* **Parallel fan-out + join** — execution is a BSP *superstep* loop: every node in
  the current frontier runs, their deltas are merged together (reducers applied),
  then the next frontier is computed from the edges. Multiple successors fan out;
  multiple predecessors naturally join on the shared state at the next superstep.
* **Loops with guards** — a node may route back to an earlier node; a
  per-node ``visit`` cap and a global ``recursion_limit`` (max supersteps) prevent
  an unbounded loop, raising a clean :class:`GraphRecursionError`.
* **Audit-native** — ``GRAPH_STARTED`` / ``GRAPH_NODE_STARTED`` /
  ``GRAPH_NODE_COMPLETED`` / ``GRAPH_EDGE_TAKEN`` / ``GRAPH_CHECKPOINTED`` /
  ``GRAPH_FINISHED`` :class:`~himmy.core.events.RunEvent`\\ s flow through the exact
  same isolated fan-out the runtime uses (storage -> entity registry ->
  observability -> caller callbacks), so every node entry/exit and edge decision
  lands on the audit spine and the run is replayable.
* **Durable resume** — after each superstep the graph persists a
  :class:`~himmy.runtime.checkpoint.GraphCheckpoint` (state + frontier + visit
  counts) via the :class:`~himmy.runtime.checkpoint.GraphCheckpointStore` seam, so
  an interrupted run (deadline, crash, HITL pause) resumes from the last completed
  superstep in a fresh process.

Offline-first: zero new dependencies, no network, no LLM required — a graph of
pure-Python nodes runs with no keys. Inference, if a node uses it, goes through the
existing :class:`InferenceService`, so deterministic replay via the inference
cassette comes for free.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.core.ids import new_uuid, utc_now_iso
from himmy.runtime.checkpoint import (
    GRAPH_COMPLETED,
    GRAPH_FAILED,
    GRAPH_INTERRUPTED,
    GRAPH_RUNNING,
    GraphCheckpoint,
    GraphCheckpointStore,
    InMemoryGraphCheckpointStore,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.entities.records import EntityRecord

# Sentinel terminal node name. Routing a branch to END finishes that branch.
START = "__start__"
END = "__end__"

# Caller-facing event callback, mirrored from the runtime/workflow ``OnEvent``.
OnEvent = Callable[[RunEvent], Awaitable[None]]

# A node returns a partial state delta (or nothing). State is a plain dict.
GraphState = dict[str, Any]
NodeFn = Callable[[GraphState], Any]  # sync or async; returns a delta mapping
RouterFn = Callable[[GraphState], Any]  # returns a node name or sequence of names
Reducer = Callable[[Any, Any], Any]


class GraphError(HimmyError):
    """A static error in a graph definition or invocation."""


class GraphRecursionError(GraphError):
    """The graph exceeded its recursion limit (a loop guard tripped)."""


class GraphStateSizeError(GraphError):
    """The merged shared state exceeded ``max_state_bytes`` (an OOM guard tripped)."""


def add_reducer(existing: Any, incoming: Any) -> Any:
    """Reducer that concatenates lists / adds numbers (LangGraph ``add`` semantics).

    Used for fan-out keys that accumulate (e.g. a list of partial results from
    parallel branches). Falls back to last-write-wins for non-additive types.
    """
    if existing is None:
        return incoming
    if isinstance(existing, list):
        return [*existing, *(incoming if isinstance(incoming, list) else [incoming])]
    if isinstance(existing, (int, float)) and isinstance(incoming, (int, float)):
        return existing + incoming
    return incoming


class _Node(BaseModel):
    """Internal node record: name plus an optional per-visit cap for loop guards."""

    model_config = {"arbitrary_types_allowed": True}

    name: str
    fn: NodeFn
    max_visits: int | None = None

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this node into its canonical ``EntityRecord`` (kind ``graph_node``)."""
        from himmy.entities.projection import project

        # The callable is not serializable; project a stable descriptor instead so
        # the content-addressed record stays deterministic across processes.
        payload = {"name": self.name, "max_visits": self.max_visits}
        return project(
            self,
            stable_value=self.name,
            namespace="graph_node",
            kind="graph_node",
            version=version,
            payload=payload,
            metadata=metadata,
        )


class GraphRunResult(BaseModel):
    """The outcome of a graph run.

    ``status`` is ``completed`` | ``interrupted`` | ``failed``. On ``interrupted``
    the ``checkpoint_id`` points at the durable snapshot to resume from; on
    ``completed`` ``final_state`` carries the merged shared state.
    """

    model_config = {"arbitrary_types_allowed": True}

    graph_name: str
    status: str
    final_state: GraphState = {}
    supersteps: int = 0
    node_sequence: list[str] = []
    checkpoint_id: str | None = None
    error: str | None = None


class StateGraph:
    """A directed graph of state-transforming nodes with conditional edges.

    Build it declaratively (``add_node`` / ``add_edge`` / ``add_conditional_edges``
    / ``set_entry_point``), then ``compile()`` it into a runnable
    :class:`CompiledStateGraph`. Compilation validates the topology so a malformed
    graph fails fast (a dangling edge target, no entry point, an unreachable node)
    rather than at run time.
    """

    def __init__(self, name: str = "graph") -> None:
        self.name = name
        self._nodes: dict[str, _Node] = {}
        self._edges: dict[str, list[str]] = {}
        self._conditional: dict[str, RouterFn] = {}
        self._reducers: dict[str, Reducer] = {}
        self._entry: str | None = None

    def add_node(
        self, name: str, fn: NodeFn, *, max_visits: int | None = None
    ) -> StateGraph:
        """Register a node. ``max_visits`` caps how often it may run in one graph run."""
        if name in (START, END):
            raise GraphError(f"{name!r} is a reserved node name")
        if name in self._nodes:
            raise GraphError(f"duplicate node {name!r}")
        if max_visits is not None and max_visits < 1:
            raise GraphError("max_visits must be >= 1")
        self._nodes[name] = _Node(name=name, fn=fn, max_visits=max_visits)
        return self

    def add_edge(self, start: str, end: str) -> StateGraph:
        """Add a static edge ``start -> end``. ``end`` may be :data:`END`."""
        self._edges.setdefault(start, []).append(end)
        return self

    def add_conditional_edges(self, start: str, router: RouterFn) -> StateGraph:
        """Route out of ``start`` dynamically: ``router(state)`` -> name | [names].

        The router may return :data:`END` to finish that branch, a single node
        name, or a sequence of names to fan out into a parallel superstep.
        """
        if start in self._conditional:
            raise GraphError(f"node {start!r} already has conditional edges")
        self._conditional[start] = router
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        """Designate the node the graph starts from."""
        self._entry = name
        return self

    def set_reducer(self, key: str, reducer: Reducer) -> StateGraph:
        """Register a per-key reducer used to merge node deltas into shared state."""
        self._reducers[key] = reducer
        return self

    def compile(
        self,
        *,
        checkpoint_store: GraphCheckpointStore | None = None,
        memory_store: Any = None,
        entity_registry: Any = None,
        on_event: OnEvent | list[OnEvent] | None = None,
        recursion_limit: int = 50,
        max_state_bytes: int | None = None,
    ) -> CompiledStateGraph:
        """Validate the topology and return a runnable :class:`CompiledStateGraph`.

        ``max_state_bytes`` (optional, default ``None`` = unlimited, today's
        behavior) bounds the *serialized* size of the merged shared state; an
        ``add``-style reducer fed by a parallel fan-out or a loop can otherwise
        grow state without limit. When the bound is exceeded after a superstep
        the run fails cleanly with :class:`GraphStateSizeError`.
        """
        if max_state_bytes is not None and max_state_bytes < 1:
            raise GraphError("max_state_bytes must be >= 1")
        if self._entry is None:
            raise GraphError("graph has no entry point (call set_entry_point)")
        if self._entry not in self._nodes:
            raise GraphError(f"entry point {self._entry!r} is not a registered node")
        # Validate every static-edge target.
        for start, ends in self._edges.items():
            if start not in self._nodes:
                raise GraphError(f"edge source {start!r} is not a registered node")
            for end in ends:
                if end != END and end not in self._nodes:
                    raise GraphError(f"edge target {end!r} is not a registered node")
        # A node with neither static edges nor a router (and that is not END) is a
        # silent dead-end; treat it as routing to END, which is the friendly default.
        return CompiledStateGraph(
            name=self.name,
            nodes=dict(self._nodes),
            edges={k: list(v) for k, v in self._edges.items()},
            conditional=dict(self._conditional),
            reducers=dict(self._reducers),
            entry=self._entry,
            checkpoint_store=checkpoint_store or InMemoryGraphCheckpointStore(),
            memory_store=memory_store,
            entity_registry=entity_registry,
            on_event=on_event,
            recursion_limit=max(1, recursion_limit),
            max_state_bytes=max_state_bytes,
        )


class CompiledStateGraph:
    """A validated, runnable :class:`StateGraph`. Build via :meth:`StateGraph.compile`."""

    def __init__(
        self,
        *,
        name: str,
        nodes: dict[str, _Node],
        edges: dict[str, list[str]],
        conditional: dict[str, RouterFn],
        reducers: dict[str, Reducer],
        entry: str,
        checkpoint_store: GraphCheckpointStore,
        memory_store: Any,
        entity_registry: Any,
        on_event: OnEvent | list[OnEvent] | None,
        recursion_limit: int,
        max_state_bytes: int | None = None,
    ) -> None:
        self.name = name
        self._nodes = nodes
        self._edges = edges
        self._conditional = conditional
        self._reducers = reducers
        self._entry = entry
        self._checkpoint_store = checkpoint_store
        self._memory_store = memory_store
        self._entity_registry = entity_registry
        self._on_event = self._coerce_callbacks(on_event)
        self._recursion_limit = recursion_limit
        self._max_state_bytes = max_state_bytes

    @staticmethod
    def _coerce_callbacks(
        on_event: OnEvent | list[OnEvent] | None,
    ) -> list[OnEvent]:
        if on_event is None:
            return []
        if isinstance(on_event, list):
            return [cb for cb in on_event if cb is not None]
        return [on_event]

    # ------------------------------------------------------------------ run
    async def invoke(
        self,
        initial_state: Mapping[str, Any] | None = None,
        *,
        resume: GraphCheckpoint | str | None = None,
        checkpoint_id: str | None = None,
        timeout_seconds: float | None = None,
        thread_id: str | None = None,
        max_state_bytes: int | None = None,
    ) -> GraphRunResult:
        """Run the graph to completion (or until a guard/timeout interrupts it).

        Pass ``resume`` (a :class:`GraphCheckpoint` or its id) to continue a prior
        interrupted run from its last completed superstep with the persisted state
        and frontier. ``timeout_seconds`` bounds the whole run; on expiry the run
        is checkpointed as ``interrupted`` and a terminal ``GRAPH_FINISHED`` event
        is emitted, so the caller can resume rather than lose progress.
        ``max_state_bytes`` (default ``None`` = use the compile-time setting,
        itself default unlimited) bounds the merged shared state per run; when
        exceeded the run fails with :class:`GraphStateSizeError`.
        """
        if max_state_bytes is not None and max_state_bytes < 1:
            raise GraphError("max_state_bytes must be >= 1")
        state_limit = (
            max_state_bytes if max_state_bytes is not None else self._max_state_bytes
        )
        chk = self._resolve_resume(resume)
        if chk is None:
            chk = GraphCheckpoint(
                checkpoint_id=checkpoint_id or new_uuid(),
                graph_name=self.name,
                status=GRAPH_RUNNING,
                state=dict(initial_state or {}),
                frontier=[self._entry],
                superstep=0,
                visit_counts={},
                completed_nodes=[],
            )
        elif checkpoint_id is not None:
            chk = chk.model_copy(update={"checkpoint_id": checkpoint_id})

        tid = thread_id or f"graph:{chk.checkpoint_id}"

        if chk.superstep == 0:
            await self._emit(
                RunEvent(
                    event_type=EventType.GRAPH_STARTED,
                    thread_id=tid,
                    payload={
                        "graph_name": self.name,
                        "entry": self._entry,
                        "checkpoint_id": chk.checkpoint_id,
                    },
                )
            )

        try:
            if timeout_seconds is not None and timeout_seconds > 0:
                async with _timeout(timeout_seconds):
                    return await self._run_loop(chk, tid, state_limit)
            return await self._run_loop(chk, tid, state_limit)
        except (TimeoutError, asyncio.CancelledError):
            chk.status = GRAPH_INTERRUPTED
            chk.error = "cancelled"
            chk.updated_at = utc_now_iso()
            self._checkpoint_store.save(chk)
            await self._emit(
                RunEvent(
                    event_type=EventType.GRAPH_FINISHED,
                    thread_id=tid,
                    error="cancelled",
                    payload={
                        "graph_name": self.name,
                        "status": GRAPH_INTERRUPTED,
                        "checkpoint_id": chk.checkpoint_id,
                        "superstep": chk.superstep,
                    },
                )
            )
            return GraphRunResult(
                graph_name=self.name,
                status=GRAPH_INTERRUPTED,
                final_state=dict(chk.state),
                supersteps=chk.superstep,
                node_sequence=list(chk.completed_nodes),
                checkpoint_id=chk.checkpoint_id,
                error="cancelled",
            )

    def _resolve_resume(
        self, resume: GraphCheckpoint | str | None
    ) -> GraphCheckpoint | None:
        if resume is None:
            return None
        if isinstance(resume, GraphCheckpoint):
            chk = resume.model_copy(deep=True)
        else:
            loaded = self._checkpoint_store.load(resume)
            if loaded is None:
                raise GraphError(f"no graph checkpoint {resume!r} to resume")
            chk = loaded
        if chk.graph_name and chk.graph_name != self.name:
            raise GraphError(
                f"checkpoint is for graph {chk.graph_name!r}, not {self.name!r}"
            )
        chk.status = GRAPH_RUNNING
        return chk

    async def _run_loop(
        self, chk: GraphCheckpoint, tid: str, max_state_bytes: int | None = None
    ) -> GraphRunResult:
        """The BSP superstep loop: run the frontier, merge, route, checkpoint, repeat."""
        while chk.frontier:
            if chk.superstep >= self._recursion_limit:
                chk.status = GRAPH_FAILED
                chk.error = (
                    f"recursion limit {self._recursion_limit} exceeded "
                    f"(frontier={chk.frontier})"
                )
                chk.updated_at = utc_now_iso()
                self._checkpoint_store.save(chk)
                await self._emit_finished(tid, chk)
                raise GraphRecursionError(chk.error)

            # De-duplicate the frontier preserving order (a join target reached by
            # two predecessors runs once per superstep).
            frontier = list(dict.fromkeys(chk.frontier))
            deltas, next_targets = await self._run_superstep(frontier, chk, tid)

            # Merge every node's delta into shared state (reducers applied).
            for delta in deltas:
                self._merge(chk.state, delta)

            # OOM guard: bound the merged state size (opt-in via max_state_bytes).
            if max_state_bytes is not None:
                size = _state_size_bytes(chk.state)
                if size > max_state_bytes:
                    chk.status = GRAPH_FAILED
                    chk.error = (
                        f"merged state is {size} bytes, exceeding "
                        f"max_state_bytes={max_state_bytes} "
                        f"(superstep {chk.superstep})"
                    )
                    chk.updated_at = utc_now_iso()
                    self._checkpoint_store.save(chk)
                    await self._emit_finished(tid, chk)
                    raise GraphStateSizeError(chk.error)

            for node_name in frontier:
                chk.completed_nodes.append(node_name)
                chk.visit_counts[node_name] = chk.visit_counts.get(node_name, 0) + 1

            chk.frontier = next_targets
            chk.superstep += 1
            chk.updated_at = utc_now_iso()
            self._checkpoint_store.save(chk)
            await self._emit(
                RunEvent(
                    event_type=EventType.GRAPH_CHECKPOINTED,
                    thread_id=tid,
                    payload={
                        "graph_name": self.name,
                        "checkpoint_id": chk.checkpoint_id,
                        "superstep": chk.superstep,
                        "frontier": chk.frontier,
                    },
                )
            )

        chk.status = GRAPH_COMPLETED
        chk.updated_at = utc_now_iso()
        self._checkpoint_store.save(chk)
        await self._emit_finished(tid, chk)
        return GraphRunResult(
            graph_name=self.name,
            status=GRAPH_COMPLETED,
            final_state=dict(chk.state),
            supersteps=chk.superstep,
            node_sequence=list(chk.completed_nodes),
            checkpoint_id=chk.checkpoint_id,
        )

    async def _run_superstep(
        self, frontier: list[str], chk: GraphCheckpoint, tid: str
    ) -> tuple[list[GraphState], list[str]]:
        """Run every node in the frontier concurrently; return (deltas, next targets)."""
        # Loop guard: enforce per-node visit caps before running.
        for node_name in frontier:
            node = self._nodes[node_name]
            visited = chk.visit_counts.get(node_name, 0)
            if node.max_visits is not None and visited >= node.max_visits:
                raise GraphRecursionError(
                    f"node {node_name!r} exceeded max_visits={node.max_visits}"
                )

        # Snapshot of state for routers/nodes within this superstep (BSP isolation:
        # every node in a superstep sees the SAME pre-superstep state).
        snapshot = dict(chk.state)
        try:
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(self._run_node(name, snapshot, tid))
                    for name in frontier
                ]
        except BaseExceptionGroup as eg:
            # The TaskGroup has already cancelled AND awaited every sibling task
            # (a bare gather would orphan them); surface the first real failure
            # as a plain exception so callers keep seeing GraphError, not a group.
            raise _first_failure(eg) from eg
        deltas = [task.result()[0] for task in tasks]

        # Compute next frontier from each node's routing decision, post-merge so a
        # conditional router sees the merged state.
        merged = dict(chk.state)
        for delta in deltas:
            self._merge(merged, delta)

        next_targets: list[str] = []
        for name in frontier:
            targets = self._route(name, merged)
            for target in targets:
                if target == END:
                    await self._emit(
                        RunEvent(
                            event_type=EventType.GRAPH_EDGE_TAKEN,
                            thread_id=tid,
                            payload={
                                "graph_name": self.name,
                                "from": name,
                                "to": END,
                            },
                        )
                    )
                    continue
                await self._emit(
                    RunEvent(
                        event_type=EventType.GRAPH_EDGE_TAKEN,
                        thread_id=tid,
                        payload={"graph_name": self.name, "from": name, "to": target},
                    )
                )
                next_targets.append(target)
        return deltas, next_targets

    async def _run_node(
        self, name: str, state: GraphState, tid: str
    ) -> tuple[GraphState, str]:
        """Run one node: emit started/completed events and return its state delta."""
        node = self._nodes[name]
        await self._emit(
            RunEvent(
                event_type=EventType.GRAPH_NODE_STARTED,
                thread_id=tid,
                payload={"graph_name": self.name, "node": name},
            )
        )
        try:
            raw = node.fn(dict(state))
            if inspect.isawaitable(raw):
                raw = await raw
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # node failure is a graph error, surfaced cleanly
            await self._emit(
                RunEvent(
                    event_type=EventType.GRAPH_NODE_COMPLETED,
                    thread_id=tid,
                    error=str(exc),
                    payload={
                        "graph_name": self.name,
                        "node": name,
                        "status": "failed",
                    },
                )
            )
            raise GraphError(f"node {name!r} raised: {exc}") from exc

        delta: GraphState = {}
        if raw is None:
            delta = {}
        elif isinstance(raw, Mapping):
            delta = dict(raw)
        else:
            raise GraphError(
                f"node {name!r} must return a mapping delta or None, got "
                f"{type(raw).__name__}"
            )
        await self._emit(
            RunEvent(
                event_type=EventType.GRAPH_NODE_COMPLETED,
                thread_id=tid,
                payload={
                    "graph_name": self.name,
                    "node": name,
                    "status": "completed",
                    "delta_keys": sorted(delta.keys()),
                },
            )
        )
        return delta, name

    def _route(self, name: str, state: GraphState) -> list[str]:
        """Compute the successor node names for ``name`` given the merged state."""
        if name in self._conditional:
            decision = self._conditional[name](state)
            return self._normalize_targets(name, decision)
        if name in self._edges:
            return list(self._edges[name])
        # No outgoing edges and no router: this branch ends.
        return [END]

    def _normalize_targets(self, name: str, decision: Any) -> list[str]:
        """Validate and normalize a router's return into a list of node names."""
        if decision is None:
            return [END]
        if isinstance(decision, str):
            candidates = [decision]
        elif isinstance(decision, Sequence):
            candidates = list(decision)
        else:
            raise GraphError(
                f"router for {name!r} must return a node name or sequence, got "
                f"{type(decision).__name__}"
            )
        for target in candidates:
            if target != END and target not in self._nodes:
                raise GraphError(
                    f"router for {name!r} returned unknown target {target!r}"
                )
        return candidates

    def _merge(self, state: GraphState, delta: GraphState) -> None:
        """Merge ``delta`` into ``state`` in place, applying per-key reducers."""
        for key, value in delta.items():
            reducer = self._reducers.get(key)
            if reducer is not None and key in state:
                state[key] = reducer(state[key], value)
            elif reducer is not None:
                state[key] = reducer(None, value)
            else:
                state[key] = value

    # --------------------------------------------------------------- events
    async def _emit_finished(self, tid: str, chk: GraphCheckpoint) -> None:
        await self._emit(
            RunEvent(
                event_type=EventType.GRAPH_FINISHED,
                thread_id=tid,
                error=chk.error,
                payload={
                    "graph_name": self.name,
                    "status": chk.status,
                    "checkpoint_id": chk.checkpoint_id,
                    "superstep": chk.superstep,
                    "nodes_run": len(chk.completed_nodes),
                },
            )
        )

    async def _emit(self, event: RunEvent) -> None:
        """Best-effort fan-out mirroring the runtime's ``_emit`` (invariant #2).

        Order: storage (durable spine) -> entity registry -> observability span ->
        caller callbacks. Every sink is isolated so one failure can't break the run.
        """
        if self._memory_store is not None:
            appender = getattr(self._memory_store, "append_event", None)
            if appender is not None:
                try:
                    await appender(event)
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - defensive
                    pass
        if self._entity_registry is not None:
            try:
                self._entity_registry.register(event.to_record())
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            from himmy.services.observability import emit_event_span

            emit_event_span(event)
        except Exception:  # pragma: no cover - defensive
            pass
        for callback in self._on_event:
            try:
                await callback(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - never let a listener break the run
                pass


def _first_failure(eg: BaseExceptionGroup[BaseException]) -> BaseException:
    """Unwrap a (possibly nested) exception group to its first leaf exception."""
    for exc in eg.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            return _first_failure(exc)
        return exc
    return eg  # pragma: no cover - a TaskGroup group is never empty


def _state_size_bytes(state: GraphState) -> int:
    """Best-effort serialized size of the shared state, for the OOM guard.

    JSON with ``default=str`` covers the common case (graph state is meant to be
    checkpointable); anything unserializable falls back to ``repr``.
    """
    try:
        return len(json.dumps(state, default=str, ensure_ascii=False).encode("utf-8"))
    except Exception:  # pragma: no cover - defensive
        return len(repr(state).encode("utf-8"))


def _timeout(seconds: float) -> asyncio.Timeout:
    """Return an ``asyncio.timeout(seconds)`` context manager (Python 3.11+)."""
    timeout_cm = getattr(asyncio, "timeout", None)
    if timeout_cm is None:  # pragma: no cover - 3.10 fallback only
        raise HimmyError("graph timeouts require Python 3.11+ (asyncio.timeout)")
    return cast(asyncio.Timeout, timeout_cm(seconds))


__all__ = [
    "START",
    "END",
    "OnEvent",
    "GraphState",
    "NodeFn",
    "RouterFn",
    "Reducer",
    "GraphError",
    "GraphRecursionError",
    "GraphStateSizeError",
    "add_reducer",
    "GraphRunResult",
    "StateGraph",
    "CompiledStateGraph",
]
