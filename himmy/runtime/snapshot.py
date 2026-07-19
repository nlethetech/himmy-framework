"""Context-snapshot resolution for :class:`SingleAgentRuntime`.

Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
step ``snapshot``). :class:`SnapshotResolver` owns the "resolve a snapshot from
arg/context, or build one" behavior: it loads a stored snapshot, or builds one
from a declared ``context_build_spec`` (tenant-scoped via ``workspace_id``),
emits ``CONTEXT_SNAPSHOT_BUILT`` on the audit spine, and honors
``strict_snapshot`` by re-raising when a *requested* snapshot is unavailable.

The runtime constructs one of these in ``__init__`` and its ``_resolve_snapshot``
method delegates here; behavior (event order, payloads, exception types) is
byte-for-byte identical to the pre-extraction inline implementation.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona
    from himmy.runtime.single_agent import SingleAgentRuntime


def _truncate(text: str, cap: int) -> str:
    text = text or ""
    return text if len(text) <= cap else text[: cap - 1] + "…"


def _context_workspace_id(ctx: dict[str, Any]) -> str | None:
    """The run's owning ``workspace_id`` (for tenant-scoping the context build), if any.

    Read from ``context_metadata`` — the same place :func:`_cache_scope_metadata` reads it
    for cache partitioning — so the per-run context-snapshot build can tenant-scope its
    STORAGE reads exactly as the HTTP ``/v1/context/snapshot`` route does. Returns ``None``
    for an unscoped run (offline / single-tenant / CLI — no ``workspace_id`` carried), which
    keeps :meth:`ContextService.build_snapshot` resolution byte-for-byte unchanged.
    """
    context_metadata = ctx.get("context_metadata")
    if isinstance(context_metadata, dict):
        value = context_metadata.get("workspace_id")
        if value:
            return str(value)
    return None


def _snapshot_grounding(snapshot: Any) -> list[dict[str, Any]]:
    """Knowledge citations a snapshot pulled into the prompt (one entry per KB field).

    Reads each ``knowledge_base``-sourced :class:`ContextField`: the query it ran and
    the chunks it retrieved, each with a snippet, similarity, and source URI. Returns
    ``[]`` when no KB field was resolved, so non-RAG agents add nothing.
    """
    out: list[dict[str, Any]] = []
    fields = getattr(snapshot, "fields", None) or {}
    for key, fld in fields.items():
        if getattr(fld, "source", None) != "knowledge_base":
            continue
        value = getattr(fld, "value", None) or {}
        chunks = value.get("chunks") if isinstance(value, dict) else None
        meta = getattr(fld, "metadata", None) or {}
        citations = []
        for c in chunks or []:
            snippet = c.get("text") or c.get("context_window") or ""
            citations.append(
                {
                    "text": _truncate(str(snippet), 400),
                    "similarity": c.get("similarity"),
                    "source_uri": c.get("source_uri"),
                }
            )
        out.append(
            {
                "source": "knowledge",
                "key": key,
                "query": meta.get("query"),
                "kb_name": meta.get("kb_name"),
                "citations": citations,
            }
        )
    return out


class SnapshotResolver:
    """Resolves (loads or builds) a context snapshot for one run.

    Holds a back-reference to the owning :class:`SingleAgentRuntime` and reads its
    live wiring (``context_service``, ``memory_store``, ``strict_snapshot``) and
    ``_emit`` at call time, so runtime reconfiguration between runs is honored
    exactly as when the logic lived inline on the runtime.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    async def resolve(
        self,
        persona: Persona,
        task: Task,
        ctx: dict[str, Any],
        snapshot_id: str | None,
    ) -> tuple[Any, str | None, str | None]:
        """Resolve a snapshot from arg/context, or build one.

        Returns ``(snapshot, resolved_id, snapshot_error)``. A snapshot was
        explicitly *requested* when a ``snapshot_id`` was supplied or a
        ``context_build_spec`` is present; in that case a load/build failure is
        diagnosed (RO-11): the error is captured for the AGENT_RUN_STARTED /
        CONTEXT_SNAPSHOT_BUILT payload, and — when ``strict_snapshot`` is on —
        re-raised as an :class:`HimmyError` so the caller knows the requested
        evidence was unavailable instead of silently running without it.
        """
        rt = self._rt
        snapshot: Any = None
        snapshot_error: str | None = None
        resolved_id = snapshot_id or ctx.get("snapshot_id")
        requested = bool(
            snapshot_id
            or ctx.get("snapshot_id")
            or ctx.get("context_build_spec") is not None
        )

        if rt.context_service is None:
            if requested and rt.strict_snapshot:
                raise HimmyError("snapshot requested but no context_service is wired")
            return (
                None,
                resolved_id,
                ("no context_service wired" if requested else None),
            )

        # Load an existing snapshot when an id was supplied and storage is present.
        if resolved_id and rt.memory_store is not None:
            loader = getattr(rt.memory_store, "load_snapshot", None)
            if loader is not None:
                try:
                    snapshot = await loader(resolved_id)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - diagnose, don't crash
                    snapshot_error = f"snapshot load failed: {exc}"
                if snapshot is None and snapshot_error is None:
                    snapshot_error = f"snapshot {resolved_id!r} not found"

        # Otherwise build one from a declared build spec.
        if snapshot is None and ctx.get("context_build_spec") is not None:
            subject_id = ctx.get("context_subject_id") or persona.agent_id
            # Tenant isolation (red-team reattack-r6): thread the run's workspace_id into
            # the per-run context build so STORAGE-sourced fields are tenant-scoped, exactly
            # as the HTTP /v1/context/snapshot route does. Without it, ContextService
            # resolved every stored field for a free-form ``subject_id`` (which defaults to
            # the SHARED ``persona.agent_id`` across tenants running the same spec) with NO
            # tenant filter, so tenant A's run could surface tenant B's cached context field
            # (a cross-tenant IDOR). ``None`` (offline / single-tenant / CLI — no
            # workspace_id on context_metadata) keeps resolution byte-for-byte unchanged.
            workspace_id = _context_workspace_id(ctx)
            try:
                snapshot = await rt.context_service.build_snapshot(
                    subject_id=subject_id,
                    task_id=task.task_id,
                    build_spec=ctx["context_build_spec"],
                    metadata=ctx.get("context_metadata"),
                    workspace_id=workspace_id,
                )
                resolved_id = snapshot.snapshot_id
                snapshot_error = None
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - diagnose, don't crash
                snapshot = None
                snapshot_error = f"snapshot build failed: {exc}"

        if snapshot is not None:
            await rt._emit(
                RunEvent(
                    event_type=EventType.CONTEXT_SNAPSHOT_BUILT,
                    thread_id=None,
                    agent_id=persona.agent_id,
                    payload={
                        "snapshot_id": getattr(snapshot, "snapshot_id", None),
                        "subject_id": getattr(snapshot, "subject_id", None),
                        "missing_required_keys": list(
                            getattr(snapshot, "missing_required_keys", []) or []
                        ),
                        # Knowledge/RAG grounding — which chunks were retrieved into
                        # the prompt, with citations (so the GUI can show "why it
                        # said that"). Empty when no KB-sourced field was resolved.
                        "grounding": _snapshot_grounding(snapshot),
                    },
                )
            )
            resolved_id = getattr(snapshot, "snapshot_id", resolved_id)
        elif snapshot_error is not None:
            # RO-11: surface the failure on the audit trail so 'requested but
            # unavailable' is distinguishable from 'no snapshot requested'.
            await rt._emit(
                RunEvent(
                    event_type=EventType.CONTEXT_SNAPSHOT_BUILT,
                    thread_id=None,
                    agent_id=persona.agent_id,
                    error=snapshot_error,
                    payload={
                        "snapshot_id": resolved_id,
                        "snapshot_error": snapshot_error,
                    },
                )
            )
            if requested and rt.strict_snapshot:
                raise HimmyError(snapshot_error)
        return snapshot, resolved_id, snapshot_error
