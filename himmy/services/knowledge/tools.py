"""Knowledge kernel: the in-run ``kb_search`` tool registration.

This lives in the knowledge package (it imports :class:`ToolRegistry` read-only)
so the tool surface and the declarative :class:`KnowledgeBaseAdapter` share one
retrieval shape — an ad-hoc mid-run lookup lands as a ``TOOL_CALLED`` event on the
thread with the same :class:`EvidenceRef` projection as a context-built lookup
(the doc's "two retrieval paths, one shape" claim). Tenancy is enforced on every
call: the KB is resolved by ``(workspace_id, client_id, kb_name)``, so a raw
``kb_id`` can never reach another tenant's chunks.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError
from himmy.services.knowledge.retrieval.config import RetrievalMode
from himmy.services.knowledge.service import KnowledgeBase, build_kb_context_field

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a tools<->knowledge cycle
    from himmy.services.tools.models import ToolDefinition
    from himmy.services.tools.registry import ToolRegistry


@contextlib.asynccontextmanager
async def _mode_override(
    kb_service: KnowledgeBase, mode: str | None
) -> AsyncIterator[None]:
    """Temporarily flip the KB's dense/hybrid retrieval mode for one call.

    ``mode=None`` leaves the KB's configured default untouched. A requested mode
    rebuilds the active :class:`RetrievalConfig` with only ``mode`` changed (so
    rerank/rewrite stages and tuning knobs are inherited), and the original config
    is always restored on exit — even if the search raises.
    """
    if mode is None or mode == kb_service._retrieval.mode:
        yield
        return
    new_mode: RetrievalMode
    if mode == "dense":
        new_mode = "dense"
    elif mode == "hybrid":
        new_mode = "hybrid"
    else:
        raise HimmyError(f"kb_search mode must be 'dense' or 'hybrid', got {mode!r}.")
    original = kb_service._retrieval
    kb_service._retrieval = replace(original, mode=new_mode)
    try:
        yield
    finally:
        kb_service._retrieval = original


#: JSON schema for ``kb_search`` arguments (drives pydantic-ai arg validation when
#: the tool is exposed through the provider path, and the offline arg validator).
KB_SEARCH_ARGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to retrieve."},
        "kb_name": {
            "type": "string",
            "description": "Name of the knowledge base to search.",
        },
        "workspace_id": {"type": "string"},
        "client_id": {"type": "string"},
        "top_k": {"type": "integer", "minimum": 1},
        "similarity_threshold": {"type": "number"},
        "metadata_filters": {"type": "object"},
        "mode": {
            "type": "string",
            "enum": ["dense", "hybrid"],
            "description": (
                "Retrieval mode for this call. 'hybrid' fuses BM25 (lexical) and "
                "dense vectors via RRF; 'dense' is pure semantic cosine. Omit to use "
                "the knowledge base's configured default."
            ),
        },
    },
    "required": ["query", "kb_name"],
}


#: Per-chunk ranking-provenance scalars the model never needs (they explain the
#: ranking for the audit trail, not the answer). Stripped from the model-facing
#: chunk in ``kb_search`` — the full dumps still reach the audit path elsewhere.
_RANK_BREADCRUMB_KEYS: frozenset[str] = frozenset(
    {
        "dense_rank",
        "dense_sim",
        "lexical_rank",
        "lexical_score",
        "rrf_score",
        "rerank_score",
        "start_pos",
        "end_pos",
    }
)


def _lean_chunk(chunk: dict[str, Any]) -> dict[str, Any]:
    """Slim one model-facing chunk so its text is not re-sent ~3x every turn.

    Drops the padded ``context_window`` superset whenever the chunk carries its
    own ``text`` (kept only for text-less chunks, e.g. images, whose snippet lives
    in ``context_window``) and strips the ranking-breadcrumb metadata scalars.
    ``text``, ``chunk_id``, ``similarity``, and ``source_uri`` are preserved so GUI
    citations and downstream grounding still resolve.
    """
    slim = dict(chunk)
    if slim.get("text"):
        slim.pop("context_window", None)
    meta = slim.get("metadata")
    if isinstance(meta, dict):
        slim["metadata"] = {
            k: v for k, v in meta.items() if k not in _RANK_BREADCRUMB_KEYS
        }
    return slim


def register_kb_search_tool(
    registry: ToolRegistry,
    kb_service: KnowledgeBase,
    *,
    name: str = "kb_search",
    default_workspace_id: str | None = None,
    default_client_id: str | None = None,
    description: str | None = None,
) -> ToolDefinition:
    """Register an in-run ``kb_search`` LOCAL tool over a :class:`KnowledgeBase`.

    The handler resolves the KB by ``(workspace_id, client_id, kb_name)`` —
    enforcing tenancy: a caller can only reach KBs in its own workspace/client, and
    ``default_workspace_id``/``default_client_id`` pin the scope when the agent does
    not pass them. The result mirrors :class:`KnowledgeBaseAdapter`'s
    :class:`~himmy.services.context.models.ContextField` projection exactly
    (``chunks``, ``rendered_text``, ``confidence``, ``evidence_refs``), so an in-run
    lookup is audited identically to a context-built one. ``ToolService.execute``
    emits the ``TOOL_CALLED`` / ``TOOL_COMPLETED`` events around this handler.
    """
    # Imported here (not at module top) so this module imports without the tools
    # package having been initialised in an unusual order; still read-only on tools.
    from himmy.services.tools.registry import register_local_tool

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        kb_name = args.get("kb_name")
        if not query or not kb_name:
            raise HimmyError("kb_search requires 'query' and 'kb_name'.")
        workspace_id = args.get("workspace_id") or default_workspace_id
        client_id = args.get("client_id") or default_client_id
        if not workspace_id or not client_id:
            raise HimmyError(
                "kb_search requires a workspace_id and client_id (tenancy scope)."
            )

        kb = await kb_service.resolve_kb(
            workspace_id=str(workspace_id),
            client_id=str(client_id),
            name=str(kb_name),
        )
        if kb is None:
            raise HimmyError(
                f"knowledge base {kb_name!r} not found for "
                f"({workspace_id}, {client_id})."
            )

        threshold = args.get("similarity_threshold")
        # An optional per-call ``mode`` override turns this lookup hybrid (or back
        # to dense) without reconfiguring the shared KB. The override only flips the
        # dense/hybrid mode and inherits the KB's rerank/rewrite stages; the original
        # config is always restored, even on error.
        mode = args.get("mode")
        async with _mode_override(kb_service, mode):
            chunks = await kb_service.search(
                kb.kb_id,
                str(query),
                top_k=int(args.get("top_k", 5)),
                similarity_threshold=(
                    float(threshold) if threshold is not None else None
                ),
                metadata_filters=args.get("metadata_filters"),
                # Re-verify tenancy at the service boundary (defence in depth).
                workspace_id=str(workspace_id),
                client_id=str(client_id),
            )

        field = build_kb_context_field(
            key=str(kb_name),
            chunks=chunks,
            kb=kb,
            workspace_id=str(workspace_id),
            client_id=str(client_id),
            query=str(query),
        )
        return {
            "chunks": [_lean_chunk(c) for c in field.value["chunks"]],
            "rendered_text": field.value["rendered_text"],
            "confidence": field.confidence,
            "evidence_refs": [
                ref.model_dump(mode="json") for ref in field.evidence_refs
            ],
            "metadata": field.metadata,
        }

    return register_local_tool(
        registry,
        name=name,
        handler=_handler,
        description=(
            description
            or "Search a per-client knowledge base for the most relevant chunks "
            "and return them with similarity scores and evidence references."
        ),
        args_json_schema=KB_SEARCH_ARGS_SCHEMA,
        metadata={"kind": "knowledge_base_search"},
    )


__all__ = ["register_kb_search_tool", "KB_SEARCH_ARGS_SCHEMA"]
