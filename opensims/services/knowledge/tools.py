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

from typing import TYPE_CHECKING, Any

from opensims.core.errors import OpenSimsError
from opensims.services.knowledge.service import KnowledgeBase, build_kb_context_field

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a tools<->knowledge cycle
    from opensims.services.tools.models import ToolDefinition
    from opensims.services.tools.registry import ToolRegistry


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
    },
    "required": ["query", "kb_name"],
}


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
    :class:`~opensims.services.context.models.ContextField` projection exactly
    (``chunks``, ``rendered_text``, ``confidence``, ``evidence_refs``), so an in-run
    lookup is audited identically to a context-built one. ``ToolService.execute``
    emits the ``TOOL_CALLED`` / ``TOOL_COMPLETED`` events around this handler.
    """
    # Imported here (not at module top) so this module imports without the tools
    # package having been initialised in an unusual order; still read-only on tools.
    from opensims.services.tools.registry import register_local_tool

    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        kb_name = args.get("kb_name")
        if not query or not kb_name:
            raise OpenSimsError("kb_search requires 'query' and 'kb_name'.")
        workspace_id = args.get("workspace_id") or default_workspace_id
        client_id = args.get("client_id") or default_client_id
        if not workspace_id or not client_id:
            raise OpenSimsError(
                "kb_search requires a workspace_id and client_id (tenancy scope)."
            )

        kb = await kb_service.resolve_kb(
            workspace_id=str(workspace_id),
            client_id=str(client_id),
            name=str(kb_name),
        )
        if kb is None:
            raise OpenSimsError(
                f"knowledge base {kb_name!r} not found for "
                f"({workspace_id}, {client_id})."
            )

        threshold = args.get("similarity_threshold")
        chunks = await kb_service.search(
            kb.kb_id,
            str(query),
            top_k=int(args.get("top_k", 5)),
            similarity_threshold=(float(threshold) if threshold is not None else None),
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
            "chunks": field.value["chunks"],
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
