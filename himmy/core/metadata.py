"""Core: the canonical vocabulary of framework-written ``metadata`` keys.

Himmy models carry an open ``metadata: dict[str, Any]`` field by design — agents,
tools, skills, and connectors must be able to attach their own keys without the
framework knowing about them ahead of time. That extensibility is a feature, so the
model fields stay ``dict[str, Any]`` rather than a closed schema.

What was missing was a single place that NAMES and TYPES the keys the framework
itself writes, so they are discoverable, IDE-completable, and type-checked at the
write sites that opt in. These ``TypedDict``s are that reference. They are all
``total=False`` (every key optional, since metadata is partial and additive) and
are meant for typed CONSTRUCTION/access — assigning one to a ``dict[str, Any]``
field is transparent. Adopt them incrementally; unknown keys remain valid
everywhere, so extensions never break.
"""

from __future__ import annotations

from typing import Any, TypedDict


class AssistantMessageMetadata(TypedDict, total=False):
    """Keys the runtime stamps onto an assistant ``Message.metadata``.

    Observability + control flags so the application layer can read a run's
    outcome (cost, tokens, error, structured output) without exceptions.
    """

    request_id: str
    trace_id: str | None
    latency_ms: float | None
    model_path: str | None
    provider_name: str | None
    input_tokens: int
    output_tokens: int
    cost: float | None
    status: str
    error: str | None
    error_code: str | None
    output_structured: Any
    workflow_complete: bool
    streamed: bool
    compacted: bool
    approved: bool
    round_trip_complete: bool
    tool_call_id: str
    tool_name: str


class RouteMetadata(TypedDict, total=False):
    """Keys the routing/cache layer stamps onto an ``InferenceResponse.metadata``."""

    fallback_chain: list[dict[str, Any]]
    route_index: int
    route_label: str | None
    cache_hit: bool


class PersonaMetadata(TypedDict, total=False):
    """Role/skill keys on a ``Persona.metadata`` (set by agent/team specs)."""

    role: str
    skills: list[str]
    resolved_skill_tools: list[str]
    skill_routing_hints: dict[str, Any]


class ContextFieldMetadata(TypedDict, total=False):
    """Scoping/provenance keys on a ``ContextField`` / snapshot ``metadata``."""

    subject_id: str
    workspace_id: str
    cached_at: str
    extraction_error: str
    calibration_ece: float


class KnowledgeAdapterMetadata(TypedDict, total=False):
    """Config keys a knowledge context adapter reads from its spec ``metadata``."""

    client_id: str
    kb_name: str
    query: str
    query_template: str
    top_k: int
    similarity_threshold: float
    metadata_filters: dict[str, Any]
    freshness_seconds: int


class ToolEventMetadata(TypedDict, total=False):
    """Attribution/result keys on tool events + ``ToolReturnRecord.metadata``."""

    pack: str
    backend: str
    actor: str
    approved: bool
    error_code: str | None
    error_message: str | None
    latency_ms: float | None
    repaired_from: str
    suggestions: list[str]


__all__ = [
    "AssistantMessageMetadata",
    "ContextFieldMetadata",
    "KnowledgeAdapterMetadata",
    "PersonaMetadata",
    "RouteMetadata",
    "ToolEventMetadata",
]
