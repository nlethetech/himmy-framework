"""Application kernel: the advisory recommendation contract returned by runs.

These types define the structured-output shape the framework recognizes: when a
run's ``output_structured`` matches :class:`RecommendationEnvelope`, the
application layer auto-extracts dashboard-facing :class:`Recommendation` items.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class Recommendation(BaseModel):
    """A single advisory recommendation produced by an agent run."""

    kind: str
    title: str
    summary: str = ""
    rationale: str = ""
    confidence: float = 0.0
    evidence_refs: list[str] = []
    metadata: dict[str, Any] = {}


class RecommendationEnvelope(BaseModel):
    """The structured-output schema a run emits to be auto-extracted as items.

    Used as the ``output_json_schema`` in structured-output runs (example 03) and
    as the parse target by :class:`RecommendationAppService`.
    """

    recommendations: list[Recommendation] = []
    summary: str | None = None


__all__ = ["Recommendation", "RecommendationEnvelope"]
