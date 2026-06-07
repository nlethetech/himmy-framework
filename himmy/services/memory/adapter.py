"""Memory context adapter: auto-inject recalled memories into a prompt.

Registered on a :class:`~himmy.services.context.service.ContextService`, this adapter
turns "remembered" facts into grounding context: when the runtime builds a snapshot for
a subject, it recalls the most relevant memories and returns them as a rendered
:class:`ContextField`, so the agent sees its long-term memory without any tool call.
"""

from __future__ import annotations

from typing import Any

from himmy.services.context.adapters import ContextAdapter
from himmy.services.context.models import ContextField
from himmy.services.memory.service import MemoryService


class MemoryContextAdapter(ContextAdapter):
    """A :class:`ContextAdapter` that recalls a subject's memories for a key."""

    name = "memory"

    def __init__(
        self, memory: MemoryService, *, top_k: int = 5, subject_id: str | None = None
    ) -> None:
        """Wrap a :class:`MemoryService`; ``top_k`` caps how many memories inject.

        ``subject_id`` pins the recall subject (overriding the run's scope) so the
        adapter reads the same subject that facts were remembered under.
        """
        self._memory = memory
        self._top_k = top_k
        self._subject_id = subject_id

    async def fetch(self, key: str, scope: dict[str, Any]) -> ContextField | None:
        """Recall memories for the scope's subject and render them as a field."""
        subject_id = (
            self._subject_id
            or scope.get("subject_id")
            or scope.get("client_id")
            or "default"
        )
        query = str(
            scope.get("query") or scope.get("spec_metadata", {}).get("query") or ""
        )
        if not query:
            return None
        hits = await self._memory.recall(
            query, subject_id=subject_id, top_k=self._top_k
        )
        if not hits:
            return None
        rendered = "\n".join(f"- {h.record.text}" for h in hits)
        value: dict[str, Any] = {
            "rendered_text": rendered,
            "memories": [
                {"text": h.record.text, "similarity": h.similarity} for h in hits
            ],
        }
        return ContextField(
            key=key,
            value=value,
            source="memory",
            confidence=hits[0].similarity,
        )


__all__ = ["MemoryContextAdapter"]
