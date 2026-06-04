"""Context kernel: the ContextAdapter plugin contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from opensims.services.context.models import ContextField


class ContextAdapter(ABC):
    """A plugin that fetches a :class:`ContextField` for a given key.

    Adapters wrap external sources — a CRM, a market-data API, a vector store —
    and are keyed by their ``name`` so a :class:`ContextSpecKey` can target one
    explicitly. Returning ``None`` means "this adapter has nothing for that key".
    """

    #: Unique adapter name, matched against ``ContextSpecKey.adapter_name``.
    name: str = "adapter"

    @abstractmethod
    async def fetch(self, key: str, scope: dict) -> ContextField | None:
        """Fetch a context field for ``key`` within ``scope``, or None."""
        raise NotImplementedError


__all__ = ["ContextAdapter"]
