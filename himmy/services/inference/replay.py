"""Deterministic record-and-replay for inference: re-run an agent run exactly.

The whole value of an agent framework is the agentic loop; debugging a failure by
"rerun and hope" is the opposite of an audit-grade system. These two client managers turn
a run into a portable, replayable artifact:

* :class:`RecordingClientManager` wraps a real manager and appends every
  ``(cache_key, response)`` to an ordered :class:`InferenceCassette` as the run happens —
  the recorded response already carries ``tool_calls``/``tool_returns``, so tool *outputs*
  are captured too.
* :class:`ReplayClientManager` answers each request from the cassette instead of a
  provider: it computes the request's :func:`compute_cache_key` and returns the next
  recorded response for that key (FIFO, so retries/duplicates replay in order). Tools are
  **not** re-executed — the recorded response is returned verbatim — so a replay has no
  side effects and is fully deterministic.

Matching is by content hash (:func:`compute_cache_key`), which excludes the random
``request_id``, so a re-run whose requests are logically identical replays exactly even
though every ``request_id`` differs.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from pydantic import BaseModel, Field

from himmy.core.errors import HimmyError
from himmy.services.inference.cache import compute_cache_key
from himmy.services.inference.client_manager import ClientManager
from himmy.services.inference.models import InferenceRequest, InferenceResponse

CASSETTE_VERSION = 1


class ReplayError(HimmyError):
    """A replay request had no recorded response (the cassette is incomplete/stale)."""


class CassetteEntry(BaseModel):
    """One recorded inference exchange: the request's content hash and its response."""

    cache_key: str
    model_key: str
    response: InferenceResponse
    request_preview: str = ""  # first user message, for human-readable debugging


class InferenceCassette(BaseModel):
    """An ordered recording of a run's inference exchanges — a portable replay artifact."""

    version: int = CASSETTE_VERSION
    label: str = ""
    entries: list[CassetteEntry] = Field(default_factory=list)

    def dump(self, path: str | Path) -> Path:
        """Write the cassette to ``path`` as JSON; returns the path."""
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> InferenceCassette:
        """Load a cassette from a JSON file."""
        raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        return cls.model_validate(raw)


def _preview(request: InferenceRequest) -> str:
    for m in reversed(request.messages):
        if m.role == "user":
            return m.content[:120]
    return ""


class RecordingClientManager:
    """Wrap a real :class:`ClientManager`, recording every exchange to a cassette."""

    def __init__(self, inner: ClientManager, *, label: str = "") -> None:
        self._inner = inner
        self.cassette = InferenceCassette(label=label)
        self.provider_name = getattr(inner, "provider_name", "recording")

    def resolve(self, model_key: str) -> str:
        return self._inner.resolve(model_key)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        response = await self._inner.generate(request)
        self.cassette.entries.append(
            CassetteEntry(
                cache_key=compute_cache_key(request),
                model_key=request.model_key,
                response=response,
                request_preview=_preview(request),
            )
        )
        return response

    def dump(self, path: str | Path) -> Path:
        """Persist the recorded cassette."""
        return self.cassette.dump(path)


class ReplayClientManager:
    """Answer inference requests from a recorded cassette — no provider, deterministic."""

    def __init__(
        self,
        cassette: InferenceCassette,
        *,
        fallback: ClientManager | None = None,
    ) -> None:
        self._cassette = cassette
        self._fallback = fallback
        self.provider_name = "replay"
        # cache_key -> queue of recorded responses, consumed in recorded order (FIFO).
        self._by_key: dict[str, deque[InferenceResponse]] = {}
        for entry in cassette.entries:
            self._by_key.setdefault(entry.cache_key, deque()).append(entry.response)

    @classmethod
    def from_file(
        cls, path: str | Path, *, fallback: ClientManager | None = None
    ) -> ReplayClientManager:
        return cls(InferenceCassette.load(path), fallback=fallback)

    def resolve(self, model_key: str) -> str:
        return f"replay:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        key = compute_cache_key(request)
        queue = self._by_key.get(key)
        if queue:
            # Return a deep copy so the caller can stamp request_id/latency freely.
            return queue.popleft().model_copy(
                deep=True, update={"request_id": request.request_id}
            )
        if self._fallback is not None:
            return await self._fallback.generate(request)
        raise ReplayError(
            f"no recorded response for request (cache_key={key[:12]}…); the cassette is "
            f"incomplete or the run diverged from the recording"
        )

    def remaining(self) -> int:
        """How many recorded responses have not yet been replayed (0 == fully consumed)."""
        return sum(len(q) for q in self._by_key.values())


__all__ = [
    "CASSETTE_VERSION",
    "ReplayError",
    "CassetteEntry",
    "InferenceCassette",
    "RecordingClientManager",
    "ReplayClientManager",
]
