"""Multi-provider dispatch: route each ``model_key`` to a different backend.

A team can mix providers — a strong "brain" on the Claude Opus CLI orchestrating cheap
local Ollama workers, with a cloud model where you want it. The orchestrator already
threads a per-member ``model_key``; this manager maps each key to its own
:class:`~himmy.services.inference.client_manager.ClientManager` and delegates, so one
``InferenceService`` can fan out across heterogeneous backends. Unknown keys fall back to
a designated default backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.core import HimmyError
from himmy.services.inference.prompt_cache import (
    CacheCapability,
    resolve_cache_capability,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.inference.client_manager import ClientManager
    from himmy.services.inference.models import InferenceRequest, InferenceResponse


class MultiProviderClientManager:
    """Dispatch by ``model_key`` to per-key client managers (a multi-backend router)."""

    def __init__(
        self,
        managers: dict[str, ClientManager],
        *,
        default_key: str | None = None,
    ) -> None:
        """Map dispatch keys to managers; ``default_key`` handles unknown keys."""
        if not managers:
            raise HimmyError("MultiProviderClientManager needs at least one manager.")
        self._managers = dict(managers)
        if default_key is not None and default_key in self._managers:
            self._default_key = default_key
        else:
            self._default_key = next(iter(self._managers))
        self.provider_name = "multi"

    def _pick(self, model_key: str) -> ClientManager:
        """Choose the manager for ``model_key`` (or the default for unknown keys)."""
        return self._managers.get(model_key, self._managers[self._default_key])

    def resolve(self, model_key: str) -> str:
        """Resolve to ``<key>:<backend-model>`` for traceability."""
        return f"{model_key}:{self._pick(model_key).resolve('default')}"

    def cache_capability_for(self, model_key: str) -> CacheCapability:
        """The prompt-cache capability of the manager that serves ``model_key``.

        Dispatch picks a concrete sub-manager per key, so the capability that matters
        is that backend's — resolved via the same call-site rule (``getattr`` default
        :attr:`CacheCapability.NONE`) so a non-caching backend no-ops the policy.
        """
        return resolve_cache_capability(self._pick(model_key), "default")

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Delegate to the picked manager (reset model_key so it uses its own model)."""
        manager = self._pick(request.model_key)
        return await manager.generate(
            request.model_copy(update={"model_key": "default"})
        )


__all__ = ["MultiProviderClientManager"]
