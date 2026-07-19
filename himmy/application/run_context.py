"""Shared run-lifecycle context for :class:`~himmy.application.services.RunAppService`.

Extracted from :mod:`himmy.application.services` as the FOUNDATION collaborator for
the staged decomposition of ``RunAppService``: it holds the immutable shared handles
(runtime, storage, registry, recommendation service, HITL/checkpoint wiring), the
SINGLE background-task set, and the RUNTIME-mutable dispatch tunables in one cohesive
object that the service — and later collaborators — read THROUGH.

Behaviour is BYTE-IDENTICAL to the former inline attributes:

- there is exactly ONE ``_tasks`` set (so :meth:`RunAppService.drain` cancels every
  execute/resume/orchestration task); collaborators must never own a private set,
- the dispatch tunables (``dispatch_enabled`` / ``run_timeout_seconds`` /
  ``default_max_attempts`` / ``dispatch_fairness``) are LIVE — :meth:`enable_dispatch`
  mutates them on this object at runtime and every reader must observe the new value,
  never a construction-time snapshot.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from collections.abc import Callable

    from himmy.application.services import RecommendationAppService
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.storage.service import StorageService


class _RunContext:
    """Shared handles + the single background-task set + live dispatch tunables.

    Construction wires the immutable collaborators and seeds the tunables to their
    inline defaults; :meth:`RunAppService.enable_dispatch` later mutates the tunables
    in place. All fields are plain attributes so readers observe live values.
    """

    def __init__(
        self,
        *,
        runtime: SingleAgentRuntime,
        storage: StorageService,
        registry: EntityRegistryProtocol | None,
        recommendations: RecommendationAppService,
        run_timeout_seconds: float,
        checkpoint_store: Any,
        agent_resolver: Callable[..., Any] | None,
        conversation_sink: Callable[..., Any] | None,
        access_policy: Any,
        graph_checkpoint_store_provider: Callable[[], Any] | None,
        dispatch_enabled: bool,
        default_max_attempts: int,
        dispatch_fairness: bool,
    ) -> None:
        # -- Immutable shared handles --------------------------------------------
        self.runtime = runtime
        self.storage = storage
        self.registry = registry
        self.recommendations = recommendations
        self.checkpoint_store = checkpoint_store
        self.agent_resolver = agent_resolver
        self.conversation_sink = conversation_sink
        self.access_policy = access_policy
        self.graph_checkpoint_store_provider = graph_checkpoint_store_provider
        # -- The SINGLE background-task set (drain/cancel spine, AAEO-1) ----------
        # Keep strong refs so tasks are not GC'd mid-flight and can be
        # drained/cancelled on shutdown. Exactly one set across every run path.
        self.tasks: set[asyncio.Task[Any]] = set()
        # -- Live-read dispatch tunables (mutated by enable_dispatch at runtime) --
        self.run_timeout_seconds = run_timeout_seconds
        self.dispatch_enabled = dispatch_enabled
        self.default_max_attempts = default_max_attempts
        self.dispatch_fairness = dispatch_fairness
