"""API kernel: the dependency container wiring kernels into application services.

``ApiContainer.build_default`` assembles the offline-first stack (in-memory
storage + entity registry + stub inference + runtime + app services). For
production, set ``HIMMY_DATABASE_URL`` and use
:meth:`ApiContainer.build_default_async` to wire a Postgres-backed store (AAEO-2);
``build_default`` stays offline-green (in-memory) so ``create_app()`` works with
zero configuration. Swap ``_build_inference`` / ``_build_storage`` or wrap
``build_default`` to inject other production backends without touching routers.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.registry import EntityRegistry
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.inference.service import InferenceService
    from himmy.services.storage.service import StorageService


class ApiContainer:
    """Holds the singletons the routers depend on for one app instance."""

    def __init__(
        self,
        *,
        storage: StorageService,
        entity_registry: EntityRegistry,
        inference: InferenceService,
        runtime: SingleAgentRuntime,
        context_app: Any,
        run_app: Any,
        recommendation_app: Any,
        dashboard: Any,
        evaluation: Any = None,
    ) -> None:
        """Store the assembled service singletons."""
        self.storage = storage
        self.entity_registry = entity_registry
        self.inference = inference
        self.runtime = runtime
        self.context_app = context_app
        self.run_app = run_app
        self.recommendation_app = recommendation_app
        self.dashboard = dashboard
        self.evaluation = evaluation

    @classmethod
    def build_default(cls) -> ApiContainer:
        """Assemble the default offline-first container (in-memory + stub inference).

        Synchronous and zero-config: always uses the in-memory
        :class:`StorageService` so ``create_app()`` stays offline-green even when
        ``HIMMY_DATABASE_URL`` is set. For a Postgres-backed store use
        :meth:`build_default_async` (which can create + migrate the pool).
        """
        from himmy.services.storage.service import StorageService

        return cls._assemble(StorageService())

    @classmethod
    async def build_default_async(cls) -> ApiContainer:
        """Async container builder that honours ``HIMMY_DATABASE_URL`` (AAEO-2).

        When ``HIMMY_DATABASE_URL`` is set, constructs a
        :class:`PostgresStorageService` (creating + migrating the pool) and wires
        it as the durable backend so background runs and idempotency survive
        restarts and span workers. Otherwise falls back to the in-memory store, so
        this is safe to call in any environment.
        """
        storage = await cls._build_storage()
        return cls._assemble(storage)

    @staticmethod
    async def _build_storage() -> Any:
        """Construct the storage backend from env (Postgres when DSN set, else memory).

        ``HIMMY_DATABASE_URL`` selects Postgres; the pool is created with the
        JSONB codec + timeouts, then migrated. When unset (the default), the
        in-memory store keeps the framework offline-green.
        """
        dsn = os.environ.get("HIMMY_DATABASE_URL")
        if dsn:
            from himmy.services.storage.postgres import PostgresStorageService

            storage = await PostgresStorageService.connect(dsn)
            await storage.migrate()
            return storage
        from himmy.services.storage.service import StorageService

        return StorageService()

    @classmethod
    def _assemble(cls, storage: Any) -> ApiContainer:
        """Wire the application services over a given storage backend."""
        from himmy.application.services import (
            ContextAppService,
            DashboardQueryService,
            RecommendationAppService,
            RunAppService,
        )
        from himmy.entities.registry import EntityRegistry
        from himmy.runtime.single_agent import SingleAgentRuntime
        from himmy.services.context.service import ContextService
        from himmy.services.evaluation.service import EvaluationService
        from himmy.services.prompts.manager import PromptManager
        from himmy.services.prompts.mapper import ContextPromptMapper

        registry = EntityRegistry()
        inference = cls._build_inference()

        context_service = ContextService(
            storage_service=storage, entity_registry=registry
        )
        runtime = SingleAgentRuntime(
            inference_service=inference,
            memory_store=storage,
            context_service=context_service,
            prompt_manager=PromptManager(),
            context_prompt_mapper=ContextPromptMapper(),
            entity_registry=registry,
        )

        recommendation_app = RecommendationAppService(
            storage=storage, entity_registry=registry
        )
        context_app = ContextAppService(
            context_service=context_service, storage=storage
        )
        run_app = RunAppService(
            runtime=runtime,
            storage=storage,
            entity_registry=registry,
            recommendation_app=recommendation_app,
        )
        dashboard = DashboardQueryService(storage=storage)
        # The LLM-judge metric path (AAEO-10) can reach the inference service.
        evaluation = EvaluationService(
            storage_service=storage, inference_service=inference
        )

        return cls(
            storage=storage,
            entity_registry=registry,
            inference=inference,
            runtime=runtime,
            context_app=context_app,
            run_app=run_app,
            recommendation_app=recommendation_app,
            dashboard=dashboard,
            evaluation=evaluation,
        )

    @staticmethod
    def _build_inference() -> InferenceService:
        """Build the inference service: gateway when keyed, else the offline stub."""
        from himmy.services.inference.client_manager import (
            GatewayClientManager,
            StubClientManager,
        )
        from himmy.services.inference.models import GatewayRuntimeConfig
        from himmy.services.inference.service import InferenceService

        if os.environ.get("PYDANTIC_AI_GATEWAY_API_KEY"):
            region = os.environ.get("HIMMY_GATEWAY_REGION", "us")
            manager: Any = GatewayClientManager(GatewayRuntimeConfig(region=region))
        else:
            manager = StubClientManager()
        return InferenceService(manager)

    async def aclose(self) -> None:
        """Close any resources the container owns (e.g. a Postgres pool)."""
        closer = getattr(self.storage, "close", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # pragma: no cover - best-effort teardown
                pass


__all__ = ["ApiContainer"]
