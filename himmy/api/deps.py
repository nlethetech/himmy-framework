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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from himmy.services.governance.consent_registry import MESSAGES_CONTENT_FIELD

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.protocol import EntityRegistryProtocol
    from himmy.entities.records import EntityRecord
    from himmy.runtime.single_agent import SingleAgentRuntime
    from himmy.services.inference.service import InferenceService
    from himmy.services.storage.service import StorageService


# --------------------------------------------------------------------------- WS4.6 wiring
#
# The wiring layer is the one place that knows the runtime's *real* spine record kinds, so
# it (not the generic ``ConsentAwareRegistry``) decides which kinds are subject-bearing,
# how to resolve a subject for each, and which payload fields to crypto-encrypt under the
# subject key. ``run_event`` / ``message`` / ``chat_thread`` are otherwise subject-less —
# the runtime stamps ``metadata['subject_id']`` onto them for a governed run (see
# ``SingleAgentRuntime._subject_metadata``); ``context_snapshot`` / ``recommendation``
# carry ``subject_id`` in their projected payload.

#: Spine record kinds the ``ConsentAwareRegistry`` gates at purpose=RETAIN. Infrastructure
#: kinds (persona/prompt/agent_state/env_state) and the audit kinds (security_event /
#: consent / erasure_tombstone) are deliberately absent so they always reach the spine.
_GATED_SPINE_KINDS = frozenset(
    {
        "run_event",
        "message",
        "chat_thread",
        "context_snapshot",
        "recommendation",
    }
)

#: Per-kind subject-bearing fields encrypted under the subject's shreddable key on an ALLOW
#: write (so ``RetentionService.erase_subject`` renders them unrecoverable). ``chat_thread``
#: lists the :data:`MESSAGES_CONTENT_FIELD` marker so its *nested* ``messages[*].content``
#: is walked through the subject cipher — otherwise the immutable spine would keep a full
#: plaintext conversation that crypto-shred can never reach. ``run_event`` payload is already
#: stripped by the A2 TRAIN gate, so it lists no field; for a denied subject the record is
#: skipped entirely so nothing is persisted regardless.
_SPINE_ENCRYPTED_FIELDS: dict[str, tuple[str, ...]] = {
    "message": ("content",),
    "chat_thread": (MESSAGES_CONTENT_FIELD,),
    "recommendation": ("title", "summary", "rationale"),
}


def _spine_subject_of(record: EntityRecord) -> str | None:
    """Resolve a gated spine record's data subject (``None`` ⇒ fail closed at the gate).

    Reads ``metadata['subject_id']`` first (the runtime stamps it onto run_event / message
    / chat_thread for a governed run) then falls back to ``payload['subject_id']``
    (context_snapshot / recommendation project it into the payload).
    """
    subject = record.metadata.get("subject_id") or record.payload.get("subject_id")
    return str(subject) if subject else None


def _spine_encrypted_fields_for(kind: str) -> tuple[str, ...]:
    """The encryptable subject-bearing fields for a gated kind (``()`` when none)."""
    return _SPINE_ENCRYPTED_FIELDS.get(kind, ())


@dataclass(frozen=True)
class _GovernedOverlay:
    """The storage/registry the services see + the WS4.6 governance singletons.

    On the offline path ``storage``/``registry`` are the bare objects and everything else
    is ``None`` so the container is wired exactly as a pre-WS4.6 build.
    """

    storage: Any
    registry: Any
    consent_decider: Any = None
    ledger: Any = None
    policy: Any = None
    retention: Any = None


class ApiContainer:
    """Holds the singletons the routers depend on for one app instance."""

    def __init__(
        self,
        *,
        storage: StorageService,
        entity_registry: EntityRegistryProtocol,
        inference: InferenceService,
        runtime: SingleAgentRuntime,
        context_app: Any,
        run_app: Any,
        recommendation_app: Any,
        dashboard: Any,
        evaluation: Any = None,
        consent_ledger: Any = None,
        consent_policy: Any = None,
        retention_service: Any = None,
        privacy_audit: Any = None,
    ) -> None:
        """Store the assembled service singletons.

        ``consent_ledger`` / ``consent_policy`` / ``retention_service`` are wired only in
        the governed branch (``HIMMY_CONSENT`` on); they stay ``None`` on the zero-config
        path so nothing about the offline behaviour changes (WS4.6 A3). ``privacy_audit``
        (WS4.7 B4) is always wired — it is inert over an empty store — and the
        ``/v1/audit/privacy`` router reads it.
        """
        self.storage = storage
        self.entity_registry = entity_registry
        self.inference = inference
        self.runtime = runtime
        self.context_app = context_app
        self.run_app = run_app
        self.recommendation_app = recommendation_app
        self.dashboard = dashboard
        self.evaluation = evaluation
        self.consent_ledger = consent_ledger
        self.consent_policy = consent_policy
        self.retention_service = retention_service
        self.privacy_audit = privacy_audit

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

        Durable by default (item #3): when ``HIMMY_DATABASE_URL`` is set, constructs a
        :class:`PostgresStorageService` (creating + migrating the pool); otherwise a
        file-backed :class:`SqliteStorageService` at ``HIMMY_STORE_PATH``. Both wire a
        durable backend so background runs and idempotency survive restarts and span
        workers. The sync :meth:`build_default` still wires the in-memory store for
        zero-config offline use.
        """
        storage = await cls._build_storage()
        return cls._assemble(storage)

    @staticmethod
    async def _build_storage() -> Any:
        """Construct the DURABLE server storage backend via :class:`StoreFactory`.

        Delegates to :meth:`StoreFactory.for_server`, the single source of truth for the
        durable-default policy: Postgres when ``HIMMY_DATABASE_URL`` is a ``postgres://``
        DSN (pool created with the JSONB codec + timeouts, then migrated), otherwise a
        file-backed SQLite store at ``HIMMY_STORE_PATH`` (default ``.himmy/storage.db``).
        Both survive restarts and span workers — what a server/multi-worker entrypoint
        needs. ``build_default`` (sync) still wires the in-memory store so direct
        programmatic/offline use is unchanged.
        """
        from himmy.services.storage.factory import StoreFactory

        return await StoreFactory.for_server()

    @classmethod
    def _assemble(cls, storage: Any) -> ApiContainer:
        """Wire the application services over a given storage backend.

        Offline-first (WS4.6 A3): the ``HIMMY_CONSENT`` switch is consulted once here.
        When it is **off** (the default) ``_governed_overlay`` returns no governance
        objects and the bare ``storage``/``registry`` are wired exactly as before — the
        zero-config path is byte-identical. When it is **on**, the storage facade and the
        registry are wrapped (``ConsentGatedStorage`` / ``ConsentAwareRegistry``) *before*
        they reach the services and runtime, and the runtime gets the ``consent_decider``
        TRAIN gate.
        """
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

        # WS4.6: the spine registry and the storage facade the services/runtime see. The
        # inner ``registry`` stays the audit/consent/erasure spine (never gated); the
        # overlay hands back wrapped versions + the decider only in a governed deployment.
        overlay = cls._governed_overlay(storage, registry)
        service_storage = overlay.storage
        service_registry = overlay.registry

        context_service = ContextService(
            storage_service=service_storage, entity_registry=service_registry
        )
        runtime = SingleAgentRuntime(
            inference_service=inference,
            memory_store=service_storage,
            context_service=context_service,
            prompt_manager=PromptManager(),
            context_prompt_mapper=ContextPromptMapper(),
            entity_registry=service_registry,
            consent_decider=overlay.consent_decider,
        )

        recommendation_app = RecommendationAppService(
            storage=service_storage, entity_registry=service_registry
        )
        context_app = ContextAppService(
            context_service=context_service, storage=service_storage
        )
        run_app = RunAppService(
            runtime=runtime,
            storage=service_storage,
            entity_registry=service_registry,
            recommendation_app=recommendation_app,
        )
        dashboard = DashboardQueryService(storage=service_storage)
        # The LLM-judge metric path (AAEO-10) can reach the inference service.
        evaluation = EvaluationService(
            storage_service=service_storage, inference_service=inference
        )

        # WS4.7 B4: the privacy auditor reads the eval kernel + the INNER spine registry
        # (so it sees the un-gated security_event / consent / erasure_tombstone /
        # privacy_audit_report kinds) and is gated on the wired provider being real. Part A
        # only couples through a None-guarded ``consent_service`` seam.
        privacy_audit = cls._build_privacy_audit(
            registry=registry,
            evaluation=evaluation,
            inference=inference,
            overlay=overlay,
        )

        return cls(
            storage=service_storage,
            entity_registry=registry,
            inference=inference,
            runtime=runtime,
            context_app=context_app,
            run_app=run_app,
            recommendation_app=recommendation_app,
            dashboard=dashboard,
            evaluation=evaluation,
            consent_ledger=overlay.ledger,
            consent_policy=overlay.policy,
            retention_service=overlay.retention,
            privacy_audit=privacy_audit,
        )

    @staticmethod
    def _build_privacy_audit(
        *,
        registry: Any,
        evaluation: Any,
        inference: Any,
        overlay: _GovernedOverlay,
    ) -> Any:
        """Construct the WS4.7 :class:`PrivacyAuditService` (always wired, inert offline).

        The auditor reads the *inner* spine ``registry`` (never the gated wrapper) so the
        ``security_event`` / ``consent`` / ``erasure_tombstone`` / ``privacy_audit_report``
        kinds it scans are always visible. Its consent-derived metrics stay ``skipped``
        until a governed deployment hands it a ``consent_service`` (the overlay's ledger);
        the LLM probe metrics are gated on a *real* provider via the wired manager's
        ``provider_name`` (``None``/``'stub'`` ⇒ skipped — must_fix #5). It runs no I/O on
        construction, so the zero-config path is unchanged.
        """
        from himmy.services.audit.log import SecurityAuditLog
        from himmy.services.evaluation.privacy import (
            PrivacyAuditConfig,
            PrivacyAuditService,
        )

        provider = getattr(
            getattr(inference, "_client_manager", None), "provider_name", None
        )
        return PrivacyAuditService(
            entity_registry=registry,
            evaluation_service=evaluation,
            security_audit=SecurityAuditLog(registry),
            inference_service=inference,
            consent_service=getattr(overlay, "ledger", None),
            retention_service=getattr(overlay, "retention", None),
            config=PrivacyAuditConfig(provider=provider),
        )

    @staticmethod
    def _governed_overlay(storage: Any, registry: Any) -> _GovernedOverlay:
        """Build the WS4.6 governance overlay (no-op unless ``HIMMY_CONSENT`` is on).

        Off (default): returns the bare ``storage``/``registry`` and ``None`` for the
        ledger/policy/retention/decider — the offline path is untouched. On: constructs a
        :class:`ConsentPolicy` + :class:`ConsentLedger` (over the inner spine) + a
        :class:`SubjectKeyVault`-backed :class:`RetentionService`, then wraps the storage
        in :class:`ConsentGatedStorage` and the registry in :class:`ConsentAwareRegistry`
        so subject-bearing writes are gated at purpose=RETAIN before any service sees them.
        """
        from himmy.services.governance.consent import build_consent_policy

        policy = build_consent_policy()
        if not policy.governed:
            return _GovernedOverlay(storage=storage, registry=registry)

        from himmy.services.audit.log import SecurityAuditLog
        from himmy.services.governance.consent_ledger import ConsentLedger
        from himmy.services.governance.consent_registry import ConsentAwareRegistry
        from himmy.services.governance.consent_storage import ConsentGatedStorage
        from himmy.services.governance.retention import (
            RetentionService,
            SubjectKeyVault,
        )

        # The audit + ledger + erasure services all write to the INNER spine so their
        # records (security_event / consent / erasure_tombstone) are never gated.
        audit = SecurityAuditLog(registry)
        key_vault = SubjectKeyVault()
        retention = RetentionService(registry, key_vault=key_vault)
        ledger = ConsentLedger(registry, policy=policy, retention_service=retention)
        decider = ledger.decision

        gated_storage = ConsentGatedStorage(storage, decider=decider, audit=audit)
        gated_registry = ConsentAwareRegistry(
            registry,
            decider=decider,
            gated_kinds=_GATED_SPINE_KINDS,
            subject_extractor=_spine_subject_of,
            key_vault=key_vault,
            encrypted_fields_for=_spine_encrypted_fields_for,
            audit=audit,
        )
        return _GovernedOverlay(
            storage=gated_storage,
            registry=gated_registry,
            consent_decider=decider,
            ledger=ledger,
            policy=policy,
            retention=retention,
        )

    @staticmethod
    def _build_inference() -> InferenceService:
        """Build the inference service: gateway when keyed, else the offline stub."""
        from himmy.config.residency import enforce_region
        from himmy.config.secrets import get_secret
        from himmy.services.inference.client_manager import (
            GatewayClientManager,
            StubClientManager,
        )
        from himmy.services.inference.models import GatewayRuntimeConfig
        from himmy.services.inference.service import InferenceService

        if get_secret("PYDANTIC_AI_GATEWAY_API_KEY"):
            region = os.environ.get("HIMMY_GATEWAY_REGION", "us")
            # Residency boundary (WS4.3): the inference gateway region must be inside
            # the residency policy, so regulated prompts/completions cannot leave the
            # jurisdiction. A no-op unless HIMMY_REGION is pinned.
            enforce_region(region, context="inference gateway")
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
