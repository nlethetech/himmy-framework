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

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from himmy.services.governance.consent_registry import MESSAGES_CONTENT_FIELD

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

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


class _LazyStoreProxy:
    """A reach-map store handle that resolves its target lazily at erase time (S4).

    The governed overlay (and thus the :class:`RetentionService`) is built BEFORE the durable
    memory + conversation singletons exist, yet the reach map needs their handles. This proxy
    defers resolution: ``delete_by_subject`` calls the injected resolver at erase time, returns
    ``0`` (no-op) when it resolves to ``None`` (offline / store never opened), and otherwise
    delegates ``delete_by_subject`` to the live store. It exposes ONLY that one method so a
    reach walk can never accidentally reach a non-erasure method.
    """

    def __init__(self, resolver: Callable[[], Any]) -> None:
        self._resolver = resolver

    def delete_by_subject(self, subject_id: str) -> int:
        store = self._resolver()
        if store is None:
            return 0
        return int(store.delete_by_subject(subject_id))


def _spine_eraser(storage: Any) -> Callable[[str], int]:
    """A sync ``(subject_id) -> int`` that hard-deletes the subject's spine threads+events.

    Adapts the wired storage backend's :meth:`delete_by_subject` (sync on the SQLite /
    in-memory facade, a coroutine on the Postgres backend) to the sync callable the
    :class:`SubjectReachMap` drives. A coroutine result is run on the shared aux
    :class:`_LoopThread` (a SEPARATE loop thread), so this never re-enters the request
    event loop that ``erase_subject`` runs on. ``storage`` is the INNER durable backend
    (the one ``_governed_overlay`` wraps), so its ``chat_threads`` + ``run_events`` rows —
    encrypted under the store-wide KEK, never the per-subject key — are physically removed.
    """

    def _erase(subject_id: str) -> int:
        result = storage.delete_by_subject(subject_id)
        if asyncio.iscoroutine(result):
            from himmy.services.storage.aux_store_factory import aux_loop

            return int(aux_loop().run(result))
        return int(result)

    return _erase


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
    key_vault: Any = None
    audit: Any = None

    def gate_conversation_store(self, store: Any) -> Any:
        """Wrap ``store`` in the S3 :class:`ConsentGatedConversationStore` when governed.

        Returns ``store`` unchanged on the ungoverned path (``consent_decider`` is ``None``),
        so the offline conversation store is byte-identical; otherwise the subject-attributed
        transcript is RETAIN-gated and per-subject envelope-encrypted at rest.
        """
        if self.consent_decider is None:
            return store
        from himmy.services.governance.sidecar_storage import (
            ConsentGatedConversationStore,
        )

        return ConsentGatedConversationStore(
            store,
            decider=self.consent_decider,
            key_vault=self.key_vault,
            audit=self.audit,
        )


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
        agent_app: Any = None,
        thread_app: Any = None,
        conversation_store: Any = None,
        knowledge_app: Any = None,
        evaluation: Any = None,
        consent_ledger: Any = None,
        consent_policy: Any = None,
        retention_service: Any = None,
        privacy_audit: Any = None,
        checkpoint_store: Any = None,
    ) -> None:
        """Store the assembled service singletons.

        ``consent_ledger`` / ``consent_policy`` / ``retention_service`` are wired only in
        the governed branch (``HIMMY_CONSENT`` on); they stay ``None`` on the zero-config
        path so nothing about the offline behaviour changes (WS4.6 A3). ``privacy_audit``
        (WS4.7 B4) is always wired — it is inert over an empty store — and the
        ``/v1/audit/privacy`` router reads it. ``checkpoint_store`` (T2f) is /v1's durable
        HITL inbox; the container owns its handle so :meth:`aclose` releases it.
        """
        self.storage = storage
        self.entity_registry = entity_registry
        self.inference = inference
        self.runtime = runtime
        self.context_app = context_app
        self.run_app = run_app
        self.agent_app = agent_app
        self.thread_app = thread_app
        self.conversation_store = conversation_store
        self.knowledge_app = knowledge_app
        self.recommendation_app = recommendation_app
        self.dashboard = dashboard
        self.evaluation = evaluation
        self.consent_ledger = consent_ledger
        self.consent_policy = consent_policy
        self.retention_service = retention_service
        self.privacy_audit = privacy_audit
        self.checkpoint_store = checkpoint_store

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
            AgentDefAppService,
            ContextAppService,
            DashboardQueryService,
            RecommendationAppService,
            RunAppService,
        )
        from himmy.entities.spine_factory import SpineFactory
        from himmy.runtime.single_agent import SingleAgentRuntime
        from himmy.services.context.service import ContextService
        from himmy.services.evaluation.service import EvaluationService
        from himmy.services.prompts.manager import PromptManager
        from himmy.services.prompts.mapper import ContextPromptMapper

        # T2.1: the spine is the DURABLE shared entity registry — no longer a bare
        # in-memory registry that evaporates on restart and is invisible to the
        # CLI's ``himmy seclog``. ``SpineFactory.for_context`` resolves the ONE
        # canonical ``.himmy/spine.db`` (or the opt-in Postgres spine) so a run's
        # provenance/lineage/security events projected here are the same on-disk database
        # the CLI reads. ``build_default`` runs BEFORE the app lifespan sets server
        # context, so this honours whatever context is active at construction; the
        # lifespan ALWAYS rebinds to the server spine before serving a request (so a
        # zero-DSN server still gets the durable spine — the build-order trap is closed in
        # ``app._build_lifespan``).
        registry = SpineFactory.for_context()
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
        # T2e: the durable workspace-scoped stored-agent (/v1/agents) resource. Shares
        # the same storage + spine the run service reads, so a stored agent projects an
        # ``agent`` node into the one canonical spine and a run launched by ``agent_id``
        # links back to it. Built BEFORE the run service so the run service can resolve a
        # paused run's stored spec on HITL resume (T2f) via this service.
        agent_app = AgentDefAppService(
            storage=service_storage,
            entity_registry=service_registry,
            reference_finder=cls._agent_reference_finder,
        )
        # T2f: /v1's OWN durable HITL checkpoint inbox (distinct from Studio's
        # ``.himmy/approvals.db``). Durable file-backed in a server context (so a paused
        # run survives a restart and the inbox is visible to all workers); in-memory in
        # the zero-config offline/test default so ``build_default`` writes no files. The
        # resolver lets the run service rebuild a paused run's tool-bearing runtime FROM
        # THE STORED DB SPEC (reviewer must_fix: /v1 has no filesystem ``agent_path``).
        checkpoint_store = cls._build_v1_checkpoint_store()
        # T2g: the ONE durable conversation store the CLI + Studio + /v1 all read. In a
        # server context this is the shared ``.himmy/conversations.db`` singleton (so a /v1
        # thread is browsable in Studio Chats / ``himmy chat``); in the zero-config offline
        # default it is a private in-memory store (no files written). Built before the run
        # service so the thread sink can be wired in.
        conversation_store = cls._build_conversation_store()
        # S3: a governed deployment wraps the conversation store so a subject-attributed
        # transcript is RETAIN-gated + per-subject envelope-encrypted at rest (and carries the
        # subject linkage the S4 reach map deletes by). No-op on the ungoverned/offline path.
        conversation_store = overlay.gate_conversation_store(conversation_store)
        # T3a: the durable workspace-namespaced knowledge resource (/v1/knowledge). Reuses
        # the container's service storage handle so the surface is consistent across
        # interfaces; the namespace boundary it enforces is a property of an AUTHENTICATED
        # deployment only (inert offline — see KnowledgeAppService's docstring).
        knowledge_app = cls._build_knowledge(service_storage)
        # The thread service is wired AFTER the run service is built (it needs the run
        # app), so the run service's conversation sink is bound via a late attribute set.
        run_app = RunAppService(
            runtime=runtime,
            storage=service_storage,
            entity_registry=service_registry,
            recommendation_app=recommendation_app,
            checkpoint_store=checkpoint_store,
            agent_resolver=agent_app.get_agent_def,
            # HITL orchestration resume reopens the SAME durable graph checkpoint store the
            # team/workflow run paused into (file-backed in a server, RAM offline).
            graph_checkpoint_store_provider=cls._graph_checkpoint_store_provider,
        )
        # T2g: the /v1 thread (conversation) resource over the one ConversationStore.
        from himmy.application.services import ThreadAppService

        thread_app = ThreadAppService(
            run_app=run_app,
            agent_app=agent_app,
            conversation_store=conversation_store,
        )
        # Wire the run service's conversation sink so an approval-resume (which finishes on
        # a background task the thread router never awaits) still re-projects the latest
        # turns into the ConversationStore. A late set keeps the construction order clean.
        run_app._conversation_sink = thread_app.project_run_thread
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
            agent_app=agent_app,
            thread_app=thread_app,
            conversation_store=conversation_store,
            knowledge_app=knowledge_app,
            recommendation_app=recommendation_app,
            dashboard=dashboard,
            evaluation=evaluation,
            consent_ledger=overlay.ledger,
            consent_policy=overlay.policy,
            retention_service=overlay.retention,
            privacy_audit=privacy_audit,
            checkpoint_store=checkpoint_store,
        )

    @staticmethod
    async def _agent_reference_finder(
        agent_id: str, workspace_id: str
    ) -> list[str]:
        """Referential check for :class:`AgentDefAppService` (T3b/T3c): who references ``agent_id``.

        Returns the routine / team / workflow ids in ``workspace_id`` that reference the
        stored agent, so a ``DELETE /v1/agents/{id}`` of an agent still wired to ANY of them
        returns HTTP 409 instead of orphaning the binding (the reviewer must_fix — and the
        production reference_finder the T3b teams/workflows need). Reads the SAME
        ``.himmy/routines.db`` / ``.himmy/teams.db`` / ``.himmy/workflows.db`` the
        ``/v1`` + Studio + CLI surfaces drive. Best-effort per store: a store error never
        blocks a delete (that store contributes no references).
        """
        references: list[str] = []
        try:
            from himmy.api.routines import get_routines_store

            references += [
                f"routine:{rid}"
                for rid in get_routines_store().find_by_agent_id(
                    agent_id, workspace_id=workspace_id
                )
            ]
        except Exception:  # pragma: no cover - a store error must not block deletes
            pass
        try:
            from himmy.api.teams_store import get_teams_store, get_workflows_store

            references += [
                f"team:{tid}"
                for tid in get_teams_store().find_by_agent_id(
                    agent_id, workspace_id=workspace_id
                )
            ]
            references += [
                f"workflow:{wid}"
                for wid in get_workflows_store().find_by_agent_id(
                    agent_id, workspace_id=workspace_id
                )
            ]
        except Exception:  # pragma: no cover - a store error must not block deletes
            pass
        return references

    @staticmethod
    def _build_v1_checkpoint_store() -> Any:
        """Build /v1's OWN HITL checkpoint store: durable in a server, in-RAM offline (T2f).

        In a server context (the lifespan-established durable deployment) this is a
        file-backed :class:`SqliteCheckpointStore` at ``.himmy/v1_approvals.db`` (resolved
        by :func:`himmy.config.project.v1_approvals_db_path`), DELIBERATELY a different
        file than Studio's ``.himmy/approvals.db`` — the two surfaces rebuild a paused run
        from different spec sources, so the inboxes stay per-surface (reviewer must_fix).
        Outside a server context (the zero-config ``build_default`` offline/test default)
        it is an in-memory store so no file is written and the offline contract is
        unchanged. ``build_default`` runs BEFORE the lifespan sets the server context, so
        the durable upgrade happens in the lifespan's rebuild (mirroring the spine).
        """
        from himmy.runtime.checkpoint import (
            InMemoryCheckpointStore,
            SqliteCheckpointStore,
        )
        from himmy.services.storage.factory import in_server_context

        if in_server_context():
            # K3: under a Postgres DSN this is the tenant-scoped Postgres mirror bound to
            # the ``"v1"`` tenant — DELIBERATELY distinct from Studio's ``"studio"`` tenant
            # so /v1 and Studio checkpoints stay isolated (the v1_approvals.db/approvals.db
            # split is preserved AS a discriminator, not collapsed). SQLite file-backed
            # otherwise; in-memory off-server.
            from himmy.config.project import v1_approvals_db_path
            from himmy.services.storage.aux_store_factory import select_aux_store

            def _pg() -> Any:
                from himmy.services.storage.postgres_aux import PostgresCheckpointStore

                return PostgresCheckpointStore(tenant="v1")

            return select_aux_store(
                lambda: SqliteCheckpointStore(v1_approvals_db_path()), _pg
            )
        return InMemoryCheckpointStore()

    @staticmethod
    def _graph_checkpoint_store_provider() -> Any:
        """The durable graph checkpoint store an orchestration HITL resume reopens.

        Delegates to the SAME path-keyed singleton the team/workflow run routers pass into
        ``create_orchestration_run`` (``teams_store.get_graph_checkpoint_store``), so an
        approval-resume reopens the exact db the run paused into (file-backed in a server,
        in-RAM offline). Wired as a lazy getter so a path/env change is honoured.
        """
        from himmy.api.teams_store import get_graph_checkpoint_store

        return get_graph_checkpoint_store()

    @staticmethod
    def _resolve_conversation_store() -> Any:
        """Resolve the live conversation store for the S4 reach map (or ``None`` offline).

        Only the SHARED durable singleton (a server deployment) is a real erasure target: a
        zero-config offline run uses a private in-memory store that holds no persisted PII, so
        it returns ``None`` (the reach step is skipped). This is read at erase time, so it sees
        the singleton the lifespan rebound to — not whatever existed when the overlay was built.
        """
        from himmy.services.storage.conversations import current_conversation_store
        from himmy.services.storage.factory import in_server_context

        if not in_server_context():
            return None
        return current_conversation_store()

    @staticmethod
    def _resolve_memory_store() -> Any:
        """Resolve the live memory module store for the S4 reach map (or ``None`` offline).

        Mirrors :meth:`_resolve_conversation_store`: in a server context this is the durable
        Studio/agent memory singleton (``.himmy/memory.db`` or the K5 Postgres mirror); the
        offline default writes no durable memory sidecar, so it returns ``None``.
        """
        from himmy.services.storage.factory import in_server_context

        if not in_server_context():
            return None
        from himmy.api.studio_memory import current_memory_store

        return current_memory_store()

    @staticmethod
    def _build_conversation_store() -> Any:
        """Build the conversation store: shared durable singleton in a server, RAM offline.

        In a server context (the lifespan-established durable deployment) this resolves
        the process-wide :func:`get_conversation_store` singleton at the canonical
        ``.himmy/conversations.db`` — the SAME database the CLI (``himmy chat``) and Studio
        Chats read, so a /v1 thread is browsable everywhere (T2g acceptance). Outside a
        server context (the zero-config ``build_default`` offline/test default) it is a
        private in-memory :class:`ConversationStore` so no file is written and the offline
        contract is unchanged. ``build_default`` runs BEFORE the lifespan sets the server
        context, so the durable upgrade happens in the lifespan's rebuild (mirroring the
        spine + the checkpoint store).
        """
        from himmy.services.storage.conversations import ConversationStore
        from himmy.services.storage.factory import in_server_context

        if in_server_context():
            from himmy.services.storage.conversations import get_conversation_store

            return get_conversation_store()
        return ConversationStore(":memory:")

    @staticmethod
    def _build_knowledge(storage: Any) -> Any:
        """Build the workspace-namespaced knowledge service (T3a) over the wired storage.

        Reuses the container's ``storage`` handle (the durable backend in a server
        context, the in-memory store offline) and the env-configured embedder so the
        KB's vector dimension matches the embedder that is actually built. The
        :class:`KnowledgeAppService` threads every operation through the single
        :func:`~himmy.services.knowledge.app_service.knowledge_namespace` choke point.
        Constructing it runs no I/O, so the zero-config offline path is unchanged.
        """
        from himmy.services.knowledge.app_service import KnowledgeAppService
        from himmy.services.knowledge.service import KnowledgeBase
        from himmy.toolkit.config import ToolkitConfig

        embedder, dim = ToolkitConfig.from_env().build_embedder_and_dim()
        kb = KnowledgeBase(storage=storage, embedder=embedder)
        return KnowledgeAppService(kb, vector_dim=dim)

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

    @classmethod
    def _governed_overlay(cls, storage: Any, registry: Any) -> _GovernedOverlay:
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

        from himmy.config.project import keyvault_db_path
        from himmy.services.audit.log import SecurityAuditLog
        from himmy.services.governance.consent_ledger import ConsentLedger
        from himmy.services.governance.consent_registry import ConsentAwareRegistry
        from himmy.services.governance.consent_storage import ConsentGatedStorage
        from himmy.services.governance.retention import (
            RetentionService,
            SubjectKeyVault,
            SubjectReachMap,
        )

        # The audit + ledger + erasure services all write to the INNER spine so their
        # records (security_event / consent / erasure_tombstone) are never gated. The key
        # vault is DURABLE (S2): it persists each subject's KEK at the canonical
        # .himmy/keyvault.db so a crypto-shred survives a restart (the in-RAM vault made
        # erase_subject report success against an already-volatile key). keyvault.db shares
        # spine.db's backup/residency posture — destroying it IS whole-system erasure.
        audit = SecurityAuditLog(registry)
        key_vault = SubjectKeyVault(keyvault_db_path())
        # S4: the reach map enumerates every MUTABLE sidecar a subject lives in so
        # erase_subject hard-deletes them (the append-only spine stays as undecryptable
        # ciphertext). The handles resolve LAZILY to the same process-wide durable singletons
        # the server's memory + conversation surfaces use — built AFTER this overlay, so a
        # lazy callable avoids a construction-order trap while still reaching the live stores.
        reach = SubjectReachMap(
            memory_store=_LazyStoreProxy(cls._resolve_memory_store),
            conversation_store=_LazyStoreProxy(cls._resolve_conversation_store),
            # S3 must_fix #1: the runtime persists every governed run's full transcript +
            # event stream into storage.db chat_threads/run_events ENCRYPTED UNDER THE
            # STORE-WIDE KEK (consent_storage.ConsentGatedStorage delegates save_thread /
            # append_event straight through to the inner backend — they are NOT per-subject
            # keyed). A crypto-shred of the subject key therefore leaves them recoverable, so
            # erasure HARD-DELETES them here. ``storage`` is that inner backend.
            spine_eraser=_spine_eraser(storage),
            knowledge_eraser=None,
        )
        retention = RetentionService(
            registry, key_vault=key_vault, reach_map=reach
        )
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
            key_vault=key_vault,
            audit=audit,
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
        """Close any resources the container owns (storage pool + durable spine).

        The durable spine (SQLite or the opt-in Postgres facade) owns a file handle /
        connection pool; flushing + closing it here releases the handle and persists any
        buffered audit writes. Its ``close`` is synchronous on every backend, so it is run
        without awaiting (the SQLite ``flush`` and the facade's loop teardown are sync).
        """
        closer = getattr(self.storage, "close", None)
        if closer is not None:
            try:
                await closer()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        spine_close = getattr(self.entity_registry, "close", None)
        if spine_close is not None:
            try:
                spine_close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        # T2f: release /v1's durable HITL checkpoint file handle (SQLite). The in-memory
        # store has no ``close`` — the getattr guard makes this a no-op there.
        cp_close = getattr(self.checkpoint_store, "close", None)
        if cp_close is not None:
            try:
                cp_close()
            except Exception:  # pragma: no cover - best-effort teardown
                pass
        # T2g: close the conversation store ONLY when it is the container-owned (offline,
        # in-memory) instance. The durable server store is the process-wide singleton the
        # CLI + Studio share, so closing it here would break their connection — leave it
        # to its own lifecycle. Identity against the singleton (without forcing it to
        # construct) distinguishes the two.
        from himmy.services.storage.conversations import current_conversation_store

        conv = self.conversation_store
        if conv is not None and conv is not current_conversation_store():
            conv_close = getattr(conv, "close", None)
            if conv_close is not None:
                try:
                    conv_close()
                except Exception:  # pragma: no cover - best-effort teardown
                    pass


__all__ = ["ApiContainer"]
