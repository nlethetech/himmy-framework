"""Storage kernel: persistence service, records, and the Postgres scaffold."""

from __future__ import annotations

from himmy.services.storage.inmemory import (
    InMemoryAgentDefStore,
    InMemoryContextStore,
    InMemoryEvaluationStore,
    InMemoryEventLog,
    InMemoryOrchestrationStore,
    InMemoryRecommendationStore,
    InMemoryRunStore,
    InMemoryThreadStore,
)
from himmy.services.storage.models import (
    LOCAL_SUBJECT,
    LOCAL_WORKSPACE,
    STUDIO_METADATA_KEY,
    ActionRecord,
    AgentDefRecord,
    AgentStateRecord,
    ContextEvidenceRecord,
    EnvironmentStateRecord,
    EpisodicMemoryObject,
    MemoryObject,
    RecommendationItem,
    RecommendationStatus,
    RunRecord,
    RunStatus,
)
from himmy.services.storage.postgres import (
    STORAGE_DDL,
    PostgresStorageService,
)
from himmy.services.storage.protocols import (
    AgentDefStore,
    ContextStore,
    EvaluationStore,
    EventLog,
    OrchestrationStore,
    RecommendationStore,
    RunStore,
    ThreadEventStore,
    ThreadStore,
)
from himmy.services.storage.service import StorageService

__all__ = [
    "StorageService",
    # Focused store protocols (the per-concern contracts both backends satisfy).
    "ThreadStore",
    "EventLog",
    "ThreadEventStore",
    "ContextStore",
    "RunStore",
    "AgentDefStore",
    "RecommendationStore",
    "EvaluationStore",
    "OrchestrationStore",
    # In-memory store implementations (composed by StorageService).
    "InMemoryThreadStore",
    "InMemoryEventLog",
    "InMemoryContextStore",
    "InMemoryRunStore",
    "InMemoryAgentDefStore",
    "InMemoryRecommendationStore",
    "InMemoryEvaluationStore",
    "InMemoryOrchestrationStore",
    # Records.
    "LOCAL_SUBJECT",
    "LOCAL_WORKSPACE",
    "STUDIO_METADATA_KEY",
    "RunRecord",
    "RunStatus",
    "RecommendationItem",
    "RecommendationStatus",
    "AgentDefRecord",
    "PostgresStorageService",
    "STORAGE_DDL",
    "MemoryObject",
    "EpisodicMemoryObject",
    "AgentStateRecord",
    "ActionRecord",
    "EnvironmentStateRecord",
    "ContextEvidenceRecord",
]
