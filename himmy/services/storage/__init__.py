"""Storage kernel: persistence service, records, and the Postgres scaffold."""

from __future__ import annotations

from himmy.services.storage.inmemory import (
    InMemoryContextStore,
    InMemoryEvaluationStore,
    InMemoryEventLog,
    InMemoryOrchestrationStore,
    InMemoryRecommendationStore,
    InMemoryRunStore,
    InMemoryThreadStore,
)
from himmy.services.storage.models import (
    ActionRecord,
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
    "RecommendationStore",
    "EvaluationStore",
    "OrchestrationStore",
    # In-memory store implementations (composed by StorageService).
    "InMemoryThreadStore",
    "InMemoryEventLog",
    "InMemoryContextStore",
    "InMemoryRunStore",
    "InMemoryRecommendationStore",
    "InMemoryEvaluationStore",
    "InMemoryOrchestrationStore",
    # Records.
    "RunRecord",
    "RunStatus",
    "RecommendationItem",
    "RecommendationStatus",
    "PostgresStorageService",
    "STORAGE_DDL",
    "MemoryObject",
    "EpisodicMemoryObject",
    "AgentStateRecord",
    "ActionRecord",
    "EnvironmentStateRecord",
    "ContextEvidenceRecord",
]
