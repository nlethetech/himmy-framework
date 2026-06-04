"""Inference kernel: provider-agnostic model calls with typed envelopes.

``InferenceService`` is the single async entry point for inference. The default
path is the deterministic, offline :class:`StubClientManager`; pydantic-ai and
the gateway are optional extras behind lazy imports.
"""

from __future__ import annotations

from opensims.services.inference.cache import (
    InferenceCache,
    InMemoryTTLCache,
    NoopInferenceCache,
    compute_cache_key,
)
from opensims.services.inference.client_manager import (
    ClientManager,
    GatewayClientManager,
    StubClientManager,
)
from opensims.services.inference.models import (
    BatchInferenceRequest,
    BatchInferenceResponse,
    BoundTool,
    GatewayModelConfig,
    GatewayRuntimeConfig,
    InferenceError,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
    LLMConfig,
    ModelPrice,
    ResponseFormat,
    ToolCallRecord,
    ToolReturnRecord,
    WorkflowDefinition,
    WorkflowState,
    synthesize_from_schema,
)
from opensims.services.inference.pydantic_ai_manager import (
    PydanticAIClientManager,
)
from opensims.services.inference.service import InferenceService, StreamDelta

__all__ = [
    "InferenceService",
    "InferenceRequest",
    "InferenceResponse",
    "InferenceMessage",
    "InferenceStatus",
    "ResponseFormat",
    "LLMConfig",
    "InferenceError",
    "InferenceErrorCode",
    "ToolCallRecord",
    "ToolReturnRecord",
    "BoundTool",
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "WorkflowDefinition",
    "WorkflowState",
    "StubClientManager",
    "GatewayClientManager",
    "GatewayModelConfig",
    "GatewayRuntimeConfig",
    "PydanticAIClientManager",
    "ClientManager",
    "synthesize_from_schema",
    "ModelPrice",
    "StreamDelta",
    "InferenceCache",
    "NoopInferenceCache",
    "InMemoryTTLCache",
    "compute_cache_key",
]
