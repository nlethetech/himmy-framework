"""Inference kernel: provider-agnostic model calls with typed envelopes.

``InferenceService`` is the single async entry point for inference. The default
path is the deterministic, offline :class:`StubClientManager`; pydantic-ai and
the gateway are optional extras behind lazy imports.
"""

from __future__ import annotations

from himmy.services.inference.anthropic_manager import AnthropicClientManager
from himmy.services.inference.cache import (
    InferenceCache,
    InMemoryTTLCache,
    NoopInferenceCache,
    compute_cache_key,
)
from himmy.services.inference.client_manager import (
    ClientManager,
    GatewayClientManager,
    StubClientManager,
)
from himmy.services.inference.local import (
    ClaudeCliClientManager,
    HimalayaGptClientManager,
    OllamaClientManager,
)
from himmy.services.inference.models import (
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
    ToolExecutor,
    ToolReturnRecord,
    WorkflowDefinition,
    WorkflowState,
    synthesize_from_schema,
)
from himmy.services.inference.openai_manager import OpenAIClientManager
from himmy.services.inference.pydantic_ai_manager import (
    PydanticAIClientManager,
)
from himmy.services.inference.routing import (
    DEFAULT_FAILOVER_CODES,
    Route,
    RoutingClientManager,
)
from himmy.services.inference.service import InferenceService, StreamDelta

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
    "ToolExecutor",
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
    "RoutingClientManager",
    "Route",
    "DEFAULT_FAILOVER_CODES",
    "OllamaClientManager",
    "ClaudeCliClientManager",
    "HimalayaGptClientManager",
    "AnthropicClientManager",
    "OpenAIClientManager",
    "synthesize_from_schema",
    "ModelPrice",
    "StreamDelta",
    "InferenceCache",
    "NoopInferenceCache",
    "InMemoryTTLCache",
    "compute_cache_key",
]
