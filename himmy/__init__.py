"""Himmy — an offline-first, entity-backed Python agent framework.

Top-level convenience re-exports the most common agent primitives. Heavier kernels
(inference, runtime, api, ...) are imported from their own subpackages so the core
import stays light and fully offline.
"""

from __future__ import annotations

from typing import Any

from himmy.agents.base_agent.agent import Agent
from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Persona",
    "Task",
    "Agent",
    "AgentSpec",
    "load_agent_spec",
    "ToolkitConfig",
    "register_packs",
    "build_embedder",
    "AgentTeam",
    "TeamMember",
    "MultiAgentOrchestrator",
    "MemoryService",
    "AgentEvalHarness",
    "build_guardrail_pipeline",
    "build_runtime",
    "build_inference",
    "build_storage",
]

# The full-stack builder pulls in the heavier runtime/inference/tools kernels, so
# it is exposed lazily (PEP 562) to keep ``import himmy`` light and offline. The
# declarative spec loader (himmy.config) is lazy for the same reason.
_LAZY = {"build_runtime", "build_inference", "build_storage"}
_LAZY_CONFIG = {"AgentSpec", "load_agent_spec"}
_LAZY_TOOLKIT = {"ToolkitConfig", "register_packs"}
_LAZY_ORCH = {"AgentTeam", "TeamMember", "MultiAgentOrchestrator"}


def __getattr__(name: str) -> Any:
    """Lazily resolve the builder + config + toolkit facades from subpackages."""
    if name in _LAZY:
        from himmy.runtime import builder

        return getattr(builder, name)
    if name in _LAZY_ORCH:
        from himmy import orchestrators

        return getattr(orchestrators, name)
    if name == "MemoryService":
        from himmy.services.memory import MemoryService

        return MemoryService
    if name == "build_embedder":
        from himmy.services.knowledge.local_embedders import build_embedder

        return build_embedder
    if name == "AgentEvalHarness":
        from himmy.services.evaluation.agent_harness import AgentEvalHarness

        return AgentEvalHarness
    if name == "build_guardrail_pipeline":
        from himmy.services.guardrails import build_guardrail_pipeline

        return build_guardrail_pipeline
    if name in _LAZY_CONFIG:
        from himmy import config

        return getattr(config, name)
    if name in _LAZY_TOOLKIT:
        from himmy import toolkit

        return getattr(toolkit, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
