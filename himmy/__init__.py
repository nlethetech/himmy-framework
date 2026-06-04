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
    "build_runtime",
    "build_inference",
    "build_storage",
]

# The full-stack builder pulls in the heavier runtime/inference/tools kernels, so
# it is exposed lazily (PEP 562) to keep ``import himmy`` light and offline.
_LAZY = {"build_runtime", "build_inference", "build_storage"}


def __getattr__(name: str) -> Any:
    """Lazily resolve the builder facade from :mod:`himmy.runtime.builder`."""
    if name in _LAZY:
        from himmy.runtime import builder

        return getattr(builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
