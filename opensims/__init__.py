"""OpenSims — an offline-first, entity-backed Python agent framework.

Top-level convenience re-exports the most common agent primitives. Heavier kernels
(inference, runtime, api, ...) are imported from their own subpackages so the core
import stays light and fully offline.
"""

from __future__ import annotations

from opensims.agents.base_agent.agent import Agent
from opensims.agents.base_agent.task import Task
from opensims.agents.personas.persona import Persona

__version__ = "0.1.0"

__all__ = ["__version__", "Persona", "Task", "Agent"]
