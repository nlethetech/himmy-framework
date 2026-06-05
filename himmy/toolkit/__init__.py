"""Himmy toolkit: batteries-included, general-purpose tools for agents.

Five named packs — ``web``, ``files``, ``data``, ``code``, ``utils`` — register
ready-to-use tools onto a :class:`~himmy.services.tools.registry.ToolRegistry`. Wire
them declaratively (``tool_packs: [web, utils]`` in an ``agent.yaml``) or in code::

    from himmy.toolkit import register_packs, ToolkitConfig
    from himmy.services.tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_packs(registry, ["web", "utils"], ToolkitConfig.from_env())
"""

from __future__ import annotations

from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.pack import (
    BUILTIN_PACKS,
    ToolPack,
    UnknownToolPackError,
    register_packs,
    resolve_packs,
)

__all__ = [
    "ToolkitConfig",
    "ToolPack",
    "BUILTIN_PACKS",
    "register_packs",
    "resolve_packs",
    "UnknownToolPackError",
]
