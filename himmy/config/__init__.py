"""Declarative configuration: load agents from files instead of hand-wiring code."""

from __future__ import annotations

from himmy.config.agent_spec import AgentSpec, load_agent_spec

__all__ = ["AgentSpec", "load_agent_spec"]
