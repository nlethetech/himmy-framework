"""Tests for the public build_runtime facade (opensims.build_runtime)."""

from __future__ import annotations

import opensims
from opensims import build_inference, build_runtime, build_storage
from opensims.agents.base_agent.task import Task
from opensims.agents.personas.persona import Persona
from tests.conftest import run_async


def test_facade_is_exported_from_top_level() -> None:
    """The builder is reachable as a top-level attribute and listed in __all__."""
    assert callable(opensims.build_runtime)
    assert {"build_runtime", "build_inference", "build_storage"} <= set(
        opensims.__all__
    )


def test_build_runtime_one_call_runs_a_task() -> None:
    """build_runtime() wires a working offline runtime in a single call."""
    runtime, inference, tools = build_runtime()
    assert inference is not None
    assert tools is not None
    thread = run_async(
        runtime.run_task(Persona(name="A"), Task(title="t", prompt="hello"))
    )
    assert thread.last_message is not None


def test_build_runtime_honors_overrides() -> None:
    """Injected collaborators are used instead of fresh defaults."""
    storage = build_storage()
    inference = build_inference()
    runtime, used_inference, _tools = build_runtime(
        storage=storage, inference=inference
    )
    assert used_inference is inference
    assert runtime is not None
