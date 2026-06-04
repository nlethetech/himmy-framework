"""Tests for the public build_runtime facade (himmy.build_runtime)."""

from __future__ import annotations

import himmy
from himmy import build_inference, build_runtime, build_storage
from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from tests.conftest import run_async


def test_facade_is_exported_from_top_level() -> None:
    """The builder is reachable as a top-level attribute and listed in __all__."""
    assert callable(himmy.build_runtime)
    assert {"build_runtime", "build_inference", "build_storage"} <= set(himmy.__all__)


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
