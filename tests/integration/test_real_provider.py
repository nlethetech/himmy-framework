"""WS5.6 — real-provider integration tests (a regression guard against stub-masking).

These run a tool-using agent against a REAL local model (Ollama) so the offline stub can
never again hide a broken core path: earlier, `himmy run` returned an EMPTY answer for any
tool agent on a real provider because the run ended on the tool *call* before the model
saw the result. This lane catches exactly that class of bug.

Marked ``integration`` and skipped when Ollama isn't reachable (so the normal offline
suite is unaffected); the CI integration lane provides Ollama + a small model.
"""

from __future__ import annotations

import os

import httpx
import pytest

from tests.conftest import run_async

_OLLAMA_URL = os.environ.get("HIMMY_OLLAMA_URL", "http://localhost:11434")
_MODEL = os.environ.get("HIMMY_INTEGRATION_MODEL", "qwen2.5:3b-instruct")


def _ollama_available() -> bool:
    try:
        return httpx.get(f"{_OLLAMA_URL}/api/tags", timeout=3.0).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_available(), reason="Ollama not reachable"),
]


def _runtime_with_tools(packs: list[str]):
    from himmy import build_runtime
    from himmy.cli.provider import build_inference_for
    from himmy.services.tools.registry import ToolRegistry
    from himmy.toolkit import ToolkitConfig, register_packs

    registry = ToolRegistry()
    register_packs(registry, packs, ToolkitConfig())
    inference = build_inference_for("ollama", _MODEL)
    runtime, _inf, _tools = build_runtime(inference=inference, tool_registry=registry)
    return runtime


def test_tool_using_agent_answers_on_a_real_model() -> None:
    """A calculator agent must CALL the tool and return a non-empty, correct answer."""
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    runtime = _runtime_with_tools(["utils"])
    persona = Persona(
        name="calc",
        instructions=["Use the calculator tool for any arithmetic."],
    )
    task = Task(title="t", prompt="What is 17 times 23? Use the calculator tool.")

    loop = run_async(runtime.run_agent_loop(persona, task, max_turns=6))
    answer = (loop.final.output_text or "").strip()

    # The regression: a tool agent on a real model returned an empty answer.
    assert answer, "tool-using agent returned an EMPTY answer (the regression)"
    assert "391" in answer, f"expected 391 in the answer, got: {answer!r}"
    # And it actually used the tool (didn't just do mental math).
    assert any(turn.tool_calls for turn in loop.turns), (
        "the calculator tool was never called"
    )


def test_plain_agent_answers_on_a_real_model() -> None:
    """A no-tool agent still produces a non-empty answer (sanity on the provider path)."""
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona

    runtime = _runtime_with_tools([])
    result = run_async(
        runtime.run_task_detailed(
            Persona(name="assistant", instructions=["Answer in one short sentence."]),
            Task(title="t", prompt="Name the capital of France in one word."),
        )
    )
    assert (result.output_text or "").strip(), "plain agent returned an empty answer"
    assert "paris" in (result.output_text or "").lower()
