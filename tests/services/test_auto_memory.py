"""Tests for auto-memory: recalled memories inject into the prompt with no tool call."""

from __future__ import annotations

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.config.agent_spec import AgentSpec
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.context.service import ContextService
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.memory import (
    InMemoryMemoryStore,
    MemoryContextAdapter,
    MemoryService,
)
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def test_make_task_wires_memory_context_spec() -> None:
    """AgentSpec.memory makes make_task declare the memory build + prompt-map specs."""
    spec = AgentSpec(name="a", memory=True)
    task = spec.make_task("recall ducks")
    bs = task.context["context_build_spec"]
    assert bs["keys"][0]["adapter_name"] == "memory"
    assert bs["keys"][0]["metadata"]["query"] == "recall ducks"
    assert task.context["context_prompt_map_spec"]["system_keys"] == ["agent_memory"]
    # default: no memory spec
    assert "context_build_spec" not in AgentSpec(name="b").make_task("x").context


def test_recalled_memory_injected_into_system_prompt() -> None:
    """A memory-wired runtime auto-injects a recalled fact (no tool call)."""
    mem = MemoryService(InMemoryMemoryStore())
    mem.remember("The user farms ducks and bees in Bardaghat", subject_id="boss")
    storage = StorageService()
    ctx = ContextService(
        storage_service=storage,
        adapters=[MemoryContextAdapter(mem, subject_id="boss")],
    )
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
        context_service=ctx,
        memory_store=storage,
    )
    task = AgentSpec(name="a", memory=True).make_task("ducks and bees in Bardaghat")
    result = run_async(runtime.run_task_detailed(Persona(name="a"), task))

    system = [m.content for m in result.thread.messages if m.role.value == "system"]
    assert any("ducks and bees" in s for s in system)
    # ...and it arrived via context, not a tool call.
    assert not result.tool_calls


def test_no_memory_no_context_block() -> None:
    """Without memory enabled, no context block is injected."""
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager())
    )
    task = Task(title="t", prompt="hello")
    result = run_async(runtime.run_task_detailed(Persona(name="a"), task))
    system = [m.content for m in result.thread.messages if m.role.value == "system"]
    assert not any("<context" in s for s in system)
