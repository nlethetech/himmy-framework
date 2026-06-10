"""Tests for the typed TaskContext boundary contract on the runtime entry points.

``task.context`` threads through the whole runtime as a plain dict, so before the
:class:`~himmy.runtime.single_agent.TaskContext` model existed a malformed value
for a recognized key (``tool_names`` as a bare string, ``compaction_spec`` as a
non-mapping, ``objectives`` as a string that iterates as characters) surfaced as
a confusing mid-run failure — or worse, mis-ran silently. These tests lock the
new contract: every public entry point (``run_task`` / ``run_agent_loop`` /
``continue_turn`` / ``reinject_system_prompt`` / ``stream_task`` /
``stream_agent_loop`` / ``resume_agent_loop``) rejects a malformed context with
a clear :class:`HimmyError` BEFORE any model turn runs, a valid context behaves
exactly as before, and unknown extra keys still pass through untouched.

Offline-first: everything runs on the StubClientManager with no provider/DB.
"""

from __future__ import annotations

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread, MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.runtime import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
    SingleAgentRuntime,
)
from himmy.runtime.single_agent import TaskContext
from himmy.services.inference.client_manager import StubClientManager
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.service import ToolService
from tests.conftest import run_async


# --------------------------------------------------------------- local helpers
class _CountingManager(StubClientManager):
    """A stub manager that counts generate() calls (proves no turn ever ran)."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def generate(self, request):  # noqa: ANN001, ANN202
        self.calls += 1
        return await super().generate(request)


def _runtime(**overrides):
    storage = StorageService()
    manager = overrides.pop("manager", None) or StubClientManager()
    rt = SingleAgentRuntime(
        inference_service=InferenceService(manager, event_sink=storage),
        memory_store=storage,
        **overrides,
    )
    return rt, storage


# ----------------------------------------------------- malformed ctx: rejected
def test_run_task_rejects_string_tool_names() -> None:
    """tool_names as a bare string fails at entry, not as a per-char tool lookup."""
    rt, _storage = _runtime()
    persona = Persona(name="A")
    task = Task(title="t", prompt="p", context={"tool_names": "calculator"})
    with pytest.raises(HimmyError) as exc_info:
        run_async(rt.run_task(persona, task))
    msg = str(exc_info.value)
    assert "run_task" in msg
    assert "invalid task context" in msg
    assert "tool_names" in msg


def test_run_task_rejects_non_mapping_compaction_spec() -> None:
    """compaction_spec must be a mapping; a string used to AttributeError mid-loop."""
    rt, _storage = _runtime()
    persona = Persona(name="A")
    task = Task(title="t", prompt="p", context={"compaction_spec": "huge"})
    with pytest.raises(HimmyError, match="compaction_spec"):
        run_async(rt.run_task(persona, task))


def test_run_task_rejects_string_objectives() -> None:
    """objectives as a string used to silently render one objective PER CHARACTER."""
    rt, _storage = _runtime()
    persona = Persona(name="A")
    task = Task(title="t", prompt="p", context={"objectives": "be helpful"})
    with pytest.raises(HimmyError, match="objectives"):
        run_async(rt.run_task(persona, task))


def test_run_agent_loop_rejects_malformed_ctx_before_any_turn() -> None:
    """The loop entry validates BEFORE the first turn — no inference call is made."""
    manager = _CountingManager()
    rt, _storage = _runtime(manager=manager)
    persona = Persona(name="A")
    task = Task(title="t", prompt="p", context={"tool_names": 42})
    with pytest.raises(HimmyError, match="run_agent_loop.*invalid task context"):
        run_async(rt.run_agent_loop(persona, task, max_turns=3))
    assert manager.calls == 0


def test_continue_turn_rejects_malformed_task_context() -> None:
    """continue_turn validates its task_context knobs at entry."""
    rt, _storage = _runtime()
    persona = Persona(name="A")
    thread = run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    with pytest.raises(HimmyError, match="continue_turn.*invalid task context"):
        run_async(
            rt.continue_turn(persona, thread, task_context={"model_key": ["fast"]})
        )


def test_reinject_system_prompt_rejects_malformed_task_context() -> None:
    """reinject_system_prompt validates too — skills as a string is rejected."""
    rt, _storage = _runtime()
    persona = Persona(name="A")
    thread = run_async(rt.run_task(persona, Task(title="t", prompt="p")))
    with pytest.raises(HimmyError, match="reinject_system_prompt"):
        run_async(
            rt.reinject_system_prompt(
                Persona(name="B"), thread, task_context={"skills": "ninja"}
            )
        )


def test_stream_task_rejects_malformed_ctx() -> None:
    """The streaming single-turn path enforces the same boundary contract."""
    rt, _storage = _runtime()
    persona = Persona(name="A")
    task = Task(title="t", prompt="p", context={"output_schema": "not-a-schema"})

    async def _consume() -> None:
        async for _delta in rt.stream_task(persona, task):
            pass  # pragma: no cover - validation raises before any delta

    with pytest.raises(HimmyError, match="stream_task.*invalid task context"):
        run_async(_consume())


def test_stream_agent_loop_rejects_malformed_ctx_before_any_turn() -> None:
    """The streamed loop validates before its first turn — no inference call."""
    manager = _CountingManager()
    rt, _storage = _runtime(manager=manager)
    persona = Persona(name="A")
    task = Task(title="t", prompt="p", context={"skill_routing_hints": {"a": 1}})

    async def _consume() -> None:
        async for _delta in rt.stream_agent_loop(persona, task, max_turns=3):
            pass  # pragma: no cover - validation raises before any delta

    with pytest.raises(HimmyError, match="stream_agent_loop"):
        run_async(_consume())
    assert manager.calls == 0


def test_resume_agent_loop_rejects_tampered_checkpoint_ctx() -> None:
    """A hand-crafted/tampered checkpoint ctx is rejected on resume (like max_turns)."""
    store = InMemoryCheckpointStore()
    rt, _storage = _runtime(checkpoint_store=store)
    persona = Persona(name="A")
    task = Task(title="t", prompt="p")
    thread = ChatThread(agent_id=persona.agent_id)
    checkpoint = AgentCheckpoint(
        persona=persona.model_dump(),
        task=task.model_dump(),
        thread=thread.model_dump(),
        ctx={"tool_names": "wire_money"},  # tampered: must be a list of names
    )
    store.save(checkpoint)
    with pytest.raises(HimmyError, match="resume_agent_loop.*invalid task context"):
        run_async(rt.resume_agent_loop(checkpoint.checkpoint_id, approved=False))


# --------------------------------------------------- valid ctx: unchanged
def test_valid_ctx_runs_unchanged() -> None:
    """A fully-populated VALID context runs exactly as before the contract landed."""
    storage = StorageService()
    tool_registry = ToolRegistry()

    def get_price(args: dict) -> dict:
        return {"price": 42}

    register_local_tool(
        tool_registry,
        name="get_price",
        handler=get_price,
        args_json_schema={"type": "object", "properties": {}},
    )
    rt = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager(), event_sink=storage),
        memory_store=storage,
        tool_service=ToolService(tool_registry, event_sink=storage),
    )
    persona = Persona(name="Analyst", description="careful")
    task = Task(
        title="brief",
        prompt="What is the price?",
        context={
            "model_key": "default",
            "tool_names": ["get_price"],
            "response_format": "AUTO_TOOLS",
            "objectives": ["answer with the price"],
            "skills": ["pricing"],
            "datetime": "2026-06-09",
            "output_format": "one line",
            "system_prefix": "PREFIX-MARKER",
        },
    )
    thread = run_async(rt.run_task(persona, task))

    roles = [m.role for m in thread.messages]
    assert MessageRole.SYSTEM in roles
    assert MessageRole.TOOL in roles  # the tool actually ran
    assert thread.last_message is not None
    assert thread.last_message.role == MessageRole.ASSISTANT
    system = next(m for m in thread.messages if m.role == MessageRole.SYSTEM)
    assert system.content.startswith("PREFIX-MARKER")


def test_unknown_extra_keys_pass_through() -> None:
    """Unrecognized application-level keys are allowed and untouched (extra='allow')."""
    rt, _storage = _runtime()
    persona = Persona(name="A")
    task = Task(
        title="t",
        prompt="p",
        context={"my_app_key": {"nested": True}, "another": 7},
    )
    thread = run_async(rt.run_task(persona, task))
    assert thread.last_message is not None
    assert thread.last_message.role == MessageRole.ASSISTANT
    # The model itself preserves the extras (the boundary contract is not lossy).
    model = TaskContext.model_validate(task.context)
    assert model.model_extra == {"my_app_key": {"nested": True}, "another": 7}


def test_task_context_model_types_known_fields() -> None:
    """The model itself: known fields are typed, None defaults keep it zero-config."""
    blank = TaskContext()
    assert blank.tool_names is None
    assert blank.model_key is None
    ok = TaskContext.model_validate(
        {"tool_names": ("a", "b"), "compaction_spec": {"max_tokens": 100}}
    )
    assert ok.tool_names == ["a", "b"]  # lax coercion: tuple -> list is fine
    with pytest.raises(Exception, match="tool_names"):
        TaskContext.model_validate({"tool_names": "a"})
