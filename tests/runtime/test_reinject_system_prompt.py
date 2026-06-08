"""Runtime-level tests for ``SingleAgentRuntime.reinject_system_prompt``.

This is the seam the multi-agent handoff / group-chat use to swap the active persona's
system prompt on a SHARED thread. It must: replace (not append) the leading SYSTEM
message, render the NEW persona's identity + instructions, register the swap into the
audit spine, bump the thread version, and be a no-op when the prompt already matches.
"""

from __future__ import annotations

import asyncio

from himmy import build_runtime
from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona


def _system(thread: object) -> str:
    msgs = [m for m in thread.messages if m.role == MessageRole.SYSTEM]  # type: ignore[attr-defined]
    return msgs[0].content if msgs else ""


def test_reinject_replaces_system_prompt_in_place() -> None:
    """Re-injecting a new persona REPLACES the single SYSTEM message in place."""
    runtime, _inf, _tools = build_runtime()
    a = Persona(name="alpha", description="ALPHA_DESK", instructions=["be terse"])
    b = Persona(name="beta", description="BETA_DESK", instructions=["be thorough"])

    thread = asyncio.run(
        runtime.run_task_detailed(a, Task(title="t", prompt="hi"))
    ).thread
    assert "alpha agent" in _system(thread)

    asyncio.run(runtime.reinject_system_prompt(b, thread))
    system = _system(thread)
    # The NEW persona is now in effect; the OLD one is gone, and there is only one.
    assert "beta agent" in system
    assert "BETA_DESK" in system
    assert "be thorough" in system
    assert "alpha agent" not in system
    assert sum(1 for m in thread.messages if m.role == MessageRole.SYSTEM) == 1


def test_reinject_bumps_version_and_audits() -> None:
    """A persona swap bumps the thread version and projects a record into the spine."""
    runtime, _inf, _tools = build_runtime()
    a = Persona(name="alpha", description="ALPHA_DESK")
    b = Persona(name="beta", description="BETA_DESK")

    thread = asyncio.run(
        runtime.run_task_detailed(a, Task(title="t", prompt="hi"))
    ).thread
    before = thread.version
    asyncio.run(runtime.reinject_system_prompt(b, thread))
    assert thread.version == before + 1
    # The new SYSTEM message carries the persona provenance metadata.
    system_msg = next(m for m in thread.messages if m.role == MessageRole.SYSTEM)
    assert system_msg.metadata.get("persona") == "beta"
    assert system_msg.metadata.get("agent_id") == b.agent_id


def test_reinject_same_persona_is_noop() -> None:
    """Re-injecting the SAME persona's identical prompt does not bump the version."""
    runtime, _inf, _tools = build_runtime()
    a = Persona(name="alpha", description="ALPHA_DESK")
    thread = asyncio.run(
        runtime.run_task_detailed(a, Task(title="t", prompt="hi"))
    ).thread
    before = thread.version
    out = asyncio.run(runtime.reinject_system_prompt(a, thread))
    assert thread.version == before  # unchanged: already the right persona
    assert "alpha agent" in out


def test_reinject_inserts_system_when_absent() -> None:
    """When a thread has no SYSTEM message, re-inject inserts one at the head."""
    from himmy.agents.base_agent.thread import ChatThread, Message

    runtime, _inf, _tools = build_runtime()
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="hello"))
    a = Persona(name="alpha", description="ALPHA_DESK")
    asyncio.run(runtime.reinject_system_prompt(a, thread))
    assert thread.messages[0].role == MessageRole.SYSTEM
    assert "alpha agent" in thread.messages[0].content
