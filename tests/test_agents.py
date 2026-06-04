"""Tests for the agents kernel: personas, tasks, threads, messages, agents."""

from __future__ import annotations

from himmy.agents import (
    Agent,
    ChatThread,
    Message,
    MessageRole,
    Persona,
    RolePersona,
    Task,
)


def test_persona_role_prefers_metadata_then_name() -> None:
    """``role`` returns metadata['role'] when set, else the persona name."""
    p = Persona(name="Bob")
    assert p.role == "Bob"
    p2 = Persona(name="Bob", metadata={"role": "analyst"})
    assert p2.role == "analyst"


def test_persona_has_default_agent_id() -> None:
    """A persona gets a generated agent_id by default."""
    p = Persona(name="Bob")
    assert isinstance(p.agent_id, str) and p.agent_id


def test_role_persona_adds_fields() -> None:
    """RolePersona extends Persona with objectives/required tools+skills."""
    rp = RolePersona(
        name="Lead",
        objectives=["win"],
        required_tools=["search"],
        required_skills=["python"],
    )
    assert rp.objectives == ["win"]
    assert rp.required_tools == ["search"]
    assert rp.required_skills == ["python"]
    # It is still a Persona (role property works).
    assert rp.role == "Lead"


def test_agent_from_persona_defaults_instructions() -> None:
    """Agent.from_persona inherits persona instructions when none provided."""
    p = Persona(name="Bob", instructions=["a", "b"])
    agent = Agent.from_persona(p, name="bound")
    assert agent.instructions == ["a", "b"]
    assert agent.persona is p
    agent2 = Agent.from_persona(p, name="bound2", instructions=["x"])
    assert agent2.instructions == ["x"]


def test_chat_thread_append_and_last_message() -> None:
    """append_message stores the message; last_message returns the latest."""
    thread = ChatThread()
    assert thread.last_message is None
    m1 = Message(role=MessageRole.USER, content="hello")
    thread.append_message(m1)
    m2 = Message(role=MessageRole.ASSISTANT, content="hi")
    thread.append_message(m2)
    assert thread.last_message is m2
    assert len(thread.messages) == 2


def test_message_role_enum_values() -> None:
    """MessageRole has the four documented members."""
    assert MessageRole.SYSTEM.value == "system"
    assert MessageRole.USER.value == "user"
    assert MessageRole.ASSISTANT.value == "assistant"
    assert MessageRole.TOOL.value == "tool"


def test_task_carries_context_and_constraints() -> None:
    """Task holds the prompt plus structured context/constraints."""
    t = Task(
        title="t",
        prompt="p",
        context={"model_key": "fast"},
        constraints={"max_tokens": 100},
    )
    assert t.context["model_key"] == "fast"
    assert t.constraints["max_tokens"] == 100
