"""Agents kernel: the ChatThread / Message append-only conversation primitives."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from himmy.core.ids import new_uuid, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.entities.records import EntityRecord


class MessageRole(str, Enum):
    """The role of a message author within a chat thread."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in a thread. First-class entity (``kind="message"``)."""

    message_id: str = Field(default_factory=new_uuid)
    role: MessageRole
    content: str = ""
    metadata: dict[str, Any] = {}
    created_at: str = Field(default_factory=utc_now_iso)

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this message into its canonical ``EntityRecord``."""
        from himmy.entities.projection import project

        return project(
            self,
            stable_value=self.message_id,
            namespace="message",
            kind="message",
            version=version,
            metadata=metadata,
        )


class ChatThread(BaseModel):
    """An append-only message history. First-class entity (``kind="chat_thread"``).

    Each turn that appends messages produces a new ``chat_thread`` record version.
    """

    thread_id: str = Field(default_factory=new_uuid)
    agent_id: str | None = None
    messages: list[Message] = []
    version: int = 1
    metadata: dict[str, Any] = {}

    def append_message(self, message: Message) -> Message:
        """Append a message and return it."""
        self.messages.append(message)
        return message

    @property
    def last_message(self) -> Message | None:
        """The most recently appended message, or None when empty."""
        return self.messages[-1] if self.messages else None

    def to_record(
        self, version: int | None = None, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this thread into its canonical ``EntityRecord``.

        When ``version`` is None the thread's own ``version`` is used.
        """
        from himmy.entities.projection import project

        return project(
            self,
            stable_value=self.thread_id,
            namespace="chat_thread",
            kind="chat_thread",
            version=self.version if version is None else version,
            metadata=metadata,
        )
