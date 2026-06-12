"""Streaming answers are subject to the OUTPUT guardrail (parity with run_task).

``stream_task`` used to build and persist the assistant message straight from the raw
provider response with NO ``_guard_output`` call — so a streamed answer skipped the
grounding / PII / DLP / blocklist output guard entirely (and emitted no
GUARDRAIL_APPLIED event). These tests pin the fix:

* GUARD-AFTER (redact / blocklist-free redactors): tokens stream, then the final
  persisted message and the terminal ``done`` payload carry the GUARDED text, and a
  GUARDRAIL_APPLIED(stage=output) event is emitted.
* BUFFER (an output guard that WITHHOLDS blocked content — DLP ``…:block`` /
  blocklist / blocking injection): tokens are NOT streamed (an already-streamed secret
  can't be recalled); the whole guarded answer arrives on the ``done`` delta.

Offline-first: a tiny echo client manager drives the deterministic stream fallback.
"""

from __future__ import annotations

from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import MessageRole
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType
from himmy.runtime import SingleAgentRuntime
from himmy.services.guardrails import (
    BlocklistGuardrail,
    DlpAction,
    DlpGuardrail,
    DlpPolicy,
    GuardrailPipeline,
    PIIGuardrail,
)
from himmy.services.inference.models import InferenceResponse, InferenceStatus
from himmy.services.inference.service import InferenceService
from himmy.services.storage.service import StorageService
from tests.conftest import run_async

_CARD = "4111 1111 1111 1111"


class _EchoManager:
    """Returns ``output_text`` verbatim (no ``generate_stream`` -> offline chunking)."""

    def __init__(self, output_text: str) -> None:
        self._output_text = output_text

    def resolve(self, model_key: str) -> str:
        return f"stub:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=self._output_text,
            input_tokens=1,
            output_tokens=1,
        )


def _runtime(
    *, output_text: str, output_guardrail: GuardrailPipeline
) -> tuple[SingleAgentRuntime, StorageService, list]:
    storage = StorageService()
    events: list = []

    async def collect(event: Any) -> None:
        events.append(event)

    rt = SingleAgentRuntime(
        inference_service=InferenceService(_EchoManager(output_text), event_sink=storage),
        memory_store=storage,
        output_guardrail=output_guardrail,
        on_event=collect,
    )
    return rt, storage, events


def _collect_stream(rt: SingleAgentRuntime, persona: Persona, task: Task) -> dict:
    """Drive stream_task, accumulating streamed text, the done response, and the thread."""
    text_streamed = ""
    final_text: str | None = None
    thread = None

    async def _drive() -> None:
        nonlocal text_streamed, final_text, thread
        async for delta in rt.stream_task(persona, task):
            if delta.delta:
                text_streamed += delta.delta
            if delta.done and delta.response is not None:
                final_text = delta.response.output_text

    run_async(_drive())
    return {"streamed": text_streamed, "final_text": final_text}


def _output_guard_events(events: list) -> list:
    return [
        e
        for e in events
        if e.event_type == EventType.GUARDRAIL_APPLIED
        and e.payload.get("stage") == "output"
    ]


def test_stream_guard_after_redacts_persisted_and_done_payload() -> None:
    """A PII redactor on output: the done payload + persisted msg carry redacted text."""
    guard = GuardrailPipeline([PIIGuardrail()])
    rt, _storage, events = _runtime(
        output_text="reach me at alice@example.com anytime", output_guardrail=guard
    )
    persona = Persona(name="agent")
    task = Task(title="t", prompt="contact?")

    out = _collect_stream(rt, persona, task)

    # The final (done) payload is the GUARDED text — the email is redacted.
    assert out["final_text"] is not None
    assert "alice@example.com" not in out["final_text"]
    # A GUARDRAIL_APPLIED(output) event was emitted (was entirely absent before).
    assert _output_guard_events(events)


def test_stream_buffer_blocks_do_not_stream_the_secret() -> None:
    """A DLP card:block output guard BUFFERS: no token carries the card; done is clean."""
    guard = GuardrailPipeline(
        [DlpGuardrail(policy=DlpPolicy(actions={"card": DlpAction.BLOCK}))]
    )
    rt, _storage, events = _runtime(
        output_text=f"your card on file is {_CARD}", output_guardrail=guard
    )
    persona = Persona(name="agent")
    task = Task(title="t", prompt="my card?")

    out = _collect_stream(rt, persona, task)

    # BUFFER mode: the streamed text never carried the raw card mid-flight.
    assert _CARD not in out["streamed"]
    assert "4111111111111111" not in out["streamed"].replace(" ", "")
    # And the terminal payload is the guarded (card-free) answer.
    assert out["final_text"] is not None
    assert _CARD not in out["final_text"]
    assert _output_guard_events(events)
    assert _output_guard_events(events)[0].payload["blocked"] is True


def test_stream_buffer_blocklist_withholds_phrase() -> None:
    """A blocklist output guard buffers and withholds: the banned phrase never streams."""
    guard = GuardrailPipeline([BlocklistGuardrail(["top-secret-token"])])
    rt, _storage, _events = _runtime(
        output_text="the value is top-secret-token, keep it safe",
        output_guardrail=guard,
    )
    persona = Persona(name="agent")
    task = Task(title="t", prompt="value?")

    out = _collect_stream(rt, persona, task)

    assert "top-secret-token" not in out["streamed"]
    assert out["final_text"] is not None
    assert "top-secret-token" not in out["final_text"]


def test_stream_persisted_message_carries_guarded_text() -> None:
    """The thread's persisted assistant message is the GUARDED answer, not the raw one."""
    guard = GuardrailPipeline([PIIGuardrail()])
    rt, _storage, _events = _runtime(
        output_text="email alice@example.com now", output_guardrail=guard
    )
    persona = Persona(name="agent")
    thread_holder: dict[str, Any] = {}

    async def _drive() -> None:
        thread = None
        from himmy.agents.base_agent.thread import ChatThread

        thread = ChatThread(agent_id=persona.agent_id)
        async for _delta in rt.stream_task(
            persona, Task(title="t", prompt="contact?"), thread
        ):
            pass
        thread_holder["thread"] = thread

    run_async(_drive())
    thread = thread_holder["thread"]
    assistant = [m for m in thread.messages if m.role == MessageRole.ASSISTANT]
    assert assistant and "alice@example.com" not in (assistant[-1].content or "")
    assert assistant[-1].metadata.get("guarded") is True


def test_stream_buffer_mixed_redact_block_does_not_leak() -> None:  # finding #3
    """A redact+block OUTPUT pipeline on the BUFFER stream path must not fall open.

    ``[PIIGuardrail, BlocklistGuardrail]`` redacts the email upstream, then the blocklist
    blocks — so the aggregate text differs from the raw answer even though the blocker
    supplied no replacement. The blocklist makes the path BUFFER (suppresses output
    content), so nothing streams; the runtime must still withhold the banned phrase from
    the buffered/done text instead of falling open on the redaction.
    """
    guard = GuardrailPipeline(
        [PIIGuardrail(), BlocklistGuardrail(["top-secret-token"])]
    )
    rt, _storage, events = _runtime(
        output_text="email alice@example.com the top-secret-token now",
        output_guardrail=guard,
    )
    persona = Persona(name="agent")
    out = _collect_stream(rt, persona, Task(title="t", prompt="value?"))

    # No token carried the banned phrase OR the raw email (BUFFER mode streams nothing
    # until the guard runs), and the terminal payload is the withheld placeholder.
    assert "top-secret-token" not in out["streamed"]
    assert "top-secret-token" not in (out["final_text"] or "")
    assert "alice@example.com" not in (out["final_text"] or "")
    assert _output_guard_events(events)
    assert _output_guard_events(events)[0].payload["blocked"] is True


def test_stream_clean_answer_streams_unchanged() -> None:
    """A clean answer with a redact-only guard streams its tokens unchanged."""
    guard = GuardrailPipeline([PIIGuardrail()])
    rt, _storage, events = _runtime(
        output_text="a perfectly clean streamed answer", output_guardrail=guard
    )
    persona = Persona(name="agent")
    out = _collect_stream(rt, persona, Task(title="t", prompt="answer"))

    assert out["streamed"] == "a perfectly clean streamed answer"
    assert out["final_text"] == "a perfectly clean streamed answer"
    assert _output_guard_events(events) == []
