"""Chaos: inference managers that die mid-stream (provider disconnects).

The inference unit suite streams happy paths; these tests inject genuine
mid-stream death and lock the contract around it: a provider raising mid-stream
surfaces promptly (no hang) with the partial deltas already delivered and the
provider generator finalized; a stream that simply *truncates* (disconnect ends
the iterator with no final response) materializes a typed FAILED done-delta
instead of raising; and the runtime's streaming surface propagates the failure
cleanly with no orphan tasks — including when the *client* drops the stream.
Every awaited scenario is guarded by ``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.core.events import EventType, RunEvent
from himmy.runtime.single_agent import SingleAgentRuntime
from himmy.services.inference.models import (
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)
from himmy.services.inference.service import InferenceService, StreamDelta
from tests.conftest import run_async

#: Hard guard on every awaited scenario: a hang fails fast, not forever.
_GUARD_SECONDS = 5.0


class _SinkRecorder:
    """A minimal EventSink capturing every emitted event in order."""

    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    async def append_event(self, event: RunEvent) -> None:
        self.events.append(event)


class _DyingStreamManager:
    """A manager whose stream yields two deltas then dies mid-stream."""

    provider_name = "chaos"

    def __init__(self) -> None:
        self.stream_finalized = False

    def resolve(self, model_key: str) -> str:
        return f"chaos:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise ConnectionError("provider connection dropped")

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str | InferenceResponse]:
        try:
            yield "Hello "
            yield "wor"
            raise ConnectionError("provider connection dropped mid-stream")
        finally:
            self.stream_finalized = True


class _TruncatedStreamManager:
    """A manager whose stream ends abruptly with NO final response (disconnect)."""

    provider_name = "chaos"

    def resolve(self, model_key: str) -> str:
        return f"chaos:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        # The re-materialization attempt also finds the provider down.
        raise ConnectionError("connection refused")

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str | InferenceResponse]:
        yield "partial "
        # Iterator ends here: the disconnect truncated the stream pre-response.


class _SlowDripManager:
    """A manager streaming many deltas, instrumented to observe finalization."""

    provider_name = "chaos"

    def __init__(self) -> None:
        self.stream_finalized = False

    def resolve(self, model_key: str) -> str:
        return f"chaos:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text="drip",
        )

    async def generate_stream(
        self, request: InferenceRequest
    ) -> AsyncIterator[str | InferenceResponse]:
        try:
            for i in range(50):
                yield f"chunk{i} "
            yield await self.generate(request)
        finally:
            self.stream_finalized = True


def _req() -> InferenceRequest:
    return InferenceRequest(
        messages=[InferenceMessage(role="user", content="hello world")]
    )


def _orphan_tasks() -> list[asyncio.Task[Any]]:
    """Every still-pending task on the loop other than the current one."""
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


def test_mid_stream_death_surfaces_promptly_with_partial_deltas() -> None:
    """A provider raising mid-stream propagates fast; partials delivered, no orphans."""
    manager = _DyingStreamManager()
    service = InferenceService(manager, max_retries=0)

    async def _scenario() -> tuple[list[StreamDelta], list[asyncio.Task[Any]]]:
        deltas: list[StreamDelta] = []
        with pytest.raises(ConnectionError):
            async for delta in service.run_stream(_req()):
                deltas.append(delta)
        return deltas, _orphan_tasks()

    deltas, orphans = run_async(asyncio.wait_for(_scenario(), timeout=_GUARD_SECONDS))
    # The deltas streamed before the death were delivered, none marked done.
    assert [d.delta for d in deltas] == ["Hello ", "wor"]
    assert all(not d.done for d in deltas)
    # The provider generator was finalized and nothing is left running.
    assert manager.stream_finalized
    assert orphans == []


def test_truncated_stream_materializes_typed_failed_response() -> None:
    """A stream ending with no final response yields a typed FAILED done-delta.

    The disconnect must NOT hang waiting for more chunks and must NOT raise out
    of the stream: the service re-materializes via ``run`` (which finds the
    provider down too) and the terminal delta carries a typed, retryable
    ``PROVIDER_UNAVAILABLE`` failure plus the ``INFERENCE_FAILED`` event.
    """
    sink = _SinkRecorder()
    service = InferenceService(
        _TruncatedStreamManager(),
        max_retries=0,
        event_sink=sink,
    )

    async def _scenario() -> list[StreamDelta]:
        return [d async for d in service.run_stream(_req())]

    deltas = run_async(asyncio.wait_for(_scenario(), timeout=_GUARD_SECONDS))
    assert [d.delta for d in deltas[:-1]] == ["partial "]
    final = deltas[-1]
    assert final.done
    assert final.response is not None
    assert final.response.status == InferenceStatus.FAILED
    assert final.response.error is not None
    assert final.response.error.code == InferenceErrorCode.PROVIDER_UNAVAILABLE
    assert final.response.error.retryable
    # The failure was observable through the event sink, not silent.
    assert EventType.INFERENCE_FAILED in [e.event_type for e in sink.events]


def test_runtime_stream_task_mid_stream_death_no_hang_no_orphans() -> None:
    """The runtime streaming surface propagates a mid-stream death cleanly."""
    manager = _DyingStreamManager()
    rt = SingleAgentRuntime(inference_service=InferenceService(manager, max_retries=0))
    persona = Persona(name="A")
    task = Task(title="t", prompt="stream me")

    async def _scenario() -> tuple[list[StreamDelta], list[asyncio.Task[Any]]]:
        received: list[StreamDelta] = []
        with pytest.raises(ConnectionError):
            async for delta in rt.stream_task(persona, task):
                received.append(delta)
        return received, _orphan_tasks()

    received, orphans = run_async(asyncio.wait_for(_scenario(), timeout=_GUARD_SECONDS))
    assert [d.delta for d in received] == ["Hello ", "wor"]
    assert manager.stream_finalized
    assert orphans == []


def test_client_disconnect_mid_stream_finalizes_provider_stream() -> None:
    """A client dropping the runtime stream early leaves no dangling provider stream.

    The consumer reads one delta and closes the generator (the SSE-disconnect
    shape). By the time the event loop exits, the provider's stream generator
    must have been finalized (its ``finally`` ran) — not left suspended.
    """
    manager = _SlowDripManager()
    rt = SingleAgentRuntime(inference_service=InferenceService(manager, max_retries=0))
    persona = Persona(name="A")
    task = Task(title="t", prompt="stream me")

    async def _scenario() -> list[asyncio.Task[Any]]:
        stream = rt.stream_task(persona, task)
        async for _delta in stream:
            break  # client drops after the first chunk
        await stream.aclose()
        return _orphan_tasks()

    orphans = run_async(asyncio.wait_for(_scenario(), timeout=_GUARD_SECONDS))
    # asyncio.run's async-generator shutdown has completed by now: the provider
    # stream's finally must have executed and no task is left behind.
    assert manager.stream_finalized
    assert orphans == []
