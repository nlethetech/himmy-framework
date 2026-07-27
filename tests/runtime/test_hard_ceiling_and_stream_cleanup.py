"""Hardening tests: the hard max_turns ceiling + streamed-loop early-close cleanup.

Two production behaviors locked in here:

1. ``HARD_MAX_TURNS`` — every loop entry point (``run_agent_loop``,
   ``stream_agent_loop``, ``resume_agent_loop``) validates the caller-supplied
   ``max_turns`` against a framework-enforced ceiling, so the runtime's
   underlying ``while True`` drive loops can never be handed an effectively
   unbounded turn budget (the runtime counterpart of the state graph's
   ``recursion_limit``).

2. Stream cleanup — a client closing ``stream_agent_loop`` mid-run (an SSE
   consumer disconnecting -> ``GeneratorExit``) or a task cancellation closes
   the in-flight inner turn generators (and the provider stream beneath them)
   deterministically, instead of abandoning suspended async generators to the
   event loop's lazy finalizer (which schedules dangling ``aclose`` tasks).

Offline-first: everything runs on scripted in-memory managers; nothing to skip.
"""

from __future__ import annotations

import asyncio
import gc
import sys
from collections.abc import AsyncIterator
from typing import Any

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread
from himmy.agents.personas.persona import Persona
from himmy.core.errors import HimmyError
from himmy.runtime import SingleAgentRuntime
from himmy.runtime.checkpoint import (
    AgentCheckpoint,
    InMemoryCheckpointStore,
    PendingToolCall,
)
from himmy.runtime.single_agent import HARD_MAX_TURNS
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    ToolCallRecord,
    ToolReturnRecord,
)
from himmy.services.inference.service import InferenceService, StreamDelta
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


# --------------------------------------------------------------------------- helpers
class _ScriptedManager:
    """A client manager returning a fixed sequence of turns (last repeats).

    No ``generate_stream`` — ``run_stream`` uses the deterministic offline
    fallback that buffers via ``run`` then chunks at 24 chars. A spec with
    ``tools=True`` carries a tool call/return (loop continues); ``False`` is a
    final answer. ``block_from_call`` (1-based) makes that generate call (and
    later ones) park on an event so a consumer can be cancelled mid-turn.
    """

    def __init__(
        self, specs: list[dict[str, Any]], *, block_from_call: int | None = None
    ) -> None:
        self._specs = specs
        self.calls = 0
        self._block_from_call = block_from_call
        self.blocked = asyncio.Event()  # set when a generate call starts blocking
        self.release = asyncio.Event()  # set by the test to unblock

    def resolve(self, model_key: str) -> str:
        return f"scripted:{model_key}"

    async def generate(self, request: Any) -> InferenceResponse:
        spec = self._specs[min(self.calls, len(self._specs) - 1)]
        self.calls += 1
        if self._block_from_call is not None and self.calls >= self._block_from_call:
            self.blocked.set()
            await self.release.wait()
        cid = f"c{self.calls}"
        has_tools = bool(spec.get("tools"))
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=spec.get("text", ""),
            tool_calls=(
                [ToolCallRecord(tool_call_id=cid, tool_name="probe", args={})]
                if has_tools
                else []
            ),
            tool_returns=(
                [
                    ToolReturnRecord(
                        tool_call_id=cid, tool_name="probe", content={"ok": True}
                    )
                ]
                if has_tools
                else []
            ),
            input_tokens=1,
            output_tokens=1,
        )


class _TrackingInferenceService(InferenceService):
    """An :class:`InferenceService` whose ``run_stream`` counts open/closed streams.

    ``streams_closed`` only increments when the wrapping generator's ``finally``
    actually runs — i.e. on exhaustion OR a deterministic ``aclose`` — so a
    suspended provider stream abandoned by an early consumer close is visible
    as ``streams_closed < streams_opened``.
    """

    def __init__(self, manager: Any) -> None:
        super().__init__(manager)
        self.streams_opened = 0
        self.streams_closed = 0

    async def run_stream(
        self, request: Any, *, chunk_size: int = 24
    ) -> AsyncIterator[StreamDelta]:
        self.streams_opened += 1
        inner = super().run_stream(request, chunk_size=chunk_size)
        try:
            async for delta in inner:
                yield delta
        finally:
            await inner.aclose()  # type: ignore[attr-defined]
            self.streams_closed += 1


def _scripted_runtime(
    specs: list[dict[str, Any]],
) -> tuple[SingleAgentRuntime, _ScriptedManager, _TrackingInferenceService]:
    manager = _ScriptedManager(specs)
    service = _TrackingInferenceService(manager)
    runtime = SingleAgentRuntime(
        inference_service=service, memory_store=StorageService()
    )
    return runtime, manager, service


def _go() -> tuple[Persona, Task]:
    return Persona(name="analyst"), Task(title="t", prompt="investigate")


def _other_tasks() -> list[asyncio.Task[Any]]:
    """All pending tasks on the loop besides the current one (leak detector)."""
    return [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]


async def _drain(gen: AsyncIterator[StreamDelta]) -> list[StreamDelta]:
    return [delta async for delta in gen]


# ------------------------------------------------------------- hard ceiling (Task 1a)
def test_run_agent_loop_rejects_out_of_range_max_turns() -> None:
    """max_turns below 1 or above HARD_MAX_TURNS is a typed configuration error."""
    rt, _manager, _service = _scripted_runtime([{"tools": False, "text": "x"}])
    persona, task = _go()
    with pytest.raises(HimmyError):
        run_async(rt.run_agent_loop(persona, task, max_turns=0))
    with pytest.raises(HimmyError, match="hard turn ceiling"):
        run_async(rt.run_agent_loop(persona, task, max_turns=HARD_MAX_TURNS + 1))


def test_stream_agent_loop_rejects_out_of_range_max_turns() -> None:
    """The streaming variant enforces the identical floor and ceiling."""
    rt, _manager, _service = _scripted_runtime([{"tools": False, "text": "x"}])
    persona, task = _go()
    with pytest.raises(HimmyError):
        run_async(_drain(rt.stream_agent_loop(persona, task, max_turns=0)))
    with pytest.raises(HimmyError, match="hard turn ceiling"):
        run_async(
            _drain(rt.stream_agent_loop(persona, task, max_turns=HARD_MAX_TURNS + 1))
        )


def test_max_turns_at_ceiling_is_accepted() -> None:
    """HARD_MAX_TURNS itself is a legal bound (inclusive ceiling)."""
    rt, _manager, _service = _scripted_runtime([{"tools": False, "text": "done"}])
    persona, task = _go()
    result = run_async(rt.run_agent_loop(persona, task, max_turns=HARD_MAX_TURNS))
    assert result.stopped_reason == "final"
    assert result.final.output_text == "done"


def test_resume_agent_loop_rejects_tampered_checkpoint_max_turns() -> None:
    """A hand-crafted checkpoint cannot smuggle an over-ceiling max_turns in."""
    persona, task = _go()
    thread = ChatThread(agent_id=persona.agent_id)
    checkpoint = AgentCheckpoint(
        persona=persona.model_dump(mode="json"),
        task=task.model_dump(mode="json"),
        thread=thread.model_dump(mode="json"),
        max_turns=HARD_MAX_TURNS + 50,
        pending_tool_calls=[PendingToolCall(tool_call_id="c1", tool_name="probe")],
    )
    store = InMemoryCheckpointStore()
    store.save(checkpoint)
    rt = SingleAgentRuntime(
        inference_service=InferenceService(
            _ScriptedManager([{"tools": False, "text": "x"}])
        ),
        memory_store=StorageService(),
        checkpoint_store=store,
    )
    with pytest.raises(HimmyError, match="hard turn ceiling"):
        run_async(rt.resume_agent_loop(checkpoint.checkpoint_id, approved=False))


# ------------------------------------------------------ stream cleanup (Task 1b)
def test_closing_stream_mid_first_turn_closes_provider_stream() -> None:
    """An early aclose (client dropped the SSE stream) closes the inner turn NOW.

    Without the try/finally in stream_agent_loop/stream_task the suspended
    stream_task + run_stream generators are abandoned to the event loop's lazy
    async-generator finalizer: ``streams_closed`` stays 0 and a dangling
    ``aclose`` task appears on the loop.
    """

    async def scenario() -> None:
        manager = _ScriptedManager([{"tools": True, "text": "x" * 200}])
        service = _TrackingInferenceService(manager)
        rt = SingleAgentRuntime(
            inference_service=service, memory_store=StorageService()
        )
        persona, task = _go()
        gen = rt.stream_agent_loop(persona, task, max_turns=5)
        first = await anext(gen)  # suspend mid first turn (chunked text)
        assert first.delta  # a real text delta flowed
        assert service.streams_opened == 1

        await gen.aclose()  # the client closed the stream mid-run

        # The in-flight provider stream was closed deterministically...
        assert service.streams_closed == 1
        # ... and not via a scheduled finalizer task left pending on the loop.
        gc.collect()
        assert _other_tasks() == []

    run_async(scenario())


def test_closing_stream_during_continuation_turns_leaves_no_dangling_work() -> None:
    """Closing mid drive-loop finalizes the run: no leaked tasks, no extra turns.

    The inner drive generator must be closed DETERMINISTICALLY by the outer
    generator's ``finally`` — not abandoned for the interpreter to garbage-
    collect through the event loop's asyncgen finalizer hook (whose timing is
    an implementation detail and whose lazy ``aclose`` runs in the wrong
    context — see the ContextVar reset in ``_subject_scope``). The hook
    counter below fails if ANY suspended async generator is left to GC.
    """

    async def scenario() -> None:
        manager = _ScriptedManager([{"tools": True, "text": "keep acting " * 5}])
        service = _TrackingInferenceService(manager)
        rt = SingleAgentRuntime(
            inference_service=service, memory_store=StorageService()
        )
        persona, task = _go()

        # Count asyncgens that reach the GC finalizer hook (i.e. were abandoned
        # while suspended instead of being closed deterministically). The hook
        # in effect at an asyncgen's FIRST iteration is the one captured for its
        # finalization, so it must be installed before the stream is iterated.
        gc_finalized: list[object] = []
        old_firstiter, old_finalizer = sys.get_asyncgen_hooks()

        def counting_finalizer(agen: object) -> None:
            gc_finalized.append(agen)
            if old_finalizer is not None:
                old_finalizer(agen)  # type: ignore[arg-type]

        sys.set_asyncgen_hooks(finalizer=counting_finalizer)
        try:
            gen = rt.stream_agent_loop(persona, task, max_turns=10)

            # Consume past turn 1's turn_end, then one turn-2 delta so the
            # continuation drive generator is live and suspended at a yield.
            while (await anext(gen)).event_type != "turn_end":
                pass
            await anext(gen)
            calls_before = manager.calls

            await gen.aclose()
            gc.collect()
        finally:
            sys.set_asyncgen_hooks(old_firstiter, old_finalizer)

        assert gc_finalized == []  # nothing was left for the lazy GC finalizer
        assert _other_tasks() == []  # no finalizer aclose tasks pending either
        # And the loop really stopped: no further model turns run afterwards.
        await asyncio.sleep(0)
        assert manager.calls == calls_before
        assert service.streams_closed == service.streams_opened

    run_async(scenario())


def test_subject_scope_reset_survives_cross_context_close() -> None:
    """``_subject_scope`` finalized in a DIFFERENT context than it was entered in
    must not raise — the streamed-loop early-``aclose`` path.

    ``stream_agent_loop`` yields to its consumer while suspended INSIDE
    ``_subject_scope``. In CPython an async generator runs each step in the caller's
    context, so when the consumer closes the stream (``aclosing(...)`` after an early
    ``break``, e.g. wrapped in ``asyncio.timeout``) from a different context than the
    first ``anext`` ran in, this scope's ``finally`` calls
    ``_CURRENT_SUBJECT.reset(token)`` against a token created elsewhere — raising
    "Token was created in a different Context" before the fix.

    We reproduce that condition deterministically: enter the CM inside a throwaway
    ``contextvars.Context`` (so ``set()`` binds the token there) and exit outside it.
    """
    import contextvars

    from himmy.runtime.audit import _CURRENT_SUBJECT
    from himmy.services.governance.consent import Decision, Effect

    rt, _manager, _service = _scripted_runtime([{"tools": False, "text": "x"}])
    # Governed mode. ``_run_subject`` returns ``None`` unless a decider is wired, so
    # without this the scope would stamp ``None`` and every assertion below would hold
    # vacuously — the test would still pass with the guard removed and the subject left
    # dangling. ``_subject_scope`` reaches the decider only through ``_run_subject``'s
    # ``is None`` check, so this one is never actually invoked.
    rt._consent_decider = lambda subject, purpose: Decision(
        subject_id=subject, purpose=purpose, effect=Effect.ALLOW
    )
    ctx = {"context_subject_id": "teacher_a"}

    entering = contextvars.Context()
    cm = rt._subject_scope(ctx)
    entering.run(cm.__enter__)  # set() binds the token to `entering`
    assert entering.run(_CURRENT_SUBJECT.get) == "teacher_a"

    # Exiting here runs reset() in the current (different) context: must not raise.
    cm.__exit__(None, None, None)

    # This context never observed the ``set()``, and the guard must not stamp it on the
    # way out either — writing ``None`` here would clobber a subject that a nesting
    # scope legitimately owns in this context.
    assert _CURRENT_SUBJECT.get() is None
    # Documents a KNOWN limitation rather than an intended behaviour: the entering
    # context keeps its stamp, because ``contextvars`` cannot reset a context you are
    # not currently in. The residue predates the guard (the raising ``reset`` left it
    # too, and additionally killed the run). If this ever starts failing, the scope has
    # gained real cross-context cleanup — update the guard's comment to match.
    assert entering.run(_CURRENT_SUBJECT.get) == "teacher_a"


def test_cancellation_mid_stream_propagates_and_cleans_up() -> None:
    """Cancelling a consumer mid-turn re-raises CancelledError and leaks nothing."""

    async def scenario() -> None:
        # Turn 1 streams normally; turn 2's generate parks so we can cancel mid-turn.
        manager = _ScriptedManager([{"tools": True, "text": "act"}], block_from_call=2)
        service = _TrackingInferenceService(manager)
        rt = SingleAgentRuntime(
            inference_service=service, memory_store=StorageService()
        )
        persona, task = _go()

        async def consume() -> None:
            async for _delta in rt.stream_agent_loop(persona, task, max_turns=5):
                pass

        consumer = asyncio.create_task(consume())
        await manager.blocked.wait()  # turn 2 is in flight inside the loop
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        assert consumer.cancelled()

        # Every opened provider stream was closed; nothing left on the loop.
        assert service.streams_closed == service.streams_opened
        gc.collect()
        assert _other_tasks() == []

    run_async(scenario())
