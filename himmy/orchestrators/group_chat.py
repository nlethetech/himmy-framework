"""Multi-agent at scale: GroupChat/selector + parallel fan-out + typed contracts.

This builds on the handoff/delegation orchestrator (:mod:`himmy.orchestrators.multi_agent`)
with three institutional collaboration modes, all over the SAME audited runtime seams:

* **GroupChat / selector** (:class:`GroupChatOrchestrator`) — many members share one
  audited :class:`~himmy.agents.base_agent.thread.ChatThread`; before each round a
  :class:`SpeakerSelector` picks the next speaker (round-robin, an LLM "manager"
  selector, or any caller-supplied policy). Each speaker's persona system prompt is
  re-rendered + re-injected onto the shared thread (reusing the handoff fix so a
  speaker never inherits the previous speaker's persona), it takes one turn, and the
  loop ends on ``final_answer``, a termination predicate, or ``max_rounds``.

* **Parallel fan-out** (:meth:`fan_out`) — run N workers CONCURRENTLY over a typed
  subtask contract, then join their typed results. Each worker runs in its own
  sub-thread (isolation), spend is summed, and the join is deterministic (results come
  back in submission order regardless of completion order).

* **Typed subtask/result contracts** (:class:`SubtaskSpec`, :class:`SubtaskResult`) —
  pydantic, not free text. A worker is asked to produce a ``SubtaskResult`` via the
  runtime's structured-output seam, and the result is validated before it joins.

Every selection / spoken turn / worker completion / join flows through the runtime's
own event emission (durable store + EntityRecord spine + caller callbacks), so a
scaled multi-agent run is fully replayable — keeping provenance-native auditing.
"""

from __future__ import annotations

import asyncio
import inspect
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from himmy.agents.base_agent.task import Task
from himmy.agents.base_agent.thread import ChatThread
from himmy.core import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.orchestrators.multi_agent import AgentTeam, OnEvent, TeamMember
from himmy.runtime.single_agent import RunResult, SingleAgentRuntime
from himmy.runtime.termination import (
    FINAL_ANSWER_TOOL,
    final_answer_text,
    register_final_answer_tool,
)
from himmy.services.inference.models import (
    InferenceMessage,
    InferenceRequest,
    InferenceStatus,
    LLMConfig,
    ResponseFormat,
)

# --------------------------------------------------------------------------- #
# Typed subtask / result contracts                                            #
# --------------------------------------------------------------------------- #


class SubtaskSpec(BaseModel):
    """A typed unit of work handed to a fan-out worker (pydantic, not free text).

    ``worker`` names the team member that should run it; ``objective`` is the work
    to do; ``inputs`` carries any structured payload the worker needs. This is the
    request half of the fan-out contract — callers build a list of these and the
    orchestrator runs them concurrently.
    """

    worker: str
    objective: str
    inputs: dict[str, Any] = Field(default_factory=dict)


class SubtaskResult(BaseModel):
    """A typed result produced by a fan-out worker (pydantic, not free text).

    ``worker`` echoes which member produced it; ``answer`` is the worker's prose
    answer; ``data`` is any structured payload. ``ok``/``error`` record whether the
    worker succeeded — a failed worker still yields a (well-formed) result so the
    join is total and the failure is visible in the typed output rather than as a
    raised exception that loses the other workers' results.
    """

    worker: str
    answer: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error: str | None = None


@dataclass
class FanOutResult:
    """The joined outcome of a parallel fan-out: one typed result per subtask.

    ``results`` preserves the SUBMISSION order of the specs (not completion order),
    so callers can zip specs to results deterministically. ``threads`` holds each
    worker's sub-thread for inspection / persistence.
    """

    results: list[SubtaskResult] = field(default_factory=list)
    threads: list[ChatThread] = field(default_factory=list)
    total_cost: float = 0.0

    @property
    def ok(self) -> bool:
        """True when every worker succeeded."""
        return all(r.ok for r in self.results)

    @property
    def answers(self) -> list[str]:
        """Just the worker answers, in submission order."""
        return [r.answer for r in self.results]


# --------------------------------------------------------------------------- #
# Speaker selection (manager / round-robin / LLM-selector)                    #
# --------------------------------------------------------------------------- #


@dataclass
class SelectionContext:
    """What a :class:`SpeakerSelector` sees when choosing the next speaker.

    ``candidates`` are the member names eligible to speak; ``last_speaker`` is the
    member who just spoke (``None`` on the first round); ``round_index`` is the
    0-based round; ``transcript`` is the running ``(speaker, text)`` history.
    """

    candidates: list[str]
    last_speaker: str | None
    round_index: int
    transcript: list[tuple[str, str]]


class SpeakerSelector(ABC):
    """Picks the next speaker over a shared group thread."""

    @abstractmethod
    async def select(self, context: SelectionContext) -> str:
        """Return the name of the next speaker (must be in ``context.candidates``)."""
        raise NotImplementedError


class RoundRobinSelector(SpeakerSelector):
    """Cycle through the candidates in order, round after round (deterministic)."""

    async def select(self, context: SelectionContext) -> str:
        """Pick the candidate after ``last_speaker`` (wrapping), or the first."""
        candidates = context.candidates
        if not candidates:
            raise HimmyError("RoundRobinSelector: no candidates to choose from")
        if context.last_speaker is None or context.last_speaker not in candidates:
            return candidates[0]
        idx = candidates.index(context.last_speaker)
        return candidates[(idx + 1) % len(candidates)]


class CallableSelector(SpeakerSelector):
    """Adapt any (async) callable into a selector — the "manager-as-function" seam."""

    def __init__(
        self,
        fn: Callable[[SelectionContext], Awaitable[str]]
        | Callable[[SelectionContext], str],
    ) -> None:
        """Wrap ``fn``; it may be sync or async and must return a candidate name."""
        self._fn = fn

    async def select(self, context: SelectionContext) -> str:
        """Invoke the wrapped callable (awaiting it when it returns an awaitable)."""
        out = self._fn(context)
        if inspect.isawaitable(out):
            return await out
        return out


class LLMSelector(SpeakerSelector):
    """A "manager" LLM that reads the transcript and names the next speaker.

    Asks the model — via the runtime's inference service and an enum-constrained
    structured-output schema — to pick the next speaker from the candidates. Offline
    (the zero-config stub), the enum schema deterministically resolves to the FIRST
    candidate, so a group chat with an LLM selector still runs end-to-end with no
    keys/network; a real model picks the most relevant speaker. On any inference
    failure or an out-of-set pick, it falls back to round-robin so the chat never
    stalls.
    """

    def __init__(
        self,
        runtime: SingleAgentRuntime,
        *,
        model_key: str = "default",
        descriptions: dict[str, str] | None = None,
    ) -> None:
        """Wire the selector to a runtime; ``descriptions`` hints each member's role."""
        self._runtime = runtime
        self._model_key = model_key
        self._descriptions = descriptions or {}
        self._fallback = RoundRobinSelector()

    async def select(self, context: SelectionContext) -> str:
        """Pick the next speaker, falling back to round-robin on any failure."""
        candidates = context.candidates
        if not candidates:
            raise HimmyError("LLMSelector: no candidates to choose from")
        if len(candidates) == 1:
            return candidates[0]
        schema = {
            "type": "object",
            "properties": {"next_speaker": {"type": "string", "enum": candidates}},
            "required": ["next_speaker"],
            "additionalProperties": False,
        }
        request = InferenceRequest(
            model_key=self._model_key,
            messages=[
                InferenceMessage(
                    role="system", content=self._system_prompt(candidates)
                ),
                InferenceMessage(role="user", content=self._render_transcript(context)),
            ],
            response_format=ResponseFormat.STRUCTURED_OUTPUT,
            output_json_schema=schema,
        )
        try:
            response = await self._runtime.inference_service.run(request)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive: never stall the chat
            return await self._fallback.select(context)
        if response.status != InferenceStatus.SUCCESS or not response.output_structured:
            return await self._fallback.select(context)
        pick = response.output_structured.get("next_speaker")
        if isinstance(pick, str) and pick in candidates:
            return pick
        return await self._fallback.select(context)

    def _system_prompt(self, candidates: list[str]) -> str:
        """Instruct the manager model to name exactly one next speaker."""
        roster = "\n".join(
            f"- {name}: {self._descriptions.get(name, 'team member')}"
            for name in candidates
        )
        return (
            "You are the manager of a multi-agent group chat. Read the conversation "
            "so far and choose which team member should speak next to move the task "
            "forward. Choose exactly one of:\n"
            f"{roster}\n"
            "Return only the chosen member's name."
        )

    def _render_transcript(self, context: SelectionContext) -> str:
        """Render the running transcript for the manager to read."""
        if not context.transcript:
            return "(no messages yet — choose who should open the discussion)"
        lines = [f"{speaker}: {text}" for speaker, text in context.transcript[-12:]]
        return "Conversation so far:\n" + "\n".join(lines)


# --------------------------------------------------------------------------- #
# GroupChat orchestrator                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class GroupChatResult:
    """The outcome of a group chat: the shared thread, the turns, and the trail."""

    thread: ChatThread
    turns: list[tuple[str, RunResult]] = field(default_factory=list)
    speaker_order: list[str] = field(default_factory=list)
    final_speaker: str = ""
    output_text: str | None = None
    stopped_reason: str = "final"  # final | final_answer | max_rounds | terminated
    total_cost: float = 0.0

    @property
    def round_count(self) -> int:
        """The number of speaker turns executed."""
        return len(self.turns)


#: A termination predicate: given ``(speaker, RunResult)`` decide whether to stop.
TerminationPredicate = Callable[[str, RunResult], bool]


class GroupChatOrchestrator:
    """Run a team as a selector-driven group chat over one shared audited thread."""

    def __init__(
        self,
        runtime: SingleAgentRuntime,
        team: AgentTeam,
        tool_registry: Any,
        *,
        selector: SpeakerSelector | None = None,
        on_event: OnEvent | list[OnEvent] | None = None,
        max_rounds: int = 8,
        default_temperature: float = 0.1,
        terminate_when: TerminationPredicate | None = None,
        speakers: list[str] | None = None,
        allow_final_answer: bool = True,
    ) -> None:
        """Wire the group chat.

        ``selector`` chooses the next speaker each round (defaults to round-robin).
        ``speakers`` optionally restricts who may speak (defaults to ALL members).
        ``terminate_when`` is an optional predicate to end the chat early; a speaker
        calling ``final_answer`` always ends it. With ``allow_final_answer`` (the
        default) each speaking member gets the ``final_answer`` tool bound so it can
        end the discussion cleanly; set it ``False`` for a fixed-length panel that
        always runs to ``max_rounds`` (or a ``terminate_when`` predicate).
        """
        if max_rounds < 1:
            raise HimmyError("GroupChatOrchestrator requires max_rounds >= 1.")
        self._runtime = runtime
        self._team = team
        self._registry = tool_registry
        self._selector = selector or RoundRobinSelector()
        self._max_rounds = max_rounds
        self._temperature = default_temperature
        self._terminate_when = terminate_when
        self._allow_final_answer = allow_final_answer
        names = speakers if speakers is not None else [m.name for m in team.members]
        for name in names:
            team.require(name)  # fail fast on an unknown speaker
        if not names:
            raise HimmyError("GroupChatOrchestrator requires at least one speaker.")
        self._speakers = names
        if on_event is None:
            self._on_event: list[OnEvent] = []
        elif callable(on_event):
            self._on_event = [on_event]
        else:
            self._on_event = list(on_event)
        if allow_final_answer:
            register_final_answer_tool(self._registry)

    async def run(
        self, prompt: str, *, initial_thread: ChatThread | None = None
    ) -> GroupChatResult:
        """Run the group chat until a final answer, a termination, or ``max_rounds``."""
        thread = initial_thread or ChatThread()
        turns: list[tuple[str, RunResult]] = []
        order: list[str] = []
        transcript: list[tuple[str, str]] = []

        await self._emit(
            RunEvent(
                event_type=EventType.GROUP_CHAT_STARTED,
                thread_id=thread.thread_id,
                payload={"speakers": list(self._speakers), "prompt": prompt},
            )
        )

        last_speaker: str | None = None
        round_index = 0
        is_first = True
        while True:
            selection = SelectionContext(
                candidates=list(self._speakers),
                last_speaker=last_speaker,
                round_index=round_index,
                transcript=list(transcript),
            )
            speaker_name = await self._selector.select(selection)
            speaker = self._team.require(speaker_name)
            order.append(speaker_name)
            await self._emit(
                RunEvent(
                    event_type=EventType.GROUP_SPEAKER_SELECTED,
                    thread_id=thread.thread_id,
                    agent_id=speaker.persona.agent_id,
                    payload={"speaker": speaker_name, "round": round_index},
                )
            )

            result = await self._speak(speaker, thread, prompt if is_first else None)
            turns.append((speaker_name, result))
            spoken_text = result.output_text or ""
            transcript.append((speaker_name, spoken_text))
            await self._emit(
                RunEvent(
                    event_type=EventType.GROUP_SPEAKER_SPOKE,
                    thread_id=thread.thread_id,
                    agent_id=speaker.persona.agent_id,
                    cost=result.cost,
                    payload={"speaker": speaker_name, "round": round_index},
                )
            )

            answer = final_answer_text(result)
            if answer is not None:
                return await self._finish(
                    thread, turns, order, speaker_name, "final_answer", answer
                )
            if self._terminate_when is not None and self._terminate_when(
                speaker_name, result
            ):
                return await self._finish(
                    thread, turns, order, speaker_name, "terminated", spoken_text
                )
            if len(turns) >= self._max_rounds:
                return await self._finish(
                    thread, turns, order, speaker_name, "max_rounds", spoken_text
                )

            last_speaker = speaker_name
            round_index += 1
            is_first = False

    async def _speak(
        self, speaker: TeamMember, thread: ChatThread, prompt: str | None
    ) -> RunResult:
        """Have ``speaker`` take one turn on the shared thread.

        On the FIRST turn we seed the thread with the task prompt (a fresh run); on
        later turns we re-inject the speaker's persona system prompt onto the shared
        thread (so a speaker never inherits the prior speaker's persona — same fix as
        a handoff) and continue the thread.
        """
        ctx = self._ctx(speaker)
        if prompt is not None and not thread.messages:
            return await self._runtime.run_task_detailed(
                speaker.persona,
                Task(title=f"{speaker.name}-open", prompt=prompt, context=ctx),
                thread=thread,
                llm_config=self._cfg(speaker),
            )
        await self._runtime.reinject_system_prompt(
            speaker.persona, thread, task_context=ctx
        )
        return await self._runtime.continue_turn(
            speaker.persona,
            thread,
            task_context=ctx,
            llm_config=self._cfg(speaker),
        )

    def _ctx(self, member: TeamMember) -> dict[str, Any]:
        """The per-turn run context: this member's bound tools (+final_answer) + model.

        ``tool_names`` is ALWAYS set (even to an empty list) so the runtime binds only
        this speaker's tools — never every registered tool, which would leak other
        members' synthetic tools into this turn.
        """
        names = list(member.tools)
        if self._allow_final_answer:
            names.append(FINAL_ANSWER_TOOL)
        return {"model_key": member.model_key, "tool_names": names}

    def _cfg(self, member: TeamMember) -> LLMConfig:
        """AUTO_TOOLS when the speaker has any bound tool (incl. final_answer)."""
        has_tools = bool(member.tools) or self._allow_final_answer
        return LLMConfig(
            model_key=member.model_key,
            temperature=self._temperature,
            response_format=ResponseFormat.AUTO_TOOLS if has_tools else None,
        )

    async def _finish(
        self,
        thread: ChatThread,
        turns: list[tuple[str, RunResult]],
        order: list[str],
        final_speaker: str,
        reason: str,
        answer: str | None,
    ) -> GroupChatResult:
        """Assemble the :class:`GroupChatResult` and emit the audited finish event."""
        total = sum(r.cost for _, r in turns)
        await self._emit(
            RunEvent(
                event_type=EventType.GROUP_CHAT_FINISHED,
                thread_id=thread.thread_id,
                cost=total,
                payload={
                    "final_speaker": final_speaker,
                    "reason": reason,
                    "rounds": len(turns),
                },
            )
        )
        return GroupChatResult(
            thread=thread,
            turns=turns,
            speaker_order=order,
            final_speaker=final_speaker,
            output_text=answer,
            stopped_reason=reason,
            total_cost=total,
        )

    # --------------------------------------------------------------- fan-out

    async def fan_out(
        self,
        specs: list[SubtaskSpec],
        *,
        max_worker_turns: int = 6,
        concurrency: int | None = None,
    ) -> FanOutResult:
        """Run ``specs`` workers CONCURRENTLY and join their typed results.

        Each spec runs its named worker in its own sub-thread, requesting a
        :class:`SubtaskResult` via the runtime's structured-output seam, and the
        validated results are returned IN SUBMISSION ORDER (deterministic join). A
        worker failure is captured as a ``SubtaskResult(ok=False, error=...)`` rather
        than raising — so one failing worker never loses the others' work.
        ``concurrency`` optionally caps simultaneous workers (default: unbounded).
        """
        return await fan_out(
            self._runtime,
            self._team,
            specs,
            on_event=self._on_event,
            max_worker_turns=max_worker_turns,
            concurrency=concurrency,
            temperature=self._temperature,
        )

    # ------------------------------------------------------------------ emit

    async def _emit(self, event: RunEvent) -> None:
        """Best-effort emit to the runtime's store/registry + caller callbacks."""
        await _emit_event(self._runtime, self._on_event, event)


# --------------------------------------------------------------------------- #
# Parallel fan-out (standalone — usable without a group chat)                  #
# --------------------------------------------------------------------------- #


async def fan_out(
    runtime: SingleAgentRuntime,
    team: AgentTeam,
    specs: list[SubtaskSpec],
    *,
    on_event: OnEvent | list[OnEvent] | None = None,
    max_worker_turns: int = 6,
    concurrency: int | None = None,
    temperature: float = 0.1,
) -> FanOutResult:
    """Run N typed subtasks concurrently across a team and join their typed results.

    Standalone parallel delegation: each :class:`SubtaskSpec` runs its named worker
    in an isolated sub-thread (so workers do not contend on one thread), requesting a
    :class:`SubtaskResult` through the runtime's structured-output seam. Results are
    returned in SUBMISSION order regardless of completion order (deterministic join),
    spend is summed, and a worker that fails yields ``SubtaskResult(ok=False, ...)``
    so the join is total. Every worker completion and the final join emit audited
    events through the runtime's own emission. ``concurrency`` optionally caps the
    number of simultaneously running workers.
    """
    callbacks = _normalize_callbacks(on_event)
    if not specs:
        await _emit_event(
            runtime,
            callbacks,
            RunEvent(event_type=EventType.FANOUT_JOINED, payload={"count": 0}),
        )
        return FanOutResult()

    if concurrency is not None and concurrency < 1:
        raise HimmyError("fan_out concurrency must be >= 1 when set.")

    await _emit_event(
        runtime,
        callbacks,
        RunEvent(
            event_type=EventType.FANOUT_STARTED,
            payload={
                "count": len(specs),
                "workers": [s.worker for s in specs],
                "concurrency": concurrency,
            },
        ),
    )

    semaphore = asyncio.Semaphore(concurrency) if concurrency is not None else None

    async def _run_one(spec: SubtaskSpec) -> tuple[SubtaskResult, ChatThread]:
        if semaphore is not None:
            async with semaphore:
                return await _run_worker(
                    runtime, team, spec, callbacks, max_worker_turns, temperature
                )
        return await _run_worker(
            runtime, team, spec, callbacks, max_worker_turns, temperature
        )

    # asyncio.gather preserves SUBMISSION order in its result list (not completion
    # order), so zipping specs[i] -> pairs[i] is a deterministic join.
    pairs = await asyncio.gather(*(_run_one(spec) for spec in specs))
    results = [p[0] for p in pairs]
    threads = [p[1] for p in pairs]
    total = sum(t.metadata.get("fanout_cost", 0.0) for t in threads)

    await _emit_event(
        runtime,
        callbacks,
        RunEvent(
            event_type=EventType.FANOUT_JOINED,
            payload={
                "count": len(results),
                "ok": all(r.ok for r in results),
                "total_cost": total,
            },
        ),
    )
    return FanOutResult(results=results, threads=threads, total_cost=total)


async def _run_worker(
    runtime: SingleAgentRuntime,
    team: AgentTeam,
    spec: SubtaskSpec,
    callbacks: list[OnEvent],
    max_worker_turns: int,
    temperature: float,
) -> tuple[SubtaskResult, ChatThread]:
    """Run one worker over a typed subtask, returning a validated typed result."""
    sub_thread = ChatThread()
    worker = team.get(spec.worker)
    if worker is None:
        result = SubtaskResult(
            worker=spec.worker,
            ok=False,
            error=f"team has no member {spec.worker!r}",
        )
        await _emit_worker_completed(runtime, callbacks, sub_thread, spec, result, 0.0)
        return result, sub_thread

    schema = SubtaskResult.model_json_schema()
    prompt = _render_subtask_prompt(spec)
    task = Task(
        title=f"fanout-{spec.worker}",
        prompt=prompt,
        context={
            "tool_names": list(worker.tools),
            "model_key": worker.model_key,
            "output_schema": schema,
        },
    )
    try:
        loop = await runtime.run_agent_loop(
            worker.persona,
            task,
            sub_thread,
            max_turns=max_worker_turns,
            llm_config=LLMConfig(
                model_key=worker.model_key,
                temperature=temperature,
                response_format=ResponseFormat.STRUCTURED_OUTPUT,
                output_json_schema=schema,
            ),
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - capture as a typed failure result
        result = SubtaskResult(worker=spec.worker, ok=False, error=str(exc))
        await _emit_worker_completed(runtime, callbacks, sub_thread, spec, result, 0.0)
        return result, sub_thread

    cost = sum(t.cost for t in loop.turns)
    result = _coerce_result(spec, loop.final)
    sub_thread.metadata["fanout_cost"] = cost
    await _emit_worker_completed(runtime, callbacks, loop.thread, spec, result, cost)
    return result, loop.thread


def _coerce_result(spec: SubtaskSpec, run: RunResult) -> SubtaskResult:
    """Validate the worker's structured output into a :class:`SubtaskResult`.

    Prefers the structured payload; falls back to wrapping the prose answer when the
    model returned text only. A validation failure is captured (not raised).
    """
    if not run.succeeded:
        return SubtaskResult(
            worker=spec.worker, ok=False, error=run.error or "inference failed"
        )
    payload = run.output_structured
    if isinstance(payload, dict):
        data = dict(payload)
        # The model fills `worker` from the schema; pin it to the real worker name.
        data["worker"] = spec.worker
        try:
            return SubtaskResult.model_validate(data)
        except Exception:  # noqa: BLE001 - fall back to the prose answer
            pass
    return SubtaskResult(worker=spec.worker, answer=run.output_text or "")


def _render_subtask_prompt(spec: SubtaskSpec) -> str:
    """Render a typed subtask into a prompt for its worker."""
    lines = [f"Objective: {spec.objective}"]
    if spec.inputs:
        import json

        lines.append("Inputs (JSON):")
        lines.append(json.dumps(spec.inputs, default=str))
    lines.append("Produce a SubtaskResult with your answer and any structured data.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Shared emission helpers (reuse the runtime's own audit fan-out)             #
# --------------------------------------------------------------------------- #


def _normalize_callbacks(on_event: OnEvent | list[OnEvent] | None) -> list[OnEvent]:
    """Coerce the ``on_event`` argument into a list of callbacks."""
    if on_event is None:
        return []
    if callable(on_event):
        return [on_event]
    return list(on_event)


async def _emit_worker_completed(
    runtime: SingleAgentRuntime,
    callbacks: list[OnEvent],
    thread: ChatThread,
    spec: SubtaskSpec,
    result: SubtaskResult,
    cost: float,
) -> None:
    """Emit the audited per-worker completion event."""
    await _emit_event(
        runtime,
        callbacks,
        RunEvent(
            event_type=EventType.FANOUT_WORKER_COMPLETED,
            thread_id=thread.thread_id,
            cost=cost,
            error=result.error,
            payload={
                "worker": spec.worker,
                "ok": result.ok,
                "objective": spec.objective,
            },
        ),
    )


async def _emit_event(
    runtime: SingleAgentRuntime,
    callbacks: list[OnEvent],
    event: RunEvent,
) -> None:
    """Best-effort emit to the runtime's store/registry + caller callbacks.

    Mirrors :meth:`MultiAgentOrchestrator._emit` so scaled multi-agent transitions
    flow through the SAME audit spine (durable memory store + EntityRecord registry)
    and the caller-facing callbacks, with no new sink wiring.
    """
    memory_store = getattr(runtime, "memory_store", None)
    appender = getattr(memory_store, "append_event", None) if memory_store else None
    if appender is not None:
        try:
            await appender(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - defensive
            pass
    registry = getattr(runtime, "entity_registry", None)
    if registry is not None:
        try:
            registry.register(event.to_record())
        except Exception:  # pragma: no cover - defensive
            pass
    for callback in callbacks:
        try:
            await callback(event)
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover - never let a listener break a run
            pass


__all__ = [
    "SubtaskSpec",
    "SubtaskResult",
    "FanOutResult",
    "SelectionContext",
    "SpeakerSelector",
    "RoundRobinSelector",
    "CallableSelector",
    "LLMSelector",
    "GroupChatOrchestrator",
    "GroupChatResult",
    "TerminationPredicate",
    "fan_out",
]
