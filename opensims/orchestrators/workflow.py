"""Orchestrators kernel: declarative, state-threaded multi-step workflows.

A :class:`Workflow` is an ordered list of :class:`WorkflowStep`s sharing one chat
thread and an accumulating ``state`` dict. :class:`WorkflowOrchestrator` drives a
sequence of ``runtime.run_task`` calls, substituting state into each step's
subtask (missing keys surface as a clean failed step, not a ``KeyError``), and
optionally forcing one-tool-per-call execution via ``ResponseFormat.WORKFLOW``.

Production hardening (RO-1/3/4/5/6): the orchestrator reads typed
:class:`~opensims.runtime.single_agent.RunResult` records straight from
``runtime.run_task_detailed`` (so ``step.tool_calls`` are real
:class:`~opensims.services.inference.models.ToolCallRecord` objects, not
re-parsed thread rows); supports an overall/per-step timeout with clean
cancellation; supports resume-from-failed-step + per-step retry with backoff +
an idempotency short-circuit; and accepts a caller-facing ``on_event`` callback.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from opensims.core.events import EventType, RunEvent
from opensims.core.ids import new_uuid
from opensims.services.inference.models import (
    LLMConfig,
    ResponseFormat,
    ToolCallRecord,
    ToolReturnRecord,
    WorkflowDefinition,
    WorkflowState,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from opensims.agents.personas.persona import Persona
    from opensims.entities.records import EntityRecord
    from opensims.runtime.single_agent import RunResult, SingleAgentRuntime


# Caller-facing event callback (RO-6), mirrored from the runtime's ``OnEvent``.
OnEvent = Callable[[RunEvent], Awaitable[None]]


class WorkflowStep(BaseModel):
    """One step in a workflow: a subtask prompt plus per-step model/tool knobs."""

    name: str
    subtask: str
    tool_names: list[str] = []
    response_format: ResponseFormat | None = None
    output_json_schema: dict[str, Any] | None = None
    output_key: str | None = None
    sequential_tools: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    step_timeout_seconds: float | None = None
    metadata: dict[str, Any] = {}

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this step into its canonical ``EntityRecord`` (kind ``workflow_step``)."""
        from opensims.entities.records import EntityRecord, stable_id_for

        stable_id = stable_id_for(self.name, namespace="workflow_step")
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="workflow_step",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )


class Workflow(BaseModel):
    """A linear, declarative sequence of steps sharing one thread and state dict."""

    workflow_id: str = Field(default_factory=new_uuid)
    name: str
    description: str = ""
    steps: list[WorkflowStep] = []
    metadata: dict[str, Any] = {}

    def to_record(
        self, version: int = 1, metadata: dict[str, Any] | None = None
    ) -> EntityRecord:
        """Project this workflow into its canonical ``EntityRecord`` (kind ``workflow``)."""
        from opensims.entities.records import EntityRecord, stable_id_for

        stable_id = stable_id_for(self.workflow_id, namespace="workflow")
        return EntityRecord.create(
            stable_id=stable_id,
            version=version,
            kind="workflow",
            payload=self.model_dump(mode="json"),
            metadata=metadata or {},
        )


class WorkflowStepResult(BaseModel):
    """The outcome of one workflow step: status, output, and tool exchanges.

    RO-4: ``tool_calls`` / ``tool_returns`` are real
    :class:`ToolCallRecord` / :class:`ToolReturnRecord` objects read from the
    step's :class:`RunResult`, not dicts re-parsed from thread rows. ``attempts``
    records how many times the step ran (≥1 with per-step retry).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    step_name: str
    status: str  # completed | failed
    output_text: str | None = None
    output_structured: Any = None
    tool_calls: list[ToolCallRecord] = []
    tool_returns: list[ToolReturnRecord] = []
    error: str | None = None
    attempts: int = 1
    metadata: dict[str, Any] = {}


class WorkflowResult(BaseModel):
    """The aggregate outcome of running a whole workflow.

    RO-3: when a workflow stops on a failed step, ``next_index`` points at the
    step to resume from and ``final_state`` carries the accumulated state, so the
    caller can re-drive ``run(..., resume=result)`` to continue from the failure.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workflow_name: str
    status: str  # completed | failed | partial | cancelled
    final_state: dict[str, Any] = {}
    thread_id: str | None = None
    step_results: list[WorkflowStepResult] = []
    start_index: int = 0
    next_index: int | None = None
    idempotency_key: str | None = None


class WorkflowOrchestrator:
    """Drives a :class:`Workflow` step-by-step with state threading and one thread."""

    def __init__(
        self,
        runtime: SingleAgentRuntime,
        *,
        default_temperature: float = 0.1,
        max_step_retries: int = 0,
        retry_base_delay_seconds: float = 0.2,
        retry_jitter_seconds: float = 0.1,
        total_timeout_seconds: float | None = None,
        step_timeout_seconds: float | None = None,
        on_event: OnEvent | list[OnEvent] | None = None,
    ) -> None:
        """Wire the runtime + step/retry/timeout policy + caller-facing callbacks.

        ``max_step_retries`` (RO-3) retries a failed step up to N times with
        exponential backoff for transient (inference-class) failures. ``*_timeout``
        (RO-1) bound the whole run / each step; on expiry a terminal
        ``WORKFLOW_FINISHED(status='cancelled')`` is emitted. ``on_event`` (RO-6)
        streams progress to the caller.
        """
        self._runtime = runtime
        self._default_temperature = default_temperature
        self._max_step_retries = max(0, max_step_retries)
        self._retry_base_delay_seconds = retry_base_delay_seconds
        self._retry_jitter_seconds = retry_jitter_seconds
        self._total_timeout_seconds = total_timeout_seconds
        self._step_timeout_seconds = step_timeout_seconds
        self._on_event = self._coerce_callbacks(on_event)
        # In-process idempotency cache: idempotency_key -> completed result.
        self._completed: dict[str, WorkflowResult] = {}

    @staticmethod
    def _coerce_callbacks(
        on_event: OnEvent | list[OnEvent] | None,
    ) -> list[OnEvent]:
        """Normalize the ``on_event`` argument to a list of callables."""
        if on_event is None:
            return []
        if isinstance(on_event, list):
            return [cb for cb in on_event if cb is not None]
        return [on_event]

    async def run(
        self,
        workflow: Workflow,
        persona: Persona,
        *,
        initial_state: dict[str, Any] | None = None,
        stop_on_step_failure: bool = True,
        resume: WorkflowResult | None = None,
        start_index: int | None = None,
        idempotency_key: str | None = None,
        timeout_seconds: float | None = None,
    ) -> WorkflowResult:
        """Execute the workflow's steps in order, threading state, and return.

        Each step's ``subtask`` is formatted against the running state; a missing
        key yields a failed step (not a ``KeyError``). After a step, when it has
        an ``output_key`` the state is updated with its structured/text output.

        RO-3 recovery: pass ``resume`` (a prior :class:`WorkflowResult`) to
        re-drive from its ``next_index`` with the accumulated ``final_state``, or
        ``start_index`` to start from an explicit step. ``idempotency_key``
        short-circuits and returns the cached completed result if one exists.
        RO-1: ``timeout_seconds`` (or the constructor default) bounds the whole
        run; on expiry the result status is ``cancelled``.
        """
        key = idempotency_key or (resume.idempotency_key if resume else None)
        if key is not None and key in self._completed:
            return self._completed[key]

        total_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self._total_timeout_seconds
        )
        try:
            if total_timeout is not None and total_timeout > 0:
                async with _timeout(total_timeout):
                    result = await self._run_inner(
                        workflow,
                        persona,
                        initial_state=initial_state,
                        stop_on_step_failure=stop_on_step_failure,
                        resume=resume,
                        start_index=start_index,
                        idempotency_key=key,
                    )
            else:
                result = await self._run_inner(
                    workflow,
                    persona,
                    initial_state=initial_state,
                    stop_on_step_failure=stop_on_step_failure,
                    resume=resume,
                    start_index=start_index,
                    idempotency_key=key,
                )
        except (TimeoutError, asyncio.CancelledError):
            # RO-1: a timed-out (asyncio.timeout -> TimeoutError on exit) or
            # externally-cancelled workflow still records a terminal event before
            # unwinding as a cancellation.
            await self._emit(
                RunEvent(
                    event_type=EventType.WORKFLOW_FINISHED,
                    agent_id=persona.agent_id,
                    error="cancelled",
                    payload={"workflow_name": workflow.name, "status": "cancelled"},
                )
            )
            raise asyncio.CancelledError() from None

        if key is not None and result.status == "completed":
            self._completed[key] = result
        return result

    async def _run_inner(
        self,
        workflow: Workflow,
        persona: Persona,
        *,
        initial_state: dict[str, Any] | None,
        stop_on_step_failure: bool,
        resume: WorkflowResult | None,
        start_index: int | None,
        idempotency_key: str | None,
    ) -> WorkflowResult:
        """The bounded core of :meth:`run` (wrapped for the total timeout)."""
        from opensims.agents.base_agent.task import Task
        from opensims.agents.base_agent.thread import ChatThread

        # --- resume bookkeeping (RO-3) -------------------------------------
        if resume is not None:
            state: dict[str, Any] = dict(resume.final_state)
            begin = (
                start_index
                if start_index is not None
                else (resume.next_index if resume.next_index is not None else 0)
            )
            step_results: list[WorkflowStepResult] = list(resume.step_results)
            any_completed = any(r.status == "completed" for r in resume.step_results)
        else:
            state = dict(initial_state or {})
            begin = start_index or 0
            step_results = []
            any_completed = False

        begin = max(0, min(begin, len(workflow.steps)))
        thread = ChatThread(agent_id=persona.agent_id)

        await self._emit(
            RunEvent(
                event_type=EventType.WORKFLOW_STARTED,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload={
                    "workflow_name": workflow.name,
                    "steps": len(workflow.steps),
                    "start_index": begin,
                },
            )
        )

        any_failed = False
        next_index: int | None = None

        for index in range(begin, len(workflow.steps)):
            step = workflow.steps[index]
            # --- substitute state into the subtask (missing key -> failed) ---
            try:
                rendered_subtask = step.subtask.format(**state)
            except (KeyError, IndexError) as exc:
                missing = exc.args[0] if exc.args else "<unknown>"
                result = WorkflowStepResult(
                    step_name=step.name,
                    status="failed",
                    error=f"Missing state key referenced in subtask: {missing}",
                    metadata=dict(step.metadata),
                )
                step_results.append(result)
                any_failed = True
                await self._emit_step_completed(thread, persona, result)
                if stop_on_step_failure:
                    next_index = index
                    break
                continue

            result = await self._run_step(step, persona, rendered_subtask, thread, Task)
            step_results.append(result)
            await self._emit_step_completed(thread, persona, result)

            if result.status == "completed":
                any_completed = True
                if step.output_key is not None:
                    state[step.output_key] = (
                        result.output_structured
                        if result.output_structured is not None
                        else result.output_text
                    )
            else:
                any_failed = True
                if stop_on_step_failure:
                    next_index = index
                    break

        if any_failed and any_completed:
            status = "partial"
        elif any_failed:
            status = "failed"
        else:
            status = "completed"

        await self._emit(
            RunEvent(
                event_type=EventType.WORKFLOW_FINISHED,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload={"workflow_name": workflow.name, "status": status},
            )
        )

        return WorkflowResult(
            workflow_name=workflow.name,
            status=status,
            final_state=state,
            thread_id=thread.thread_id,
            step_results=step_results,
            start_index=begin,
            next_index=next_index,
            idempotency_key=idempotency_key,
        )

    async def _run_step(
        self,
        step: WorkflowStep,
        persona: Persona,
        rendered_subtask: str,
        thread: Any,
        task_cls: Any,
    ) -> WorkflowStepResult:
        """Run one step with per-step retry/backoff (RO-3) and timeout (RO-1)."""
        temperature = (
            step.temperature
            if step.temperature is not None
            else self._default_temperature
        )

        last_result: WorkflowStepResult | None = None
        for attempt in range(self._max_step_retries + 1):
            try:
                if step.step_timeout_seconds or self._step_timeout_seconds:
                    budget = step.step_timeout_seconds or self._step_timeout_seconds
                    assert budget is not None  # guaranteed by the if above
                    async with _timeout(budget):
                        result = await self._run_step_once(
                            step,
                            persona,
                            rendered_subtask,
                            thread,
                            task_cls,
                            temperature,
                        )
                else:
                    result = await self._run_step_once(
                        step,
                        persona,
                        rendered_subtask,
                        thread,
                        task_cls,
                        temperature,
                    )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - convert to a failed step
                result = WorkflowStepResult(
                    step_name=step.name,
                    status="failed",
                    error=str(exc),
                    metadata=dict(step.metadata),
                )

            result.attempts = attempt + 1
            if result.status == "completed":
                return result
            last_result = result
            # Retry transient (inference-class) failures with backoff.
            if attempt < self._max_step_retries:
                await asyncio.sleep(self._backoff_delay(attempt))
                continue
            break

        assert last_result is not None
        return last_result

    async def _run_step_once(
        self,
        step: WorkflowStep,
        persona: Persona,
        rendered_subtask: str,
        thread: Any,
        task_cls: Any,
        temperature: float,
    ) -> WorkflowStepResult:
        """One attempt at a step (forced-sequential or single run)."""
        if step.sequential_tools and len(step.tool_names) >= 2:
            return await self._run_sequential_tools(
                step, persona, rendered_subtask, thread, task_cls, temperature
            )
        return await self._run_single(
            step, persona, rendered_subtask, thread, task_cls, temperature
        )

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for retry attempt ``attempt`` (0-based)."""
        base = self._retry_base_delay_seconds * (2**attempt)
        return base + random.uniform(0.0, self._retry_jitter_seconds)

    async def _run_single(
        self,
        step: WorkflowStep,
        persona: Persona,
        rendered_subtask: str,
        thread: Any,
        task_cls: Any,
        temperature: float,
    ) -> WorkflowStepResult:
        """Run one step as a single ``run_task_detailed`` call (RO-4/RO-5)."""
        response_format = step.response_format
        if response_format is None and step.output_json_schema is not None:
            response_format = ResponseFormat.STRUCTURED_OUTPUT

        cfg = LLMConfig(
            response_format=response_format,
            output_json_schema=step.output_json_schema,
            temperature=temperature,
            max_tokens=step.max_tokens,
        )
        task = task_cls(
            title=step.name,
            prompt=rendered_subtask,
            context={"tool_names": step.tool_names} if step.tool_names else {},
        )
        result = await self._runtime.run_task_detailed(
            persona, task, thread=thread, llm_config=cfg
        )
        return self._step_result_from_run(step, result)

    async def _run_sequential_tools(
        self,
        step: WorkflowStep,
        persona: Persona,
        rendered_subtask: str,
        thread: Any,
        task_cls: Any,
        temperature: float,
    ) -> WorkflowStepResult:
        """Drive one forced tool per inference call across the step's tool list.

        RO-4/RO-5: aggregates the real :class:`ToolCallRecord`/
        :class:`ToolReturnRecord` objects from each iteration's typed
        :class:`RunResult` (preserving ORDER, the load-bearing WORKFLOW guarantee)
        instead of re-parsing thread rows. A failed iteration fails the step.
        """
        definition = WorkflowDefinition(steps=list(step.tool_names))
        state = WorkflowState(definition=definition)
        tool_calls: list[ToolCallRecord] = []
        tool_returns: list[ToolReturnRecord] = []
        last_text: str | None = None

        while not state.is_complete:
            cfg = LLMConfig(
                response_format=ResponseFormat.WORKFLOW,
                workflow=state,
                temperature=temperature,
            )
            task = task_cls(
                title=f"{step.name}:{state.current_tool_name}",
                prompt=rendered_subtask,
                context={"tool_names": step.tool_names},
            )
            result = await self._runtime.run_task_detailed(
                persona, task, thread=thread, llm_config=cfg
            )
            if not result.succeeded:
                return WorkflowStepResult(
                    step_name=step.name,
                    status="failed",
                    output_text=last_text,
                    tool_calls=tool_calls,
                    tool_returns=tool_returns,
                    error=result.error or "workflow step inference failed",
                    metadata=dict(step.metadata),
                )
            # Typed records, in call order, straight from the response.
            tool_calls.extend(result.tool_calls)
            tool_returns.extend(result.tool_returns)
            if result.output_text:
                last_text = result.output_text
            state = state.advance()

        return WorkflowStepResult(
            step_name=step.name,
            status="completed",
            output_text=last_text,
            tool_calls=tool_calls,
            tool_returns=tool_returns,
            metadata=dict(step.metadata),
        )

    @staticmethod
    def _step_result_from_run(
        step: WorkflowStep, result: RunResult
    ) -> WorkflowStepResult:
        """Build a step result from a typed :class:`RunResult` (RO-4/RO-5).

        Reads status/output/structured output and the real typed tool records
        directly from the response — no thread-row re-parsing, no fragile
        USER-boundary walk, and a FAILED run becomes a failed step.
        """
        if not result.succeeded:
            return WorkflowStepResult(
                step_name=step.name,
                status="failed",
                output_text=result.output_text,
                output_structured=result.output_structured,
                tool_calls=list(result.tool_calls),
                tool_returns=list(result.tool_returns),
                error=result.error or "step inference failed",
                metadata=dict(step.metadata),
            )

        # Prefer the structured output; fall back to parsing JSON text so that a
        # text-only structured payload still threads into state.
        output_structured: Any = result.output_structured
        if output_structured is None and result.output_text:
            import json

            try:
                parsed = json.loads(result.output_text)
                if isinstance(parsed, (dict, list)):
                    output_structured = parsed
            except (ValueError, TypeError):
                output_structured = None

        return WorkflowStepResult(
            step_name=step.name,
            status="completed",
            output_text=result.output_text,
            output_structured=output_structured,
            tool_calls=list(result.tool_calls),
            tool_returns=list(result.tool_returns),
            metadata=dict(step.metadata),
        )

    async def _emit(self, event: RunEvent) -> None:
        """Best-effort emit to the runtime's store/registry + caller callbacks.

        RO-6: workflow-level events (WORKFLOW_STARTED/STEP_COMPLETED/FINISHED) are
        streamed to any caller-supplied ``on_event`` callbacks too, so a UI can
        receive per-step progress without polling storage. Observability spans for
        these markers are emitted via :func:`emit_event_span` (invariant #6).
        """
        memory_store = getattr(self._runtime, "memory_store", None)
        if memory_store is not None:
            appender = getattr(memory_store, "append_event", None)
            if appender is not None:
                try:
                    await appender(event)
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - defensive
                    pass
        registry = getattr(self._runtime, "entity_registry", None)
        if registry is not None:
            try:
                registry.register(event.to_record())
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            from opensims.services.observability import emit_event_span

            emit_event_span(event)
        except Exception:  # pragma: no cover - defensive
            pass
        for callback in self._on_event:
            try:
                await callback(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - never let a listener break the run
                pass

    async def _emit_step_completed(
        self, thread: Any, persona: Persona, result: WorkflowStepResult
    ) -> None:
        """Emit a WORKFLOW_STEP_COMPLETED event for one finished step."""
        await self._emit(
            RunEvent(
                event_type=EventType.WORKFLOW_STEP_COMPLETED,
                thread_id=thread.thread_id,
                agent_id=persona.agent_id,
                payload={"step_name": result.step_name, "status": result.status},
                error=result.error,
            )
        )


def _timeout(seconds: float):
    """Return an ``asyncio.timeout(seconds)`` context manager (RO-1).

    Available on Python 3.11+ (the project targets 3.12); raises
    :class:`asyncio.CancelledError` inside the block on expiry, which ``run``
    catches to emit a terminal cancelled WORKFLOW_FINISHED before re-raising.
    """
    timeout_cm = getattr(asyncio, "timeout", None)
    if timeout_cm is None:  # pragma: no cover - 3.10 fallback only
        from opensims.core.errors import OpenSimsError

        raise OpenSimsError("workflow timeouts require Python 3.11+ (asyncio.timeout)")
    return timeout_cm(seconds)


__all__ = [
    "Workflow",
    "WorkflowStep",
    "WorkflowOrchestrator",
    "WorkflowStepResult",
    "WorkflowResult",
    "OnEvent",
]
