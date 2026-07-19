"""HITL resume / checkpoint machinery for :class:`SingleAgentRuntime`.

Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
step ``resume``). :class:`ResumeCoordinator` owns the human-in-the-loop
resume/checkpoint body:

* ``_resume_agent_loop_locked`` — the resume body run under the per-checkpoint
  lock: atomic store claim, decision application per pending tool call, exactly-
  once idempotency via the durable ledger, the resolve save, one continuation
  turn, then the shared drive loop;
* ``_record_resume_final_output`` — persists the resolved member's final output
  onto its terminal checkpoint (orchestration crash-recovery anchor);
* ``_resume_lock_for`` — the per-checkpoint in-process lock (created on first
  use), keyed on the runtime's ``_resume_locks`` state; and
* ``_save_checkpoint`` — persist a paused run as a durable checkpoint.

The PUBLIC ``resume_agent_loop`` and the ``_resume_locks`` dict stay ON the
runtime; this coordinator holds a back-reference to the owning
:class:`SingleAgentRuntime` and reads its live wiring (``_checkpoint_store``,
``tool_service``, ``_resume_locks``, ``_subject_scope``, ``_append_tool_messages``,
``_emit``, ``_register_thread_version``, ``_continue_turn``, ``_emit_turn_completed``,
``_drive_loop``) at call time. Resume/idempotency semantics — the atomic claim,
the exactly-once ledger replay, event ORDER, and exception types — are
byte-for-byte identical to the pre-extraction inline code.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError
from himmy.core.events import EventType, RunEvent
from himmy.runtime.checkpoint import (
    APPROVED,
    REJECTED,
    AgentCheckpoint,
    PendingToolCall,
)
from himmy.services.inference.models import (
    InferenceResponse,
    InferenceStatus,
    LLMConfig,
    ToolCallRecord,
    ToolReturnRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.base_agent.task import Task
    from himmy.agents.base_agent.thread import ChatThread
    from himmy.agents.personas.persona import Persona
    from himmy.runtime.single_agent import AgentLoopResult, SingleAgentRuntime


class ResumeCoordinator:
    """Runs the HITL resume/checkpoint machinery for one runtime.

    Holds a back-reference to the owning :class:`SingleAgentRuntime` and reads its
    live wiring at call time, so runtime reconfiguration between runs is honored
    exactly as when the logic lived inline on the runtime. The public
    ``resume_agent_loop`` and the ``_resume_locks`` dict remain on the runtime.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    def resume_lock_for(self, checkpoint_id: str) -> asyncio.Lock:
        """The per-checkpoint resume lock (created on first use), serializing resumes.

        Two concurrent resumes of the SAME checkpoint on one event loop must not
        interleave between the atomic claim and the gated tool's execution, or both
        could run the approved action. Different checkpoints get different locks, so
        unrelated resumes still proceed in parallel.
        """
        rt = self._rt
        lock = rt._resume_locks.get(checkpoint_id)
        if lock is None:
            lock = asyncio.Lock()
            rt._resume_locks[checkpoint_id] = lock
        return lock

    async def resume_agent_loop_locked(
        self,
        checkpoint_id: str,
        *,
        approved: bool,
        llm_config: LLMConfig | None,
        hitl: bool,
        actor: str = "human",
    ) -> AgentLoopResult:
        """The resume body, run under the per-checkpoint lock (see resume_agent_loop)."""
        import himmy.runtime.single_agent as _single_agent

        rt = self._rt
        assert rt._checkpoint_store is not None  # guaranteed by the caller
        # Atomic claim: compare-and-set the status to ``resolving`` as a single store
        # operation. Concurrent resumes of the same checkpoint (two tabs, an
        # automation retry, two workers on the shared SQLite file) race here, and only
        # the winner proceeds to execute the approval-gated tool — the loser's claim
        # returns False and it is refused outright, so the gated action runs EXACTLY
        # once. (The old plain status check was a TOCTOU: both callers read
        # awaiting_approval, both executed, then both flipped the status.) An
        # already-resolved (approved/rejected) checkpoint also loses here.
        if not rt._checkpoint_store.claim(checkpoint_id):
            current = rt._checkpoint_store.load(checkpoint_id)
            status = current.status if current is not None else "unknown"
            raise HimmyError(
                f"checkpoint {checkpoint_id!r} already resolved ({status})."
            )
        # We hold the claim: re-load so the executed_tool_results ledger (and status)
        # reflect the just-claimed row rather than the pre-claim snapshot.
        checkpoint = rt._checkpoint_store.load(checkpoint_id)
        if checkpoint is None:  # pragma: no cover - the claim just observed it
            raise HimmyError(f"unknown checkpoint {checkpoint_id!r}.")
        # Checkpoints written by the validated entry points always pass; this guards
        # the resumed drive loop against a hand-crafted/tampered checkpoint row.
        _single_agent._validate_max_turns(checkpoint.max_turns, "resume_agent_loop")
        if approved and rt.tool_service is None:
            raise HimmyError(
                "cannot resume approved: no tool_service to execute the action."
            )

        from himmy.agents.base_agent.task import Task as _Task
        from himmy.agents.base_agent.thread import ChatThread as _ChatThread
        from himmy.agents.personas.persona import Persona as _Persona
        from himmy.services.tools.models import ToolInvocation

        persona = _Persona.model_validate(checkpoint.persona)
        task = _Task.model_validate(checkpoint.task)
        thread = _ChatThread.model_validate(checkpoint.thread)
        # Like the max_turns guard above: contexts checkpointed by the validated
        # entry points always pass; this catches a hand-crafted/tampered ctx row.
        ctx = _single_agent._validated_ctx(checkpoint.ctx, "resume_agent_loop")
        trace_id = f"{thread.thread_id}:{task.task_id}"
        resume_llm = llm_config
        if resume_llm is None and checkpoint.llm_config is not None:
            resume_llm = LLMConfig.model_validate(checkpoint.llm_config)

        # WS4.6: publish the subject (rebuilt from the checkpoint's ``ctx``, which carries
        # ``context_subject_id``) for the WHOLE resume path so the resumed tool messages,
        # the APPROVAL_* / turn events, the bumped thread version, the inner _continue_turn
        # (which does NOT self-wrap — only the public continue_turn does), and _drive_loop
        # all emit subject-tagged spine records — otherwise a HITL-resumed run for a fully
        # consented subject would emit subject-less records the fail-closed
        # ConsentAwareRegistry silently drops. Governed only; a no-op when no
        # consent_decider is wired. Inner scopes nest cleanly.
        # The per-checkpoint idempotency record: each approved execution is recorded
        # (and persisted) the moment it completes, so a repeated resume — including a
        # retry after a crash between executing a state-mutating tool and resolving
        # the checkpoint — replays the recorded result instead of running it twice.
        idempotency = _single_agent._CheckpointToolIdempotencyStore(
            checkpoint, rt._checkpoint_store
        )

        with rt._subject_scope(ctx):
            # Apply the human decision to each pending tool call, recording the outcome
            # on the thread as a TOOL message (so the next model turn sees it).
            for call in checkpoint.pending_tool_calls:
                if approved:
                    assert rt.tool_service is not None
                    execution = await rt.tool_service.execute(
                        ToolInvocation(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            args=dict(call.args),
                            metadata={
                                "approved": True,
                                "idempotency_key": call.tool_call_id,
                            },
                        ),
                        idempotency_store=idempotency,
                    )
                    tool_returns = [
                        ToolReturnRecord(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            content=execution.result,
                            outcome=execution.outcome,
                            metadata={
                                "approved_by": actor,
                                "error_code": execution.error_code.value
                                if execution.error_code
                                else None,
                            },
                        )
                    ]
                    event_type = EventType.APPROVAL_GRANTED
                else:
                    tool_returns = [
                        ToolReturnRecord(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            content={"rejected": True, "reason": "rejected by human"},
                            outcome="rejected",
                            metadata={"approved_by": actor},
                        )
                    ]
                    event_type = EventType.APPROVAL_REJECTED
                synthetic = InferenceResponse(
                    request_id=f"resume:{checkpoint_id}",
                    status=InferenceStatus.SUCCESS,
                    tool_calls=[
                        ToolCallRecord(
                            tool_call_id=call.tool_call_id,
                            tool_name=call.tool_name,
                            args=dict(call.args),
                        )
                    ],
                    tool_returns=tool_returns,
                )
                await rt._append_tool_messages(
                    thread,
                    synthetic,
                    request_id=synthetic.request_id,
                    trace_id=trace_id,
                    agent_id=persona.agent_id,
                )
                await rt._emit(
                    RunEvent(
                        event_type=event_type,
                        trace_id=trace_id,
                        thread_id=thread.thread_id,
                        agent_id=persona.agent_id,
                        tool_call_id=call.tool_call_id,
                        payload={
                            "checkpoint_id": checkpoint_id,
                            "tool_name": call.tool_name,
                            # Enriched (P0-B) so approvals are mineable per agent /
                            # tool / decision / latency without re-joining tables.
                            "decision": "granted" if approved else "rejected",
                            "agent_name": persona.name,
                            "actor": actor,
                            "time_to_decision_ms": _single_agent._decision_latency_ms(
                                checkpoint.created_at
                            ),
                        },
                    )
                )

            # Resolve the checkpoint exactly once (idempotency guard above).
            rt._checkpoint_store.save(
                checkpoint.model_copy(
                    update={"status": APPROVED if approved else REJECTED}
                )
            )
            thread.version += 1
            rt._register_thread_version(thread)

            # One continuation turn so the model reacts to the decision, then drive on.
            index = checkpoint.turns_completed + 1
            await rt._emit(
                RunEvent(
                    event_type=EventType.AGENT_TURN_STARTED,
                    trace_id=trace_id,
                    thread_id=thread.thread_id,
                    agent_id=persona.agent_id,
                    payload={"turn": index},
                )
            )
            result = await rt._continue_turn(
                persona, thread, ctx, trace_id, llm_config=resume_llm
            )
            await rt._emit_turn_completed(trace_id, thread, persona, index, result)
            loop = await rt._drive_loop(
                persona,
                task,
                thread,
                ctx,
                trace_id,
                turns=[result],
                max_turns=checkpoint.max_turns,
                cost_budget=checkpoint.cost_budget,
                llm_config=resume_llm,
                hitl=hitl,
                stop_on_no_progress=False,
                turns_offset=checkpoint.turns_completed,
                cost_offset=checkpoint.cost_completed,
            )
            self.record_resume_final_output(checkpoint_id, loop)
            return loop

    def record_resume_final_output(
        self, checkpoint_id: str, loop: AgentLoopResult
    ) -> None:
        """Persist the resolved member's FINAL output onto its terminal checkpoint (#2).

        The crash-recovery anchor for orchestration HITL. By the time this runs the
        gated tool has fired exactly once and the checkpoint is already resolved
        (``approved``/``rejected``); the member's final answer is only produced by the
        drive loop AFTER that terminal save, so it is written back here — DURABLY, onto
        the SAME store row that holds the claim + idempotency ledger, BEFORE the caller
        returns (and therefore before the orchestration graph advance persists). If a
        crash then strikes between the member resolving and the graph persisting its
        advance, the graph recovery reads this text back and threads the REAL member
        output downstream instead of an empty string.

        A member that paused AGAIN on a second gated tool has NOT produced a final
        answer (the original checkpoint stays terminal but re-pauses into a fresh
        checkpoint), so nothing is recorded — ``final_output`` stays ``None``. The
        write merges onto a freshly LOADED row so the ledger written during execution
        is preserved, and never resurrects a since-pruned checkpoint.
        """
        rt = self._rt
        assert rt._checkpoint_store is not None  # guaranteed by the caller
        if loop.stopped_reason == "awaiting_approval":
            return
        latest = rt._checkpoint_store.load(checkpoint_id)
        if latest is None:  # pragma: no cover - pruned mid-resume; nothing to anchor
            return
        final_text = (loop.final.output_text or "") if loop.turns else ""
        rt._checkpoint_store.save(
            latest.model_copy(update={"final_output": final_text})
        )

    def save_checkpoint(
        self,
        persona: Persona,
        task: Task,
        thread: ChatThread,
        ctx: dict[str, Any],
        llm_config: LLMConfig | None,
        max_turns: int,
        cost_budget: float | None,
        turns_completed: int,
        cost_completed: float,
        pending: list[PendingToolCall],
    ) -> AgentCheckpoint:
        """Persist a paused run as a durable checkpoint and return it."""
        rt = self._rt
        assert rt._checkpoint_store is not None
        checkpoint = AgentCheckpoint(
            persona=persona.model_dump(mode="json"),
            task=task.model_dump(mode="json"),
            thread=thread.model_dump(mode="json"),
            ctx=ctx,
            llm_config=llm_config.model_dump(mode="json")
            if llm_config is not None
            else None,
            max_turns=max_turns,
            cost_budget=cost_budget,
            turns_completed=turns_completed,
            cost_completed=cost_completed,
            pending_tool_calls=pending,
        )
        rt._checkpoint_store.save(checkpoint)
        return checkpoint
