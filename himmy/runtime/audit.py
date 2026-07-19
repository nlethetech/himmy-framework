"""Audit-spine emitter for :class:`SingleAgentRuntime` (event fan-out + projection).

Extracted verbatim from ``single_agent.py`` (P3 decomposition, lane ``runtime``
step ``audit``). :class:`AuditEmitter` owns the runtime's whole audit-spine
concern in one cohesive object:

* the best-effort event fan-out across every configured sink — durable memory
  store, entity-registry projection, observability span, Prometheus metrics, and
  the caller-facing ``on_event`` callbacks — in the exact same ORDER, with each
  sink isolated so one failing sink can never break the run (``emit``);
* projecting run artefacts (messages, thread versions, personas/prompts) into the
  entity registry and wiring their documented lineage relations — the group that
  previously lived in the ``_EntityRegistrar`` collaborator (``register_entity`` /
  ``register_message`` / ``register_thread_version`` / ``link_lineage``);
* the governed-run subject stamp read off the :data:`_CURRENT_SUBJECT` contextvar
  (``subject_metadata``), the single source of truth shared by the emit + register
  paths; and
* the opt-in thread persistence (``maybe_save_thread``).

The runtime constructs one of these in ``__init__`` and its former
``_emit`` / ``_register_*`` / ``_link_lineage`` / ``_subject_metadata`` /
``_maybe_save_thread`` methods become thin delegating shims. Behaviour — event
ORDER, the swallowed-sink ``log.warning`` + ``event_sink_drops_total`` metric, the
``{subject_id: ...}`` stamp, ``CancelledError`` propagation, and every exception
type — is byte-for-byte identical to the pre-extraction inline code.

The :data:`_CURRENT_SUBJECT` contextvar is defined HERE (it is the audit-spine's
subject source) and re-imported by ``single_agent`` so its historical import path
(``from himmy.runtime.single_agent import _CURRENT_SUBJECT``) and the
``_subject_scope`` set/reset stay valid; the :class:`contextvars.ContextVar` keeps
the stamp correct under concurrent runs.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.agents.personas.persona import Persona
    from himmy.core.events import RunEvent
    from himmy.runtime.single_agent import SingleAgentRuntime


# Module logger. Kept under the ``himmy.runtime.single_agent`` name (rather than
# this module's ``__name__``) so the swallowed-sink warnings surface on the exact
# logger they did before the extraction — behaviour-preserving for operators and
# any log-scraping consumer keyed on the logger identity.
log = logging.getLogger("himmy.runtime.single_agent")


# WS4.6 — the human data subject participating in the *current* run, set ONLY when a
# ``consent_decider`` is wired (governed deployments) and the task carries a
# ``context_subject_id``. It is published by ``SingleAgentRuntime._subject_scope``
# around EVERY public entry point that emits subject-bearing spine records — all FIVE
# of them: ``run_task`` (and its ``run_task_detailed`` alias), ``run_agent_loop``,
# ``continue_turn``, ``stream_task``, and ``resume_agent_loop`` (the HITL-resume path,
# which rebuilds ``ctx`` from the checkpoint and then emits resumed run events / tool
# messages / a bumped thread version) — so the multi-turn, streaming AND resume paths
# are governed too (not just ``run_task``). Otherwise their records would be
# subject-less and the ConsentAwareRegistry would fail closed and silently drop them
# even for a consented subject.
# ``emit`` / ``register_message`` / ``register_thread_version`` stamp this onto the
# ``run_event`` / ``message`` / ``chat_thread`` record metadata so a
# ``ConsentAwareRegistry`` can resolve (and gate / crypto-shred) the subject behind
# those otherwise subject-less spine records. A :class:`contextvars.ContextVar` keeps
# this correct under concurrent runs and the default ``None`` keeps the ungoverned path
# byte-identical — no metadata is ever stamped when no decider is configured.
_CURRENT_SUBJECT: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "himmy_run_subject", default=None
)


def _count_sink_drop(sink: str) -> None:
    """Increment the bounded ``event_sink_drops_total{sink}`` counter (best-effort).

    Complements the ``log.warning`` in ``emit`` with an operator-scrapable metric so a
    silent audit-spine write loss is visible on ``/metrics``. Never raises — observability
    must never break a run (the sink drop it records is already itself a swallowed error).
    """
    try:
        from himmy.services.observability.metrics import get_registry

        get_registry().event_sink_drops_total.inc((sink,))
    except Exception:  # pragma: no cover - metrics must never break a run
        pass


class AuditEmitter:
    """Owns the audit-spine fan-out + entity projection for one runtime.

    A thin collaborator over the runtime's live sink wiring (``memory_store``,
    ``entity_registry``, ``_on_event``, ``save_threads``), all read off the runtime
    at call time so a caller that re-wires a sink after construction still steers this
    exactly as the inline code did. Absorbs the former ``_EntityRegistrar`` — the
    registry-projection helpers now live here alongside ``emit`` so the whole spine
    concern (fan-out + registration + lineage + save) is one object.
    """

    def __init__(self, runtime: SingleAgentRuntime) -> None:
        self._rt = runtime

    # --------------------------------------------------------------- subject
    @staticmethod
    def subject_metadata() -> dict[str, Any] | None:
        """The ``{subject_id: ...}`` stamp for the current run, or ``None`` (ungoverned).

        Returns ``None`` whenever no governed run subject is published (the offline /
        ungoverned path), so ``to_record(metadata=None)`` keeps the projected record
        byte-identical to a pre-WS4.6 runtime. When a governed run names a subject, the
        stamp lets a :class:`ConsentAwareRegistry` resolve + gate the otherwise
        subject-less spine records.
        """
        subject = _CURRENT_SUBJECT.get()
        return {"subject_id": subject} if subject else None

    # --------------------------------------------------------------- entities
    def register_entity(self, obj: Any, *, stamp_subject: bool = False) -> Any:
        """Register a domain object's record when a registry is wired; else None.

        ``stamp_subject`` is only set for the genuinely subject-bearing run artefacts
        (messages, threads); infrastructure records (persona/prompt/agent) are never
        stamped so a governed ``registry.query(metadata={'subject_id': ...})`` returns
        only real subject data.
        """
        entity_registry = self._rt.entity_registry
        if entity_registry is None:
            return None
        to_record = getattr(obj, "to_record", None)
        if to_record is None:
            return None
        try:
            metadata = self.subject_metadata() if stamp_subject else None
            record = to_record(metadata=metadata)
            return entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def register_message(self, message: Any) -> Any:
        """Register a Message entity (kind="message") when a registry is present."""
        return self.register_entity(message, stamp_subject=True)

    def register_thread_version(self, thread: Any) -> Any:
        """Project the current chat_thread version into a record (when a registry).

        RO-8: the version bump now happens in the run body (regardless of
        registry), so this helper only projects the record at the already-bumped
        version. Returns ``None`` when no registry is wired.
        """
        entity_registry = self._rt.entity_registry
        if entity_registry is None:
            return None
        try:
            record = thread.to_record(metadata=self.subject_metadata())
            return entity_registry.register(record)
        except Exception:  # pragma: no cover - defensive
            return None

    def link_lineage(
        self,
        *,
        persona_record: Any,
        prompt_record: Any,
        thread_record: Any,
        snapshot: Any,
        persona: Persona,
        thread: Any,
    ) -> None:
        """Wire the documented lineage relations between run artefacts."""
        entity_registry = self._rt.entity_registry
        if entity_registry is None or thread_record is None:
            return
        link = entity_registry.link
        try:
            if persona_record is not None:
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=persona_record.record_id,
                    relation="uses_persona",
                )
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=persona_record.record_id,
                    relation="thread_for_agent",
                )
            if prompt_record is not None:
                link(
                    from_record_id=thread_record.record_id,
                    to_record_id=prompt_record.record_id,
                    relation="in_thread",
                )
            if snapshot is not None:
                snapshot_record = getattr(snapshot, "to_record", None)
                if snapshot_record is not None:
                    sr = entity_registry.register(snapshot.to_record())
                    link(
                        from_record_id=thread_record.record_id,
                        to_record_id=sr.record_id,
                        relation="built_from",
                    )
                    link(
                        from_record_id=sr.record_id,
                        to_record_id=thread_record.record_id,
                        relation="observed_in_run",
                    )
        except Exception:  # pragma: no cover - defensive
            pass

    # --------------------------------------------------------------- fan-out
    async def emit(self, event: RunEvent) -> None:
        """Best-effort fan-out of one event to all configured sinks.

        Order: storage (the durable spine) -> entity registry -> observability
        span (invariant #6) -> caller-facing ``on_event`` callbacks (RO-6). Every
        sink is isolated so one failing sink can never break the run or starve
        the others, and ``CancelledError`` is honored so a cancelled run unwinds.
        """
        rt = self._rt
        if rt.memory_store is not None:
            appender = getattr(rt.memory_store, "append_event", None)
            if appender is not None:
                try:
                    await appender(event)
                except asyncio.CancelledError:
                    raise
                except Exception:  # pragma: no cover - defensive
                    # Durable audit-spine write lost; keep the run alive but signal.
                    log.warning(
                        "memory_store.append_event failed (event dropped from "
                        "durable spine): event_type=%s trace_id=%s",
                        getattr(event.event_type, "value", event.event_type),
                        event.trace_id,
                        exc_info=True,
                    )
                    _count_sink_drop("memory_store")
        if rt.entity_registry is not None:
            try:
                rt.entity_registry.register(
                    event.to_record(metadata=self.subject_metadata())
                )
            except Exception:  # pragma: no cover - defensive
                # Entity-registry projection lost; keep the run alive but signal.
                log.warning(
                    "entity_registry.register failed (event dropped from "
                    "projection): event_type=%s trace_id=%s",
                    getattr(event.event_type, "value", event.event_type),
                    event.trace_id,
                    exc_info=True,
                )
                _count_sink_drop("entity_registry")
        try:
            from himmy.services.observability import emit_event_span

            emit_event_span(event)
        except Exception:  # pragma: no cover - defensive
            pass
        # Prometheus metrics: translate the event into bounded-cardinality counters.
        try:
            from himmy.services.observability.metrics import get_metrics_sink

            await get_metrics_sink().append_event(event)
        except Exception:  # pragma: no cover - defensive
            pass
        # RO-6: stream incremental progress to caller-supplied callbacks.
        for callback in rt._on_event:
            try:
                await callback(event)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - never let a listener break the run
                pass

    async def maybe_save_thread(self, thread: Any) -> None:
        """Persist the thread when ``save_threads`` and a memory store are present."""
        rt = self._rt
        if not rt.save_threads or rt.memory_store is None:
            return
        saver = getattr(rt.memory_store, "save_thread", None)
        if saver is None:
            return
        try:
            await saver(thread)
        except Exception:  # pragma: no cover - defensive
            pass
