"""Cross-cutting run side-effects for :class:`~himmy.application.services.RunAppService`.

Extracted from :mod:`himmy.application.services` as the ``RunSideEffects`` collaborator in
the staged decomposition of ``RunAppService`` (LANE runapp, step 4 — side effects): it owns
the small cohesive band of helpers that build the per-run tool-capability gate, derive the
within-tenant subject axis, and fire the two best-effort lineage/projection side effects
(conversation re-projection + run<->agent link).

Behaviour is BYTE-IDENTICAL to the former inline methods:

- :meth:`build_tool_authorizer` returns ``None`` when no RBAC policy is wired (the offline /
  zero-config default) so tool dispatch is byte-unchanged; when a policy IS wired it rebuilds
  ``ToolCapabilityAuthorizer.from_actor`` from the *persisted* actor — the same descriptor the
  ambient tool-authorizer contextvar is bound from at the call site, whose set/reset lifetime
  stays in ``services`` untouched (the confused-deputy gate on the approved write path),
- :meth:`subject_scope_from_actor` mirrors the single-agent namespacing (``None`` unless a
  ``subject_scoped`` per-user actor),
- the two projections (:meth:`notify_conversation_sink` / :meth:`project_run_agent_link`) are
  best-effort — any error is swallowed with the SAME warning, provenance not the work,
- the collaborator reads its handles LIVE through the shared :class:`_RunContext` (never a
  construction-time snapshot), so a re-pointed ``registry``/``conversation_sink``/``access_policy``
  is observed at once.

``RunAppService``'s former private methods (``_build_tool_authorizer`` /
``_subject_scope_from_actor`` / ``_notify_conversation_sink`` / ``_project_run_agent_link``)
delegate here as thin shims, so every internal caller and test poke stays byte-identical.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from himmy.application.services import (
    _maybe_await,
    logger,
)
from himmy.entities.records import stable_id_for
from himmy.services.storage.models import (
    AgentDefRecord,
    RunRecord,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids import cycles
    from himmy.application.run_context import _RunContext


class RunSideEffects:
    """Cross-cutting run helpers (tool-authz gate, subject scope, lineage projections).

    Holds no state of its own beyond the shared context handle; every method reads
    ``access_policy``/``conversation_sink``/``registry`` live from the context so behaviour
    matches the former inline implementation byte-for-byte.
    """

    def __init__(self, *, context: _RunContext) -> None:
        """Wire the shared run-lifecycle context (source of the live handles)."""
        self._ctx = context

    def build_tool_authorizer(self, actor: dict[str, Any] | None) -> Any:
        """Rebuild this run's tool-capability gate from its persisted actor (P0).

        Returns ``None`` when no RBAC policy is wired (the offline / zero-config default
        and every programmatic caller) so ``build_runtime_for_spec`` builds a per-run
        runtime with NO authorizer — tool dispatch byte-identical to before. When a policy
        IS wired the gate is rebuilt from the actor descriptor: an actor carrying
        ``tool_authz_enforce`` yields an ENFORCING authorizer over its recorded roles; the
        ANONYMOUS / all_tenants offline actor (no flag) yields a NON-enforcing pass-through.
        Building from the persisted actor (not a live Principal) is what lets the gate
        survive the leased-dispatch recovery path, where a fresh process re-executes the
        run with no in-memory principal.
        """
        if self._ctx.access_policy is None:
            return None
        from himmy.services.tools.capability import ToolCapabilityAuthorizer

        return ToolCapabilityAuthorizer.from_actor(actor, self._ctx.access_policy)

    @staticmethod
    def subject_scope_from_actor(actor: dict[str, Any] | None) -> str | None:
        """The within-tenant subject axis for a run's tool stores, from its persisted actor.

        Mirrors the single-agent path: under a ``subject_scoped`` per-user actor the run's
        memory/KB/tasks/notes packs are namespaced by the user so two users of ONE tenant never
        read each other's facts/docs. The flag is persisted by ``Principal.actor_metadata`` (set
        only when actually ``subject_scoped`` + not a ``tenant_admin``), so a non-subject-scoped /
        offline run returns ``None`` — byte-for-byte unchanged.
        """
        if actor and actor.get("subject_scoped"):
            return actor.get("subject")
        return None

    async def notify_conversation_sink(self, run: RunRecord) -> None:
        """Re-project a thread-linked run's updated ChatThread (T2g, best-effort).

        Invoked when a run that the thread router started (carrying
        ``metadata['conversation_id']``) reaches a terminal / AWAITING_APPROVAL state, so
        the ConversationStore projection reflects the latest turns even when the run
        finished on a background task the router never awaited (an approval-resume). A
        no-op when no sink is wired or the run is not thread-linked; any sink error is
        swallowed (provenance, not the work).
        """
        if self._ctx.conversation_sink is None:
            return
        conversation_id = (run.metadata or {}).get("conversation_id")
        if not conversation_id:
            return
        try:
            await _maybe_await(self._ctx.conversation_sink(conversation_id, run))
        except Exception:  # noqa: BLE001 - conversation projection is best-effort
            logger.warning(
                "failed to project conversation %s for run %s",
                conversation_id,
                run.run_id,
            )

    async def project_run_agent_link(
        self, run: RunRecord, agent_def: AgentDefRecord
    ) -> None:
        """Register the stored-agent node + link the run's thread -> agent (T2e).

        Ensures the ``agent`` entity exists in the shared spine (idempotent,
        content-addressed) and draws a ``run_of_agent`` edge from the run's
        ``chat_thread`` hub to that agent node, so a run launched by ``agent_id`` joins
        the agent's lineage graph. Best-effort: a projection failure never fails the run
        (it is provenance, not the work).
        """
        if self._ctx.registry is None or not run.thread_id:
            return
        try:
            agent_record = await _maybe_await(
                self._ctx.registry.register(agent_def.to_record())
            )
            thread_sid = stable_id_for(run.thread_id, namespace="chat_thread")
            thread_record = await _maybe_await(self._ctx.registry.get_latest(thread_sid))
            if thread_record is None:
                return
            # Dedupe the edge so a re-run of the same thread/agent never doubles it.
            existing = await _maybe_await(
                self._ctx.registry.links_from(thread_record.record_id)
            )
            for link in existing:
                if (
                    link.to_record_id == agent_record.record_id
                    and link.relation == "run_of_agent"
                ):
                    return
            await _maybe_await(
                self._ctx.registry.link(
                    from_record_id=thread_record.record_id,
                    to_record_id=agent_record.record_id,
                    relation="run_of_agent",
                    metadata={"run_id": run.run_id, "agent_id": agent_def.agent_id},
                )
            )
        except Exception:  # pragma: no cover - lineage projection is best-effort
            logger.warning(
                "failed to project run<->agent lineage for run %s agent %s",
                run.run_id,
                agent_def.agent_id,
            )
