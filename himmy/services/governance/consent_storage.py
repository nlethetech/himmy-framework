"""WS4.6 — ``ConsentGatedStorage``: enforce purpose=RETAIN at the persistence boundary.

A transparent decorator over the :class:`~himmy.services.storage.service.StorageService`
facade (so it covers Postgres too — both satisfy the same protocol). For every
subject-bearing ``save_*`` it consults the injected consent decider for purpose
:attr:`~himmy.services.governance.consent.Purpose.RETAIN`:

* :attr:`Effect.ALLOW` ⇒ delegate to the wrapped store (persist).
* :attr:`Effect.EPHEMERAL` / :attr:`Effect.DENY` ⇒ **skip the write** (the record was used
  for the live request only) and emit a ``consent_denied_persist`` security event.

A subject-bearing record whose ``subject_id`` cannot be resolved **fails closed** (skip) in a
governed deployment. Every non-gated method is delegated unchanged via ``__getattr__``; the
gated method set is exposed as :attr:`GATED_SAVE_METHODS` so a contract test fails loudly if
``StorageService`` grows a new subject-bearing sink that this decorator does not cover.

This decorator is constructed **only** in the governed branch of the API container; the
zero-config path keeps the bare ``StorageService`` and is byte-for-byte unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from himmy.services.governance.consent import Effect, Purpose
from himmy.services.governance.consent_resolver import SubjectResolver

logger = logging.getLogger("himmy.services.governance")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable

    from himmy.services.audit.log import SecurityAuditLog
    from himmy.services.governance.consent import Decision
    from himmy.services.storage.models import (
        EpisodicMemoryObject,
        MemoryObject,
        RecommendationItem,
        RunRecord,
    )

    ConsentDecider = Callable[[str, Purpose], Decision]

#: The ``save_*`` methods this decorator gates. A contract test asserts every
#: subject-bearing StorageService sink is covered here (sink-drift guard).
GATED_SAVE_METHODS = frozenset(
    {
        "save_run",
        "save_run_if_absent_by_idempotency",
        "save_run_if_under_quota",
        "save_run_if_owner",
        "save_snapshot",
        "save_memory",
        "save_episodic_memory",
        "save_recommendation",
    }
)


class ConsentGatedStorage:
    """Gate subject-bearing writes at purpose=RETAIN; delegate everything else."""

    def __init__(
        self,
        inner: Any,
        *,
        decider: ConsentDecider,
        resolver: SubjectResolver | None = None,
        audit: SecurityAuditLog | None = None,
    ) -> None:
        """Wrap ``inner`` storage; ``decider`` resolves a RETAIN decision per subject."""
        self._inner = inner
        self._decider = decider
        self._resolver = resolver or SubjectResolver()
        self._audit = audit

    # ---------------------------------------------------------------- gating
    def _may_persist(self, obj: object) -> bool:
        """Whether ``obj`` may be written; emit a denial event + skip when not."""
        if not self._resolver.is_subject_bearing(obj):
            return True
        subject = self._resolver.subject_of(obj)
        if not subject:  # subject-bearing but unresolved → fail closed
            self._audit_deny("", type(obj).__name__, "unresolved subject")
            return False
        decision = self._decider(subject, Purpose.RETAIN)
        if decision.effect is Effect.ALLOW:
            return True
        self._audit_deny(subject, type(obj).__name__, decision.reason)
        return False

    def _audit_deny(self, subject: str, resource: str, reason: str) -> None:
        """Record a ``consent_denied_persist`` security event + structured warning.

        The security-event log is opt-in (``self._audit`` is often ``None`` off the
        governed API path); the WARNING ensures a consent denial is never wholly
        silent. ``subject`` is intentionally omitted from the log to avoid leaking a
        subject identifier into operational logs.
        """
        logger.warning("consent denied persist for resource=%s: %s", resource, reason)
        if self._audit is None:
            return
        from himmy.services.audit.models import SecurityEvent

        self._audit.record(
            SecurityEvent(
                event_type="consent_denied_persist",
                outcome="deny",
                actor={"subject": subject} if subject else {},
                resource=resource,
                action="persist",
                detail=reason,
            )
        )

    # --------------------------------------------------------- gated writes
    async def save_run(self, run: RunRecord) -> RunRecord:
        """Persist a run unless RETAIN is not consented (then ephemeral)."""
        if self._may_persist(run):
            return cast("RunRecord", await self._inner.save_run(run))
        return run

    async def save_run_if_absent_by_idempotency(
        self, run: RunRecord
    ) -> tuple[RunRecord, bool]:
        """Idempotent create, gated. Decision is computed before the atomic section."""
        if self._may_persist(run):
            return cast(
                "tuple[RunRecord, bool]",
                await self._inner.save_run_if_absent_by_idempotency(run),
            )
        return run, True

    async def save_run_if_under_quota(
        self, run: RunRecord, *, cap: int
    ) -> tuple[RunRecord, bool]:
        """Quota-gated atomic create, also consent-gated. Decision precedes the atomic
        section; an unconsented subject's run is ephemeral and consumes no quota."""
        if self._may_persist(run):
            return cast(
                "tuple[RunRecord, bool]",
                await self._inner.save_run_if_under_quota(run, cap=cap),
            )
        return run, True

    async def save_run_if_owner(self, run: RunRecord, owner_id: str) -> bool:
        """Owner-conditional terminal write, consent-gated. An unconsented subject's run is
        not persisted (returns ``False``, the same 'did not land' signal as a lost-owner CAS)."""
        if self._may_persist(run):
            return cast("bool", await self._inner.save_run_if_owner(run, owner_id))
        return False

    async def save_snapshot(self, snapshot: Any) -> Any:
        """Persist a context snapshot unless RETAIN is not consented."""
        if self._may_persist(snapshot):
            return await self._inner.save_snapshot(snapshot)
        return snapshot

    async def save_memory(self, obj: MemoryObject) -> MemoryObject:
        """Persist a cognitive memory object unless RETAIN is not consented."""
        if self._may_persist(obj):
            return cast("MemoryObject", await self._inner.save_memory(obj))
        return obj

    async def save_episodic_memory(
        self, obj: EpisodicMemoryObject
    ) -> EpisodicMemoryObject:
        """Persist an episodic memory object unless RETAIN is not consented."""
        if self._may_persist(obj):
            return cast(
                "EpisodicMemoryObject", await self._inner.save_episodic_memory(obj)
            )
        return obj

    async def save_recommendation(self, item: RecommendationItem) -> RecommendationItem:
        """Persist a recommendation unless RETAIN is not consented."""
        if self._may_persist(item):
            return cast(
                "RecommendationItem", await self._inner.save_recommendation(item)
            )
        return item

    # ------------------------------------------------------------ delegation
    def __getattr__(self, name: str) -> Any:
        """Delegate every non-gated attribute/method to the wrapped store."""
        return getattr(self._inner, name)


__all__ = ["ConsentGatedStorage", "GATED_SAVE_METHODS"]
