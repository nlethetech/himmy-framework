"""WS4.6 — ``ConsentFilteredExporter``: the single TRAIN-purpose export funnel.

Any dataset / SFT / replay-corpus builder that derives training material from persisted
runs MUST route through :meth:`ConsentFilteredExporter.filter`. It drops every record whose
subject lacks an :attr:`~himmy.services.governance.consent.Effect.ALLOW` decision for purpose
:attr:`~himmy.services.governance.consent.Purpose.TRAIN`, so a subject who never opted in to
training can never appear in an exported corpus — even if their run was retained (RETAIN and
TRAIN are independent purposes).

The decider is injected (``Callable[[str, Purpose], Decision]``, e.g.
:meth:`~himmy.services.governance.consent_ledger.ConsentLedger.decision`) so the exporter has
no storage/runtime coupling and is unit-testable with zero I/O. Subject identity is read via
an injected ``subject_of`` extractor (typically
:meth:`~himmy.services.governance.consent_resolver.SubjectResolver.subject_of`), so the
exporter is record-type agnostic.

This exporter is constructed **only** when a deployment opts in to governed export; the
zero-config path never builds it, so the offline corpus pipeline is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

from himmy.services.governance.consent import Effect, Purpose

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Callable, Iterable

    from himmy.services.governance.consent import Decision

    ConsentDecider = Callable[[str, Purpose], Decision]
    SubjectOf = Callable[[object], str | None]

T = TypeVar("T")


class ConsentFilteredExporter:
    """Filter an iterable of records to only TRAIN-consented subjects."""

    def __init__(
        self,
        *,
        decider: ConsentDecider,
        keep_unresolved: bool = False,
    ) -> None:
        """Wire the TRAIN decider.

        ``decider`` resolves a :class:`Decision` for ``(subject_id, Purpose.TRAIN)``.
        ``keep_unresolved`` controls a record whose subject cannot be resolved: it defaults
        to ``False`` (**fail closed** — a record with no resolvable subject is dropped, never
        exported), matching the secure-by-default posture of the persistence gates.
        """
        self._decider = decider
        self._keep_unresolved = keep_unresolved

    def allows(self, subject_id: str | None) -> bool:
        """Whether a subject may appear in a TRAIN export (unresolved ⇒ fail-closed)."""
        if not subject_id:
            return self._keep_unresolved
        return self._decider(subject_id, Purpose.TRAIN).effect is Effect.ALLOW

    def filter(self, records: Iterable[T], *, subject_of: SubjectOf) -> list[T]:
        """Drop every record whose subject is not TRAIN-consented.

        Args:
            records: the candidate records to export.
            subject_of: extracts the subject id from a record (``None`` ⇒ unresolved).

        Returns:
            The subset that may be used for training, preserving input order.
        """
        return [r for r in records if self.allows(subject_of(r))]


__all__ = ["ConsentFilteredExporter"]
