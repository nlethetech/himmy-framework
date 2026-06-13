"""Studio run feedback (P0-A): capture a human's thumbs up/down on a run.

The single least-noisy learning signal there is — and the one a small local judge
can never substitute for. Every later loop (exemplar harvesting, prompt
refinement, lesson extraction) reads from here, so the job of this module is to
*record* the signal durably and auditably:

* the current verdict + note land on the run row (``studio_runs.feedback_*``), the
  fast read for list/detail views and analytics tallies;
* every write ALSO appends an immutable, versioned ``run_feedback`` record to the
  durable entity spine (:func:`himmy.api.studio_entities.get_entity_registry`), so
  the history of how a human rated a run survives independent of the run sidecar
  and is erasure-scopable per subject.

Re-rating a run is allowed: the run row is overwritten, and the spine gains a new
version — a full, ordered audit of every verdict change.
"""

from __future__ import annotations

from pydantic import BaseModel

from himmy.api.studio_entities import get_entity_registry
from himmy.api.studio_runs import FEEDBACK_VERDICTS, get_run_store
from himmy.core.ids import utc_now_iso
from himmy.entities.records import stable_id_for

#: Entity-spine kind for human run feedback.
RUN_FEEDBACK_KIND = "run_feedback"


def _feedback_subject() -> str:
    """The erasure/governance subject feedback is scoped to.

    Mirrors the memory subsystem's ``HIMMY_MEMORY_SUBJECT`` scope so a single
    ``erase --subject`` reaches a person's memories and their feedback alike.
    """
    import os

    return os.environ.get("HIMMY_MEMORY_SUBJECT", "default")


class RunFeedback(BaseModel):
    """Human feedback on a run, as recorded (mirrors the spine payload)."""

    run_id: str
    verdict: str  # "up" | "down"
    note: str | None = None
    actor: str = "human"
    created_at: str
    version: int = 1  # the run_feedback spine version this write produced


def _feedback_stable_id(run_id: str) -> str:
    """Deterministic spine stable_id for a run's feedback version chain."""
    return stable_id_for(run_id, namespace=RUN_FEEDBACK_KIND)


def record_feedback(
    run_id: str,
    *,
    verdict: str,
    note: str | None = None,
    actor: str = "human",
    source: str = "studio",
    require_studio_run: bool = True,
    extra_metadata: dict[str, str] | None = None,
) -> RunFeedback | None:
    """Record feedback on a run, appending an immutable spine version.

    The entity spine is the source of truth; the run row (``studio_runs``) is a
    best-effort current-state copy updated only when a row for ``run_id`` exists.

    ``require_studio_run`` (the Studio default) makes a missing run row return
    ``None`` so the API can answer 404. CLI turns have no Studio run row, so the
    REPL passes ``require_studio_run=False`` and the feedback lands on the spine
    keyed by the turn's ``trace_id``.

    Raises ``ValueError`` on an unknown verdict so the route can return 422.
    """
    if verdict not in FEEDBACK_VERDICTS:
        raise ValueError(
            f"unknown feedback verdict {verdict!r}; "
            f"expected one of {sorted(FEEDBACK_VERDICTS)}"
        )
    note = (note or "").strip() or None
    at = utc_now_iso()

    found = get_run_store().set_feedback(run_id, verdict=verdict, note=note, at=at)
    if require_studio_run and not found:
        return None

    payload = {
        "run_id": run_id,
        "verdict": verdict,
        "note": note,
        "actor": actor,
        "created_at": at,
    }
    metadata: dict[str, str] = {
        "run_id": run_id,
        "subject_id": _feedback_subject(),
        "source": source,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    record = get_entity_registry().new_version(
        stable_id=_feedback_stable_id(run_id),
        kind=RUN_FEEDBACK_KIND,
        payload=payload,
        metadata=metadata,
    )
    return RunFeedback(
        run_id=run_id,
        verdict=verdict,
        note=note,
        actor=actor,
        created_at=at,
        version=record.version,
    )


def get_feedback(run_id: str) -> RunFeedback | None:
    """Current feedback on a run from the spine (latest version), or None."""
    latest = get_entity_registry().get_latest(_feedback_stable_id(run_id))
    if latest is None or latest.kind != RUN_FEEDBACK_KIND:
        return None
    p = latest.payload
    return RunFeedback(
        run_id=run_id,
        verdict=str(p.get("verdict", "")),
        note=p.get("note"),
        actor=str(p.get("actor", "human")),
        created_at=str(p.get("created_at", latest.created_at)),
        version=latest.version,
    )


def feedback_history(run_id: str) -> list[RunFeedback]:
    """Every feedback version for a run, oldest first (the audit trail)."""
    records = get_entity_registry().get_history(_feedback_stable_id(run_id))
    out: list[RunFeedback] = []
    for r in records:
        if r.kind != RUN_FEEDBACK_KIND:
            continue
        p = r.payload
        out.append(
            RunFeedback(
                run_id=run_id,
                verdict=str(p.get("verdict", "")),
                note=p.get("note"),
                actor=str(p.get("actor", "human")),
                created_at=str(p.get("created_at", r.created_at)),
                version=r.version,
            )
        )
    return out


__all__ = [
    "RUN_FEEDBACK_KIND",
    "RunFeedback",
    "record_feedback",
    "get_feedback",
    "feedback_history",
]
