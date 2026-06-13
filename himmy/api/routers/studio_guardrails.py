"""API kernel: Studio guardrails router — the read-only safety-layer viewer (T3e).

Mounted under ``/api/studio/guardrails`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). It is the GUI sibling of the CLI's
``himmy guardrails`` command + its live ``🛡`` shield line — a *trust* surface, so
the framework's safety layer is something you can SEE rather than something buried.
One endpoint:

* ``GET /``  — the built-in guardrail catalog (the SAME rows ``himmy guardrails``
  prints, via :func:`himmy.cli.guardrails_view.builtin_rows`) plus a recent-firings
  list.

DATA SOURCE (reviewer must_fix): firings are read from the ALREADY-PRESENT
:class:`~himmy.api.studio_runs.StudioRunStore` (``.himmy/studio.db``) — every run's
cognition trace records a ``GUARDRAIL_APPLIED`` event as a ``CognitionStep`` of
kind ``safety`` (see :meth:`himmy.api.studio_service` ``_translate``), carrying the
verdict's ``stage``/``blocked``/``redacted``/``flags``/``reasons``. We scan the most
recent runs and surface each safety step linked to its ``run_id`` — NOT an unbuilt
"coherence RunAppService run-frame" source, so this ships TODAY over the existing
Studio store.

Offline-first: with no runs recorded the firings list is simply empty (the screen
renders its clean empty state); the catalog is always non-empty (the built-ins are
declared in code), so this endpoint never 500s.
"""

from __future__ import annotations

from fastapi import Query, Request
from pydantic import BaseModel

from himmy.api.routers.studio_common import build_studio_router

router = build_studio_router("guardrails", tag="studio-guardrails")

#: How many recent runs to scan for firings (bounds the studio.db read so a huge
#: history can never stall the GUI). A run can fire several guardrails, so the
#: firings cap below is independent.
_RUN_SCAN_DEFAULT = 200
_RUN_SCAN_MAX = 1000

#: Hard cap on firings returned (newest first) — keeps the response bounded.
_FIRINGS_DEFAULT = 50
_FIRINGS_MAX = 500

#: The CognitionStep kind a ``GUARDRAIL_APPLIED`` event is recorded under (the
#: Studio cognition translator emits ``kind="safety"`` for every guardrail firing).
_SAFETY_KIND = "safety"


class GuardrailInfo(BaseModel):
    """One built-in guardrail in the catalog (name + one-line description)."""

    name: str
    description: str


class GuardrailFiring(BaseModel):
    """One guardrail firing, linked back to the run it happened in.

    Mirrors the shape the CLI's ``🛡`` shield line renders: the guardrail family
    (``name``), what it did (``action`` = blocked|redacted|flagged), the matched
    label classes/reasons (``detail``), and the input/output ``stage`` — plus the
    ``run_id`` so the GUI can deep-link to the run's trace.
    """

    run_id: str
    agent_name: str | None = None
    created_at: str
    seq: int
    name: str  # guardrail family, e.g. "pii" / "injection" / "grounding"
    action: str  # "blocked" | "redacted" | "flagged"
    stage: str | None = None  # "input" | "output"
    detail: str = ""
    blocked: bool = False
    redacted: bool = False
    flags: list[str] = []
    reasons: list[str] = []


class GuardrailsResponse(BaseModel):
    """The guardrail catalog + the recent firings list (the screen's full payload)."""

    builtins: list[GuardrailInfo]
    firings: list[GuardrailFiring]
    firing_count: int  # firings surfaced (after the cap)
    runs_scanned: int  # how many recent runs were inspected


# ----------------------------------------------------------------------- helpers


def _firing_name(flags: list[str]) -> str:
    """The guardrail family from the verdict flags (``pii:email`` → ``pii``).

    Mirrors :func:`himmy.cli.guardrails_view._guardrail_label`: the leading token
    before ``:`` is the family. Falls back to a generic label for a bare redaction
    that flagged nothing.
    """
    for flag in flags:
        token = str(flag).split(":", 1)[0].strip()
        if token:
            return token
    return "guardrail"


def _firing_action(*, blocked: bool, redacted: bool) -> str:
    """``blocked`` / ``redacted`` / ``flagged`` — what the guardrail did."""
    if blocked:
        return "blocked"
    if redacted:
        return "redacted"
    return "flagged"


def _firing_detail(flags: list[str], reasons: list[str]) -> str:
    """A short human detail: matched label classes, else the first reason.

    Close to :func:`himmy.cli.guardrails_view._detail`: ``pii:email`` → ``email``.
    The one deliberate difference is the bare-family case — a plain ``injection``
    flag carries no sub-label, and since the GUI shows the family in its OWN column
    (``name``), the detail falls through to the verdict's first reason (more useful
    than repeating ``injection``). The raw ``flags``/``reasons`` are also returned on
    the firing for callers that want the exact CLI string.
    """
    # Keep the sub-label (the part after ':'); drop bare family flags like a plain
    # ``injection`` which carry no sub-classification.
    labels = [f.split(":", 1)[1] for f in flags if ":" in f and f.split(":", 1)[1]]
    if labels:
        return ", ".join(dict.fromkeys(labels))
    clean_reasons = [str(r) for r in reasons if r]
    if clean_reasons:
        return clean_reasons[0]
    return ""


# --------------------------------------------------------------------- endpoint


@router.get("", response_model=GuardrailsResponse)
async def guardrails(
    request: Request,
    scan: int = Query(_RUN_SCAN_DEFAULT, ge=1, le=_RUN_SCAN_MAX),
    limit: int = Query(_FIRINGS_DEFAULT, ge=1, le=_FIRINGS_MAX),
) -> GuardrailsResponse:
    """The built-in guardrail catalog + recent firings from the Studio run store.

    ``builtins`` is the SAME list ``himmy guardrails`` prints
    (:func:`himmy.cli.guardrails_view.builtin_rows`). ``firings`` scans the most
    recent ``scan`` runs in ``.himmy/studio.db`` for ``safety`` cognition steps and
    returns up to ``limit`` of them, newest first, each linked to its ``run_id``.
    """
    from himmy.api.studio_runs import get_run_store
    from himmy.cli.guardrails_view import builtin_rows

    builtins = [GuardrailInfo(**row) for row in builtin_rows()]

    store = get_run_store()
    summaries = store.list(limit=scan, offset=0)
    firings: list[GuardrailFiring] = []
    runs_scanned = 0
    for summary in summaries:
        runs_scanned += 1
        run = store.get(summary.id)
        if run is None:
            continue
        for step in run.steps:
            if step.kind != _SAFETY_KIND:
                continue
            flags = [str(f) for f in (step.flags or [])]
            reasons = [str(r) for r in (step.reasons or [])]
            firings.append(
                GuardrailFiring(
                    run_id=run.id,
                    agent_name=run.agent_name,
                    created_at=run.created_at,
                    seq=step.seq,
                    name=_firing_name(flags),
                    action=_firing_action(
                        blocked=bool(step.blocked), redacted=bool(step.redacted)
                    ),
                    stage=step.stage,
                    detail=_firing_detail(flags, reasons),
                    blocked=bool(step.blocked),
                    redacted=bool(step.redacted),
                    flags=flags,
                    reasons=reasons,
                )
            )

    # Newest run first, and within a run preserve step order (so a run that fired
    # input then output guardrails reads in causal order under its newest-first slot).
    firings.sort(key=lambda f: (f.created_at, f.seq), reverse=True)
    firings = firings[:limit]
    return GuardrailsResponse(
        builtins=builtins,
        firings=firings,
        firing_count=len(firings),
        runs_scanned=runs_scanned,
    )


__all__ = ["router"]
