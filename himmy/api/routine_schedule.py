"""Phase-1 scheduling math: cron + per-routine timezone + missed-run/catch-up.

This module is the *pure*, side-effect-free heart of the Phase-1 routines upgrade. It
holds the timezone-aware next-fire computation (for the new ``cron`` and one-shot ``at``
schedule kinds) and the missed-run / catch-up policy math, isolated from the
store/scheduler in :mod:`himmy.api.routines` so it can be golden-tested with frozen clocks
(including DST and the non-integer +5:45 Asia/Kathmandu offset) and so the optional
``cronsim`` dependency is imported lazily — a deployment that never uses a ``cron`` routine
never needs the extra installed.

DESIGN INVARIANTS (red-team gates from the GATE):

* **Next-fire is computed IN the routine timezone, then persisted as UTC.** ``cronsim``
  iterates in the wall-clock zone, so '30 6 * * *' Asia/Kathmandu yields 06:30 NPT =
  00:45 UTC every day, and a DST fall-back '30 1 * * *' fires the duplicated 01:30 EXACTLY
  ONCE (cronsim's verified behaviour; croniter double-fires and was rejected by the GATE).
* **The timezone default is generic-OSS-safe**: ``timezone`` unset → env ``HIMMY_TZ`` →
  host-local. Asia/Kathmandu is the recommended Nepal-fleet value (documented), not a
  hardcoded default.
* **Missed-run / catch-up reads the AUTHORITATIVE anchor (last_run_at), never a wall-clock
  delta** — clock-skew safe. ``coalesce`` (default) fires once now if anything was missed;
  ``skip`` drops the gap; ``run_each`` backfills each missed occurrence, HARD-CAPPED by
  ``max_catchup`` and dropping occurrences older than ``grace_seconds`` (a catch-up storm
  after a long laptop sleep can never enqueue an unbounded backlog).
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

#: Hard ceiling on how many occurrences ``cron_occurrences_between`` will enumerate before
#: giving up — a guardrail against a pathological expression + a multi-year gap pinning CPU.
#: It is generously above ``max_catchup`` (default 100) so the cap, not this, is what an
#: operator tunes; this only stops a runaway enumeration.
_OCCURRENCE_ENUMERATION_CEILING = 10_000


class CronUnavailableError(RuntimeError):
    """A ``cron`` routine was used but the optional ``cronsim`` dependency is not installed.

    Raised at the point of computing a cron next-fire (validation, due-math, or catch-up)
    rather than at import, so the rest of the routines layer — and every daily/every
    routine — works without the extra. The message points at the ``cron`` extra.
    """

    def __init__(self) -> None:
        super().__init__(
            "cron schedules require the optional 'cronsim' dependency; install it with "
            "`pip install 'himmy[cron]'` (or `pip install cronsim`)"
        )


def _import_cronsim() -> tuple[type, type[Exception]]:
    """Return ``(CronSim, CronSimError)`` or raise :class:`CronUnavailableError`."""
    try:
        from cronsim import CronSim, CronSimError
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise CronUnavailableError() from exc
    return CronSim, CronSimError


def cron_is_available() -> bool:
    """True when the optional ``cronsim`` dependency can be imported."""
    try:
        _import_cronsim()
    except CronUnavailableError:
        return False
    return True


def default_timezone_name() -> str:
    """The fallback IANA tz name when a routine sets none: ``HIMMY_TZ`` else host-local.

    Generic-OSS-safe: there is no hardcoded region. A Nepal fleet sets ``HIMMY_TZ=
    Asia/Kathmandu`` (the documented recommendation). When ``HIMMY_TZ`` is unset or invalid
    we fall back to the host's local zone name (or ``UTC`` if even that cannot be resolved),
    so the scheduler always has a concrete, valid zone to compute in.
    """
    env = (os.environ.get("HIMMY_TZ") or "").strip()
    if env and _valid_tz(env):
        return env
    local = datetime.now().astimezone().tzinfo
    name = getattr(local, "key", None)  # ZoneInfo carries .key; a fixed offset does not
    if isinstance(name, str) and _valid_tz(name):
        return name
    return "UTC"


def _valid_tz(name: str) -> bool:
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return False
    return True


def resolve_zone(timezone: str | None) -> ZoneInfo:
    """Resolve a routine's effective :class:`ZoneInfo` (explicit → default → UTC)."""
    name = timezone or default_timezone_name()
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        # default_timezone_name already validates its result, so this only triggers when an
        # explicit-but-now-unresolvable name was stored; degrade to UTC rather than crash a
        # whole tick (one bad routine must not poison the loop).
        return ZoneInfo("UTC")


def validate_timezone(name: str) -> None:
    """Raise ``ValueError`` if ``name`` is not a resolvable IANA timezone (model validator)."""
    if not _valid_tz(name):
        raise ValueError(
            f"timezone {name!r} is not a valid IANA timezone "
            "(e.g. 'Asia/Kathmandu', 'UTC', 'America/New_York')"
        )


def validate_cron(expr: str) -> None:
    """Raise ``ValueError`` if ``expr`` is not a parseable cron expression.

    Uses ``cronsim``'s own parser (so the validation matches the runtime exactly). When
    ``cronsim`` is absent we cannot validate the syntax, so we accept the string at
    create-time and surface :class:`CronUnavailableError` only when the routine actually
    fires — keeping the dependency truly optional (you can author a cron routine on a box
    without the extra and run it on one that has it).
    """
    try:
        CronSim, CronSimError = _import_cronsim()
    except CronUnavailableError:
        return
    try:
        # Anchor is irrelevant for a pure parse check; any aware datetime works.
        CronSim(expr, datetime(2000, 1, 1, tzinfo=UTC))
    except CronSimError as exc:
        raise ValueError(f"invalid cron expression {expr!r}: {exc}") from exc


def _as_utc(dt: datetime) -> datetime:
    """Normalise an aware datetime to UTC (naive is assumed UTC)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def cron_next_fire(expr: str, after_utc: datetime, zone: ZoneInfo) -> datetime:
    """The first cron occurrence STRICTLY after ``after_utc``, computed in ``zone``, as UTC.

    The instant is computed in the routine's wall-clock zone (so DST + non-integer offsets
    are honoured — verified: +5:45 Asia/Kathmandu, NY spring/fall) and returned normalised
    to UTC for persistence. ``cronsim`` yields the next occurrence at-or-after its anchor; we
    nudge the anchor forward by one microsecond so the result is STRICTLY after the boundary
    (a fire AT ``after_utc`` must not re-yield ``after_utc`` and loop).
    """
    CronSim, _ = _import_cronsim()
    anchor_local = (_as_utc(after_utc) + timedelta(microseconds=1)).astimezone(zone)
    occ = next(CronSim(expr, anchor_local))
    return _as_utc(occ)


def cron_occurrences_between(
    expr: str,
    after_utc: datetime,
    until_utc: datetime,
    zone: ZoneInfo,
    *,
    limit: int,
) -> list[datetime]:
    """Cron occurrences in ``(after_utc, until_utc]``, computed in ``zone``, as UTC, capped.

    Used by the ``run_each`` catch-up backfill. Enumerates STRICTLY after ``after_utc`` and
    up to and INCLUDING ``until_utc`` (the current planned boundary), stopping at ``limit``
    results or :data:`_OCCURRENCE_ENUMERATION_CEILING` iterations (whichever first) so a
    pathological gap can never pin the CPU. The cap surfacing (how many were dropped) is the
    caller's job — this only enumerates.
    """
    CronSim, _ = _import_cronsim()
    after = _as_utc(after_utc)
    until = _as_utc(until_utc)
    anchor_local = (after + timedelta(microseconds=1)).astimezone(zone)
    it = CronSim(expr, anchor_local)
    out: list[datetime] = []
    for _ in range(_OCCURRENCE_ENUMERATION_CEILING):
        occ_utc = _as_utc(next(it))
        if occ_utc > until:
            break
        out.append(occ_utc)
        if len(out) >= limit:
            break
    return out


__all__ = [
    "CronUnavailableError",
    "cron_is_available",
    "cron_next_fire",
    "cron_occurrences_between",
    "default_timezone_name",
    "resolve_zone",
    "validate_cron",
    "validate_timezone",
]
