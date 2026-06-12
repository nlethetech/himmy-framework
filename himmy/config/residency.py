"""Data residency / region pinning (WS4.3).

When ``HIMMY_REGION`` is set, the deployment is pinned: an operation targeting a region
that isn't allowed (``HIMMY_ALLOWED_REGIONS``, defaulting to just the home region) is
refused via :func:`enforce_region`. Unset ⇒ no pinning (unchanged behavior). Wire
:func:`enforce_region` at the points where data leaves a region — the storage/inference
builders and any cross-region connector — to keep regulated data in-jurisdiction.
"""

from __future__ import annotations

import os

from himmy.core.errors import HimmyError


class ResidencyError(HimmyError):
    """An operation targeted a region outside the configured residency policy."""


def current_region() -> str | None:
    """The home region (``HIMMY_REGION``), or ``None`` when not pinned."""
    region = os.environ.get("HIMMY_REGION")
    return region.strip() or None if region else None


def allowed_regions() -> set[str]:
    """The set of regions this deployment may use (home region + explicit allows)."""
    explicit = {
        r.strip()
        for r in os.environ.get("HIMMY_ALLOWED_REGIONS", "").split(",")
        if r.strip()
    }
    home = current_region()
    if home:
        explicit.add(home)
    return explicit


def region_allowed(region: str | None) -> bool:
    """Whether ``region`` is permitted (always True when residency isn't pinned)."""
    if current_region() is None:
        return True
    if region is None:
        return False
    return region in allowed_regions()


def enforce_region(region: str | None, *, context: str | None = None) -> None:
    """Raise :class:`ResidencyError` when ``region`` violates the residency policy.

    ``context`` names the boundary being checked (e.g. ``"storage"``/``"inference"``)
    so the error points the operator at the offending knob; it is otherwise inert when
    residency is not pinned.
    """
    if not region_allowed(region):
        where = f"{context}: " if context else ""
        raise ResidencyError(
            f"data residency: {where}region {region!r} is not allowed "
            f"(home={current_region()!r}, allowed={sorted(allowed_regions())})"
        )


def storage_region() -> str | None:
    """The region the durable store is pinned to (``HIMMY_STORE_REGION``), or None.

    Defaults to the home region (``HIMMY_REGION``) when unset, so a single
    ``HIMMY_REGION=eu`` keeps storage in-region without a second variable; set
    ``HIMMY_STORE_REGION`` explicitly when the database lives in a different region than
    the deployment home (it must still be inside ``HIMMY_ALLOWED_REGIONS``).
    """
    explicit = os.environ.get("HIMMY_STORE_REGION")
    if explicit and explicit.strip():
        return explicit.strip()
    return current_region()


__all__ = [
    "ResidencyError",
    "current_region",
    "allowed_regions",
    "region_allowed",
    "enforce_region",
    "storage_region",
]
