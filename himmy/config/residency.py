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


def enforce_region(region: str | None) -> None:
    """Raise :class:`ResidencyError` when ``region`` violates the residency policy."""
    if not region_allowed(region):
        raise ResidencyError(
            f"data residency: region {region!r} is not allowed "
            f"(home={current_region()!r}, allowed={sorted(allowed_regions())})"
        )


__all__ = [
    "ResidencyError",
    "current_region",
    "allowed_regions",
    "region_allowed",
    "enforce_region",
]
