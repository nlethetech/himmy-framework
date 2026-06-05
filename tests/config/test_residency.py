"""WS4.3 — data residency / region pinning."""

from __future__ import annotations

from typing import Any

import pytest

from himmy.config.residency import (
    ResidencyError,
    allowed_regions,
    enforce_region,
    region_allowed,
)


def test_not_pinned_allows_everything(monkeypatch: Any) -> None:
    monkeypatch.delenv("HIMMY_REGION", raising=False)
    assert region_allowed("us") and region_allowed("eu") and region_allowed(None)
    enforce_region("anywhere")  # no raise


def test_pinned_region_allows_home_only(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.delenv("HIMMY_ALLOWED_REGIONS", raising=False)
    assert region_allowed("eu")
    assert not region_allowed("us")
    assert not region_allowed(None)
    with pytest.raises(ResidencyError):
        enforce_region("us")
    enforce_region("eu")  # home is fine


def test_explicit_allowed_regions(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.setenv("HIMMY_ALLOWED_REGIONS", "eu, eu-west, ch")
    assert allowed_regions() == {"eu", "eu-west", "ch"}
    assert region_allowed("ch")
    assert not region_allowed("us")
