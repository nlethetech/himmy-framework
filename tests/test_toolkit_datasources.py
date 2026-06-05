"""Tests for the data-sources pack: weather/geocode/wikipedia (fake fetcher)."""

from __future__ import annotations

import json

from himmy.services.tools.registry import ToolRegistry
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.datasources import register_datasources_pack


class FakeFetcher:
    """Returns canned JSON keyed by a substring of the requested URL."""

    def __init__(self, routes: dict[str, object]) -> None:
        self._routes = routes

    def get_text(self, url: str) -> str:
        for needle, payload in self._routes.items():
            if needle in url:
                return json.dumps(payload)
        raise AssertionError(f"unexpected url: {url}")

    def get_bytes(self, url: str) -> bytes:
        return self.get_text(url).encode("utf-8")


def _registry(routes: dict[str, object]) -> ToolRegistry:
    registry = ToolRegistry()
    register_datasources_pack(
        registry, ToolkitConfig(), fetcher=FakeFetcher(routes)
    )
    return registry


def test_weather_returns_current() -> None:
    routes = {
        "api.open-meteo.com": {
            "current": {"temperature_2m": 21.5, "weather_code": 1},
            "current_units": {"temperature_2m": "°C"},
        }
    }
    out = _registry(routes).handler_for("weather")(
        {"latitude": 27.7, "longitude": 83.6}
    )
    assert out["current"]["temperature_2m"] == 21.5
    assert out["units"]["temperature_2m"] == "°C"


def test_geocode_normalizes_results() -> None:
    routes = {
        "nominatim": [
            {"display_name": "Bardaghat, Nepal", "lat": "27.69", "lon": "83.63"}
        ]
    }
    out = _registry(routes).handler_for("geocode")({"query": "Bardaghat"})
    assert out["results"][0]["name"] == "Bardaghat, Nepal"
    assert out["results"][0]["latitude"] == 27.69
    assert out["results"][0]["longitude"] == 83.63


def test_wikipedia_extracts_summary() -> None:
    routes = {
        "wikipedia.org": {
            "query": {
                "pages": {
                    "42": {"title": "Permaculture", "extract": "Permaculture is a design system."}
                }
            }
        }
    }
    out = _registry(routes).handler_for("wikipedia")({"query": "permaculture"})
    assert out["title"] == "Permaculture"
    assert "design system" in out["extract"]
    assert out["url"] == "https://en.wikipedia.org/wiki/Permaculture"


def test_wikipedia_no_match() -> None:
    out = _registry({"wikipedia.org": {"query": {"pages": {}}}}).handler_for(
        "wikipedia"
    )({"query": "zzzzz"})
    assert out["title"] is None
    assert out["extract"] == ""
