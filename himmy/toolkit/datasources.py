"""Data-sources pack: keyless public connectors — ``weather``, ``geocode``, ``wikipedia``.

Three general-purpose, no-API-key data sources an agent commonly needs:

* ``weather`` — current conditions for a lat/lon via Open-Meteo;
* ``geocode`` — turn a place name into coordinates via OpenStreetMap Nominatim;
* ``wikipedia`` — a plain-text summary of the best-matching article.

Each is a thin tool over a connector client that fetches and normalizes the upstream
JSON. Network goes through the injectable :class:`~himmy.connectors.fetcher.Fetcher`
seam (GET + JSON), so the suite runs offline against a fake fetcher. The endpoints are
fixed, trusted hosts, so no per-call URL guard is needed (only the query is templated).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote, quote_plus

from himmy.connectors.fetcher import Fetcher, HttpxFetcher, get_json
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig

_WEATHER_SCHEMA = {
    "type": "object",
    "properties": {
        "latitude": {"type": "number"},
        "longitude": {"type": "number"},
    },
    "required": ["latitude", "longitude"],
    "additionalProperties": False,
}

_GEOCODE_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "A place name to locate."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
    },
    "required": ["query"],
    "additionalProperties": False,
}

_WIKI_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string", "description": "A topic to look up."}},
    "required": ["query"],
    "additionalProperties": False,
}


def register_datasources_pack(
    registry: ToolRegistry,
    config: ToolkitConfig,
    *,
    fetcher: Fetcher | None = None,
) -> None:
    """Register ``weather``, ``geocode``, and ``wikipedia`` over public APIs."""
    fetch = fetcher or HttpxFetcher(timeout=config.http_timeout)

    def weather(args: dict[str, Any]) -> dict[str, Any]:
        lat = float(args["latitude"])
        lon = float(args["longitude"])
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
        )
        data = get_json(fetch, url)
        return {
            "latitude": lat,
            "longitude": lon,
            "current": data.get("current", {}),
            "units": data.get("current_units", {}),
        }

    def geocode(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        limit = max(1, min(int(args.get("limit", 5)), 10))
        url = (
            "https://nominatim.openstreetmap.org/search"
            f"?q={quote_plus(query)}&format=json&limit={limit}"
        )
        data = get_json(fetch, url)
        return {
            "query": query,
            "results": [
                {
                    "name": item.get("display_name"),
                    "latitude": float(item["lat"]),
                    "longitude": float(item["lon"]),
                }
                for item in data
            ],
        }

    def wikipedia(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        url = (
            "https://en.wikipedia.org/w/api.php?action=query&format=json"
            "&prop=extracts&exintro&explaintext&redirects=1&generator=search"
            f"&gsrsearch={quote(query)}&gsrlimit=1"
        )
        data = get_json(fetch, url)
        pages = (data.get("query", {}) or {}).get("pages", {}) or {}
        for page in pages.values():
            title = page.get("title")
            return {
                "query": query,
                "title": title,
                "extract": page.get("extract", ""),
                "url": f"https://en.wikipedia.org/wiki/{quote(str(title))}"
                if title
                else None,
            }
        return {"query": query, "title": None, "extract": "", "url": None}

    register_local_tool(
        registry,
        name="weather",
        handler=weather,
        description="Current weather for a latitude/longitude (Open-Meteo, keyless).",
        args_json_schema=_WEATHER_SCHEMA,
        metadata={"pack": "data-sources"},
    )
    register_local_tool(
        registry,
        name="geocode",
        handler=geocode,
        description="Geocode a place name to coordinates (OpenStreetMap Nominatim).",
        args_json_schema=_GEOCODE_SCHEMA,
        metadata={"pack": "data-sources"},
    )
    register_local_tool(
        registry,
        name="wikipedia",
        handler=wikipedia,
        description="Plain-text summary of the best-matching Wikipedia article.",
        args_json_schema=_WIKI_SCHEMA,
        metadata={"pack": "data-sources"},
    )


__all__ = ["register_datasources_pack"]
