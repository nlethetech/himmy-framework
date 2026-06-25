"""SSRF-via-redirect regression for the data-sources pack (Finding #11).

``register_datasources_pack`` used to build its default fetcher with no SSRF guard
and no pinned transport, while still following up to 10 redirects. A poisoned /
MITM'd upstream could therefore 302 the agent to ``169.254.169.254`` (cloud IMDS) or
loopback and the hop would be followed, leaking internal data.

These tests drive the registered tools through the *default* (``fetcher=None``)
production path over an injected mock transport, so the guard wiring inside
``register_datasources_pack`` is what is actually exercised. Fully offline.
"""

from __future__ import annotations

import pytest

from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit.config import ToolkitConfig
from himmy.toolkit.datasources import register_datasources_pack

# A literal public IP host — keeps the redirect source off real DNS.
_PUBLIC_IP = "93.184.216.34"


def _patched_client(handler):
    """An ``httpx.Client`` factory that swaps in a mock transport for the handler."""
    import httpx

    real_client = httpx.Client

    def _client(**kwargs: object) -> httpx.Client:
        # The fetcher passes timeout/follow_redirects/transport; the mock transport
        # replaces real networking entirely.
        kwargs.pop("timeout", None)
        kwargs.pop("follow_redirects", None)
        kwargs.pop("transport", None)
        kwargs.pop("headers", None)
        return real_client(transport=httpx.MockTransport(handler))

    return _client


def _geocode_handler() -> ToolRegistry:
    """Register the data-sources pack with its DEFAULT fetcher (the guarded path)."""
    registry = ToolRegistry()
    register_datasources_pack(registry, ToolkitConfig())
    return registry


@pytest.mark.parametrize(
    "evil_target",
    [
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata (IMDS)
        "http://127.0.0.1/",  # loopback
    ],
)
def test_datasources_refuses_redirect_to_internal_ip(evil_target: str) -> None:
    """A 3xx from the upstream toward an internal address is blocked by the guard.

    On the pre-fix code (no ``guard`` / no pinned transport) the redirect would be
    followed and ``SECRET INTERNAL DATA`` returned; with the fix the per-hop guard
    raises ``ToolSecurityError`` before the hop is taken.
    """
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "nominatim.openstreetmap.org":
            return httpx.Response(302, headers={"Location": evil_target})
        # Reaching the internal target would mean the SSRF guard failed.
        return httpx.Response(200, text='[{"display_name":"x","lat":"0","lon":"0"}]')

    geocode = _geocode_handler().handler_for("geocode")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx, "Client", _patched_client(handler))
        with pytest.raises(ToolSecurityError):
            geocode({"query": "anywhere"})


def test_datasources_redirect_to_public_host_still_followed() -> None:
    """A redirect to another public host is still honored (no false positive)."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "nominatim.openstreetmap.org":
            return httpx.Response(302, headers={"Location": f"http://{_PUBLIC_IP}/p"})
        return httpx.Response(
            200, text='[{"display_name":"Public Place","lat":"1.5","lon":"2.5"}]'
        )

    geocode = _geocode_handler().handler_for("geocode")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(httpx, "Client", _patched_client(handler))
        out = geocode({"query": "anywhere"})

    assert out["results"][0]["latitude"] == 1.5
    assert out["results"][0]["longitude"] == 2.5
