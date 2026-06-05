"""WS3.3 — egress allow-listing in guard_url."""

from __future__ import annotations

import pytest

from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit._net import _host_allowed, guard_url


def test_host_allowed_exact_and_subdomain() -> None:
    allow = {"example.com", "*.trusted.org"}
    assert _host_allowed("example.com", allow)
    assert _host_allowed("api.example.com", allow)  # subdomain
    assert _host_allowed("a.b.trusted.org", allow)
    assert not _host_allowed("evil.com", allow)
    assert not _host_allowed("notexample.com", allow)  # not a subdomain


def test_allow_list_permits_listed_host() -> None:
    # allow_private skips DNS resolution so the test needs no network.
    url = "https://api.example.com/v1"
    assert guard_url(url, allow_private=True, allow_hosts={"example.com"}) == url


def test_allow_list_blocks_unlisted_host() -> None:
    with pytest.raises(ToolSecurityError):
        guard_url("https://evil.com/x", allow_private=True, allow_hosts={"example.com"})


def test_no_allow_list_preserves_default_behavior() -> None:
    url = "https://example.com/x"
    assert guard_url(url, allow_private=True, allow_hosts=None) == url


def test_allow_list_still_rejects_bad_scheme() -> None:
    with pytest.raises(ToolSecurityError):
        guard_url("ftp://example.com", allow_private=True, allow_hosts={"example.com"})
