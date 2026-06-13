"""Additional ToolService hardening tests (HTTP/auth/retry/timeout/redaction/events).

Complements ``tests/tools/test_tool_service.py`` by locking in production behavior
the base suite does not yet pin: the HEADER / PREENCODED_BASIC / NONE auth modes,
header-arg secret redaction in events, retry exhaustion + non-retryable codes not
being retried, the HTTP timeout path via the definition-level timeout, args-schema
INVALID_REQUEST for required path args, and event payloads for failure paths.

Offline-only: all HTTP is driven through ``httpx.MockTransport`` (no network); the
pydantic-ai toolset binding is exercised only behind a skipif when the extra is
absent.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from himmy.core.events import EventType
from himmy.services.storage.service import StorageService
from himmy.services.tools import (
    HttpAuthConfig,
    HttpAuthMode,
    HttpToolConfig,
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
    register_http_tool,
    register_local_tool,
)
from himmy.services.tools.security import REDACTED
from tests.conftest import run_async

try:  # optional provider extra — gates the pydantic-ai binding test
    import pydantic_ai  # noqa: F401

    _HAS_PYDANTIC_AI = True
except Exception:  # pragma: no cover - import guard
    _HAS_PYDANTIC_AI = False


def _http_service(transport: httpx.MockTransport) -> ToolService:
    client = httpx.AsyncClient(transport=transport, follow_redirects=False)
    return ToolService(ToolRegistry(), http_client=client)


# ----------------------------------------------------------------- auth modes
def test_http_header_auth_uses_custom_header(monkeypatch) -> None:
    """HEADER mode writes the secret to the configured header name verbatim."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.setenv("APIKEY", "k-123")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.headers.get("x-api-key")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(
                mode=HttpAuthMode.HEADER, env_var="APIKEY", header_name="X-API-Key"
            ),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["api_key"] == "k-123"


def test_http_preencoded_basic_passes_through(monkeypatch) -> None:
    """PREENCODED_BASIC sends the env value verbatim (already base64), unencoded here."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.setenv("CREDS", "QWxhZGRpbjpvcGVuIHNlc2FtZQ==")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(mode=HttpAuthMode.PREENCODED_BASIC, env_var="CREDS"),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["auth"] == "Basic QWxhZGRpbjpvcGVuIHNlc2FtZQ=="


def test_http_api_key_query_appends_secret_as_param(monkeypatch) -> None:
    """API_KEY_QUERY puts the secret in the query string, not a header."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.setenv("APIKEY", "k-xyz")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["api_key"] = request.url.params.get("api_key")
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(
                mode=HttpAuthMode.API_KEY_QUERY, env_var="APIKEY", query_param="api_key"
            ),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["api_key"] == "k-xyz"
    assert seen["auth"] is None  # not a header


def test_http_basic_auth_with_username(monkeypatch) -> None:
    """BASIC with a username encodes ``username:secret`` (the secret is the password)."""
    import base64

    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.setenv("PW", "s3cr3t")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(
                mode=HttpAuthMode.BASIC, env_var="PW", username="svc-account"
            ),
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    expected = base64.b64encode(b"svc-account:s3cr3t").decode()
    assert seen["auth"] == f"Basic {expected}"


def test_http_static_query_always_sent(monkeypatch) -> None:
    """static_query params are merged onto every request (e.g. an API version)."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["v"] = request.url.params.get("v")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE", path_template="/x", static_query={"v": "2024-01"}
        ),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["v"] == "2024-01"


def test_http_idempotency_key_header_on_mutation(monkeypatch) -> None:
    """A configured idempotency_arg rides as the Idempotency-Key header on a POST."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["key"] = request.headers.get("idempotency-key")
        return httpx.Response(201, json={"created": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            method="POST",
            path_template="/orders",
            body_arg_names=["sku"],
            idempotency_arg="request_id",
        ),
    )
    run_async(
        svc.execute(
            ToolInvocation(
                tool_name="t", args={"sku": "A1", "request_id": "req-7"}
            )
        )
    )
    run_async(svc.aclose())
    assert seen["key"] == "req-7"


def test_non_idempotent_post_not_retried_on_timeout(monkeypatch) -> None:
    """A POST with no idempotency key is NOT retried on TIMEOUT (may have landed)."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow", request=request)

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="post_once",
        http_config=HttpToolConfig(
            base_url_env_var="BASE", method="POST", path_template="/orders"
        ),
        retry_hints={"max_attempts": 4, "base_delay_seconds": 0.0},
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="post_once")))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT
    assert calls["n"] == 1  # not retried despite max_attempts=4


def test_idempotent_post_is_retried_on_timeout(monkeypatch) -> None:
    """A POST carrying an idempotency key MAY be retried (upstream dedupes the resend)."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(201, json={"created": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="post_idem",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            method="POST",
            path_template="/orders",
            idempotency_arg="request_id",
        ),
        retry_hints={"max_attempts": 3, "base_delay_seconds": 0.0},
    )
    result = run_async(
        svc.execute(
            ToolInvocation(tool_name="post_idem", args={"request_id": "r1"})
        )
    )
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert calls["n"] == 2  # retried once, then succeeded


def test_http_cursor_pagination_concatenates(monkeypatch) -> None:
    """CURSOR pagination follows next-cursor across pages and concatenates records."""
    from himmy.services.tools.models import HttpPaginationConfig, HttpPaginationMode

    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if not cursor:
            return httpx.Response(
                200, json={"items": [1, 2], "next": "c2"}
            )
        if cursor == "c2":
            return httpx.Response(200, json={"items": [3], "next": None})
        return httpx.Response(200, json={"items": [], "next": None})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="paged",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            pagination=HttpPaginationConfig(
                mode=HttpPaginationMode.CURSOR,
                items_path="items",
                cursor_path="next",
                cursor_param="cursor",
                max_pages=5,
            ),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="paged")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert result.result["items"] == [1, 2, 3]


def test_http_pagination_capped_at_max_pages(monkeypatch) -> None:
    """A never-ending cursor stops at max_pages (the infinite-loop / DoS guard)."""
    from himmy.services.tools.models import HttpPaginationConfig, HttpPaginationMode

    monkeypatch.setenv("BASE", "https://api.example.com")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json={"items": [calls["n"]], "next": "always"})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="paged",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            pagination=HttpPaginationConfig(
                mode=HttpPaginationMode.CURSOR,
                items_path="items",
                cursor_path="next",
                max_pages=3,
            ),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="paged")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert calls["n"] == 3  # capped


def test_non_idempotent_post_with_omitted_key_not_retried(monkeypatch) -> None:
    """A POST whose idempotency_arg is CONFIGURED but OMITTED at call time is NOT retried.

    Regression for the openapi-harden review: ``_is_retry_safe`` used to return True
    whenever ``idempotency_arg`` was set, but the ``Idempotency-Key`` header is only
    attached when the arg is actually supplied — so an omitted key means the upstream
    gets NO dedup header. Blind-retrying such a call on a ReadTimeout can double-act.
    """
    monkeypatch.setenv("BASE", "https://api.example.com")
    seen: dict = {"keys": [], "n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        seen["keys"].append(request.headers.get("idempotency-key"))
        raise httpx.ReadTimeout("slow", request=request)

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="post_no_key",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            method="POST",
            path_template="/orders",
            body_arg_names=["sku"],
            idempotency_arg="request_id",
        ),
        retry_hints={"max_attempts": 4, "base_delay_seconds": 0.0},
    )
    # Invoke WITHOUT request_id — the key is configured but not supplied.
    result = run_async(
        svc.execute(ToolInvocation(tool_name="post_no_key", args={"sku": "A1"}))
    )
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT
    assert seen["n"] == 1  # exactly one wire attempt despite max_attempts=4
    assert seen["keys"] == [None]  # no Idempotency-Key was ever sent


def test_link_header_cross_host_next_is_refused(monkeypatch) -> None:
    """A cross-host Link rel="next" is refused (default-deny) — bearer secret never leaks.

    Regression for the openapi-harden review: with the default empty egress allow-list,
    a hostile upstream returning ``Link: <https://evil.example/x>; rel="next"`` must NOT
    be followed, because the connector reuses its Authorization header on every page and
    would otherwise forward the bearer to evil.example.
    """
    from himmy.services.tools.models import HttpPaginationConfig, HttpPaginationMode

    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.setenv("TOKEN", "sek-123")
    hosts_seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts_seen.append(request.url.host)
        if request.url.host == "api.example.com":
            return httpx.Response(
                200,
                json={"items": [1, 2]},
                headers={"link": '<https://evil.example/x>; rel="next"'},
            )
        # Should never be reached — the cross-host hop must be refused first.
        return httpx.Response(200, json={"items": [99]})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="paged_link",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(mode=HttpAuthMode.BEARER, env_var="TOKEN"),
            pagination=HttpPaginationConfig(
                mode=HttpPaginationMode.LINK_HEADER, items_path="items", max_pages=5
            ),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="paged_link")))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST
    assert "cross-host" in (result.error_message or "")
    # The second (off-host) request was never issued — the secret stayed put.
    assert hosts_seen == ["api.example.com"]


def test_link_header_same_origin_next_is_followed(monkeypatch) -> None:
    """A same-origin Link rel="next" IS followed and pages concatenate as expected."""
    from himmy.services.tools.models import HttpPaginationConfig, HttpPaginationMode

    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("page") != "2":
            return httpx.Response(
                200,
                json={"items": [1, 2]},
                headers={
                    "link": '<https://api.example.com/x?page=2>; rel="next"'
                },
            )
        return httpx.Response(200, json={"items": [3]})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="paged_link_ok",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            pagination=HttpPaginationConfig(
                mode=HttpPaginationMode.LINK_HEADER, items_path="items", max_pages=5
            ),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="paged_link_ok")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert result.result["items"] == [1, 2, 3]
    assert result.result["page_count"] == 2  # both pages counted


def test_link_header_allow_listed_cross_host_next_is_followed(monkeypatch) -> None:
    """A cross-host Link is followed only when its host is on the egress allow-list."""
    from himmy.services.tools.models import HttpPaginationConfig, HttpPaginationMode

    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.example.com":
            return httpx.Response(
                200,
                json={"items": [1]},
                headers={"link": '<https://pages.example.com/x>; rel="next"'},
            )
        return httpx.Response(200, json={"items": [2]})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="paged_link_allow",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            egress_allow_hosts=["api.example.com", "pages.example.com"],
            pagination=HttpPaginationConfig(
                mode=HttpPaginationMode.LINK_HEADER, items_path="items", max_pages=5
            ),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="paged_link_allow")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert result.result["items"] == [1, 2]
    assert result.result["page_count"] == 2


def test_cursor_pagination_reports_pages_fetched(monkeypatch) -> None:
    """page_count reflects pages actually fetched in CURSOR mode (not a hardcoded 1).

    Regression for the openapi-harden review: page_count was only bumped in PAGE mode,
    so CURSOR/LINK_HEADER always reported 1 regardless of how many pages were fetched.
    """
    from himmy.services.tools.models import HttpPaginationConfig, HttpPaginationMode

    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        if not cursor:
            return httpx.Response(200, json={"items": [1, 2], "next": "c2"})
        if cursor == "c2":
            return httpx.Response(200, json={"items": [3], "next": None})
        return httpx.Response(200, json={"items": [], "next": None})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="paged_count",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            pagination=HttpPaginationConfig(
                mode=HttpPaginationMode.CURSOR,
                items_path="items",
                cursor_path="next",
                cursor_param="cursor",
                max_pages=5,
            ),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="paged_count")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert result.result["items"] == [1, 2, 3]
    assert result.result["page_count"] == 2  # two pages fetched, not 1


def test_http_auth_none_sends_no_authorization(monkeypatch) -> None:
    """NONE auth (the default) attaches no Authorization header."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/x"),
    )
    run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert seen["auth"] is None


def test_http_auth_env_unset_sends_no_header(monkeypatch) -> None:
    """A configured auth mode with an unset env var attaches no header (no crash)."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    monkeypatch.delenv("UNSET_TOKEN", raising=False)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            auth=HttpAuthConfig(mode=HttpAuthMode.BEARER, env_var="UNSET_TOKEN"),
        ),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t")))
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert seen["auth"] is None


# ----------------------------------------------------------------- SSRF (host pin)
def test_http_post_assembles_body_args(monkeypatch) -> None:
    """A POST routes declared body_arg_names into the JSON body, not the path."""
    monkeypatch.setenv("BASE", "https://api.example.com")
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["body"] = request.content.decode() if request.content else ""
        return httpx.Response(201, json={"created": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="create",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            method="POST",
            path_template="/orders",
            body_arg_names=["sku", "qty"],
        ),
    )
    result = run_async(
        svc.execute(ToolInvocation(tool_name="create", args={"sku": "A1", "qty": 2}))
    )
    run_async(svc.aclose())
    assert result.outcome == "success"
    assert seen["method"] == "POST"
    assert '"sku"' in seen["body"] and '"qty"' in seen["body"]


def test_http_backslash_in_path_arg_rejected(monkeypatch) -> None:
    """A backslash in a path arg is rejected as INVALID_REQUEST (traversal variant)."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/c/{cid}"),
    )
    result = run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"cid": "a\\..\\b"}))
    )
    run_async(svc.aclose())
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_crlf_in_path_arg_rejected(monkeypatch) -> None:
    """A CRLF-bearing path arg (header/log injection) is rejected as INVALID_REQUEST."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/c/{cid}"),
    )
    result = run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"cid": "x\r\nHost: evil"}))
    )
    run_async(svc.aclose())
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_http_missing_path_arg_is_invalid_request(monkeypatch) -> None:
    """A referenced path placeholder with no supplied arg fails as INVALID_REQUEST."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("upstream should never be reached")

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/c/{cid}"),
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="t", args={})))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


# ----------------------------------------------------------------- timeout
def test_http_definition_timeout_fires_on_slow_response(monkeypatch) -> None:
    """A handler slower than the definition timeout is cancelled and reported TIMEOUT."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(5.0)
        return httpx.Response(200, json={"ok": True})

    svc = _http_service(httpx.MockTransport(handler))
    register_http_tool(
        svc.registry,
        name="slow",
        http_config=HttpToolConfig(base_url_env_var="BASE", path_template="/x"),
        timeout_seconds=0.05,
    )
    result = run_async(svc.execute(ToolInvocation(tool_name="slow")))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT


# ----------------------------------------------------------------- retry semantics
def test_retry_exhausted_returns_last_failure() -> None:
    """A handler that always fails transiently exhausts attempts and returns failed."""
    registry = ToolRegistry()
    calls = {"n": 0}

    async def always_timeout(args: dict) -> dict:
        calls["n"] += 1
        raise TimeoutError()

    register_local_tool(
        registry,
        name="dead",
        handler=always_timeout,
        retry_hints={"max_attempts": 3, "base_delay_seconds": 0.0},
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="dead")))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.TIMEOUT
    assert calls["n"] == 3  # all attempts consumed


def test_non_retryable_error_is_not_retried() -> None:
    """A non-retryable failure (EXECUTION_ERROR) is returned after a single attempt."""
    registry = ToolRegistry()
    calls = {"n": 0}

    async def boom(args: dict) -> dict:
        calls["n"] += 1
        raise ValueError("hard failure")

    register_local_tool(
        registry,
        name="boom",
        handler=boom,
        retry_hints={"max_attempts": 5, "base_delay_seconds": 0.0},
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="boom")))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.EXECUTION_ERROR
    assert calls["n"] == 1  # not retried despite max_attempts=5


def test_retry_hints_max_retries_alias() -> None:
    """retry_hints={'max_retries': N} means N+1 total attempts."""
    registry = ToolRegistry()
    calls = {"n": 0}

    async def flaky(args: dict) -> dict:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise TimeoutError()
        return {"n": calls["n"]}

    register_local_tool(
        registry,
        name="flaky",
        handler=flaky,
        retry_hints={"max_retries": 2, "base_delay_seconds": 0.0},  # => 3 attempts
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="flaky")))
    assert result.outcome == "success"
    assert calls["n"] == 3


# ----------------------------------------------------------------- redaction in events
def test_http_header_secret_redacted_in_event(monkeypatch) -> None:
    """A header-arg whose name hints a secret is redacted on the TOOL_CALLED event."""
    monkeypatch.setenv("BASE", "https://api.example.com")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    sink = StorageService()
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    svc = ToolService(ToolRegistry(), event_sink=sink, http_client=client)
    register_http_tool(
        svc.registry,
        name="t",
        http_config=HttpToolConfig(
            base_url_env_var="BASE",
            path_template="/x",
            header_arg_names=["x_api_token"],
        ),
    )
    run_async(
        svc.execute(ToolInvocation(tool_name="t", args={"x_api_token": "super-secret"}))
    )
    run_async(svc.aclose())
    events = run_async(sink.list_events())
    called = next(e for e in events if e.event_type == EventType.TOOL_CALLED)
    # "token" is a sensitive hint substring -> redacted by default.
    assert called.payload["args"]["x_api_token"] == REDACTED
    assert "super-secret" not in str(called.payload)


def test_failed_call_emits_tool_failed_with_code(monkeypatch) -> None:
    """A failed dispatch emits TOOL_FAILED carrying the normalized error code."""
    sink = StorageService()
    svc = ToolService(ToolRegistry(), event_sink=sink)
    result = run_async(svc.execute(ToolInvocation(tool_name="nope")))
    assert result.error_code == ToolErrorCode.NOT_FOUND
    events = run_async(sink.list_events())
    failed = next(e for e in events if e.event_type == EventType.TOOL_FAILED)
    assert failed.payload["error_code"] == ToolErrorCode.NOT_FOUND.value
    assert failed.payload["tool_outcome"] == "failed"


def test_denied_approval_event_marks_outcome_denied() -> None:
    """An approval-blocked call emits TOOL_FAILED with outcome 'denied'."""
    registry = ToolRegistry()
    register_local_tool(
        registry, name="danger", handler=lambda a: {"ok": True}, requires_approval=True
    )
    sink = StorageService()
    svc = ToolService(registry, event_sink=sink)
    run_async(svc.execute(ToolInvocation(tool_name="danger")))
    events = run_async(sink.list_events())
    failed = next(e for e in events if e.event_type == EventType.TOOL_FAILED)
    assert failed.payload["error_code"] == ToolErrorCode.POLICY_BLOCKED.value
    assert failed.payload["tool_outcome"] == "denied"


# ----------------------------------------------------------------- input validation
def test_required_arg_missing_is_invalid_request_before_dispatch() -> None:
    """A required arg absent from invocation fails INVALID_REQUEST (handler not called)."""
    registry = ToolRegistry()
    called = {"hit": False}

    def handler(args: dict) -> dict:
        called["hit"] = True
        return {"ok": True}

    register_local_tool(
        registry,
        name="needsx",
        handler=handler,
        args_json_schema={
            "type": "object",
            "required": ["x"],
            "properties": {"x": {"type": "string"}},
        },
    )
    svc = ToolService(registry)
    result = run_async(svc.execute(ToolInvocation(tool_name="needsx", args={})))
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST
    assert called["hit"] is False


def test_additional_properties_false_strips_or_rejects_by_mode() -> None:
    """additionalProperties:false: lenient (default) strips the extra; strict rejects.

    Tier 1.2 makes arg-tolerance the default (so small models that hallucinate a
    spurious key still succeed); ``lenient_args=False`` restores strict rejection.
    """
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="strict",
        handler=lambda a: {"seen": a},
        args_json_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "additionalProperties": False,
        },
    )
    # Default (lenient): the hallucinated 'rogue' key is dropped before validation.
    lenient = run_async(
        ToolService(registry).execute(
            ToolInvocation(tool_name="strict", args={"a": 1, "rogue": 2})
        )
    )
    assert lenient.outcome == "success"
    assert lenient.result["seen"] == {"a": 1}
    # Strict opt-out: the unexpected key is rejected.
    strict = run_async(
        ToolService(registry, lenient_args=False).execute(
            ToolInvocation(tool_name="strict", args={"a": 1, "rogue": 2})
        )
    )
    assert strict.outcome == "failed"
    assert strict.error_code == ToolErrorCode.INVALID_REQUEST


# ----------------------------------------------------------------- post-hook + validation order
def test_post_hook_runs_before_output_validation_and_can_pass() -> None:
    """A post-hook reshaping into a schema-valid value passes output validation."""

    async def reshape(result, definition):
        # Original handler returns {'sum': 7}; reshape to a still-valid value.
        return {"sum": 100}

    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="adder",
        handler=lambda a: {"sum": a.get("a", 0) + a.get("b", 0)},
        output_json_schema={
            "type": "object",
            "properties": {"sum": {"type": "integer"}},
        },
    )
    svc = ToolService(registry, post_execution_hook=reshape)
    result = run_async(
        svc.execute(ToolInvocation(tool_name="adder", args={"a": 3, "b": 4}))
    )
    assert result.outcome == "success"
    assert result.result == {"sum": 100}


# ----------------------------------------------------------------- registry params
def test_register_local_tool_sets_timeout_and_retry() -> None:
    """register_local_tool surfaces timeout_seconds + retry_hints onto the definition."""
    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="cfg",
        handler=lambda a: {"ok": True},
        timeout_seconds=2.5,
        retry_hints={"max_attempts": 4},
        sensitive_arg_names=["pw"],
    )
    d = registry.get("cfg")
    assert d is not None
    assert d.timeout_seconds == 2.5
    assert d.retry_hints == {"max_attempts": 4}
    assert d.sensitive_arg_names == ["pw"]


def test_register_http_tool_definition_timeout_overrides_config() -> None:
    """Definition-level timeout_seconds takes precedence over http_config.timeout_seconds."""
    registry = ToolRegistry()
    register_http_tool(
        registry,
        name="h",
        http_config=HttpToolConfig(
            base_url_env_var="BASE", path_template="/x", timeout_seconds=9.0
        ),
        timeout_seconds=1.0,
    )
    svc = ToolService(registry)
    d = registry.get("h")
    assert d is not None
    # The service resolves definition > http_config > default.
    assert svc._resolve_timeout(d) == 1.0


# ----------------------------------------------------------------- pydantic-ai toolset (gated)
def test_toolset_binding_raises_clear_error_without_extra() -> None:
    """Without pydantic-ai, the toolset binding raises an actionable HimmyError."""
    if _HAS_PYDANTIC_AI:
        pytest.skip("pydantic-ai installed; the absent-extra path is not exercised")
    from himmy.core.errors import HimmyError
    from himmy.services.tools.runtime_adapter import ToolServiceToolset

    registry = ToolRegistry()
    register_local_tool(registry, name="x", handler=lambda a: {"ok": True})
    toolset = ToolServiceToolset(ToolService(registry))
    with pytest.raises(HimmyError):
        toolset.as_pydantic_ai_toolset()


@pytest.mark.skipif(not _HAS_PYDANTIC_AI, reason="requires the providers extra")
def test_toolset_binding_builds_tools_with_extra() -> (
    None
):  # pragma: no cover - extra only
    """With pydantic-ai installed, the toolset yields one Tool per definition."""
    from himmy.services.tools.runtime_adapter import ToolServiceToolset

    registry = ToolRegistry()
    register_local_tool(
        registry,
        name="adder",
        handler=lambda a: {"sum": a.get("a", 0) + a.get("b", 0)},
        args_json_schema={
            "type": "object",
            "required": ["a"],
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
    )
    toolset = ToolServiceToolset(ToolService(registry))
    tools = toolset.as_pydantic_ai_toolset()
    assert len(tools) == 1
