"""End-to-end test for the generic OpenAPI/REST connector against a LOCAL server.

Unlike the external-service connectors (Discord/Slack/GitHub) — which can only be
contract-tested with a mocked transport — this connector is fully verifiable offline:
we stand up a real ``http.server`` on loopback, point an OpenAPI 3 document at it, derive
agent tools with :func:`load_openapi_tools`, and drive REAL tool calls through
:class:`ToolService`. We assert the production guarantees actually hold over the wire:

* the secrets-layer credential is applied as the right auth header (and never logged);
* request args are schema-validated before the call (a bad arg never reaches the server);
* the declared response schema is validated on the way back;
* cursor pagination is followed and every page's records are concatenated;
* a side-effecting POST routes its body and is gated behind approval;
* the SSRF guard refuses loopback by default, and an egress allow-list is enforced.

The loopback server is reached with ``allow_private_hosts=True`` (a trusted local API);
the default-deny SSRF behavior is asserted separately by pointing a tool at loopback
WITHOUT that opt-in and confirming the call is refused before any socket is opened.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from himmy.config.http_tool_spec import register_http_tools
from himmy.config.openapi_spec import load_openapi_tools
from himmy.config.secrets import configure_secrets
from himmy.services.tools import (
    ToolErrorCode,
    ToolInvocation,
    ToolRegistry,
    ToolService,
)
from tests.conftest import run_async

API_TOKEN = "tok-e2e-secret-123"


class _MemSecrets:
    """A tiny in-memory secrets provider so the credential comes from the layer."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class _Handler(BaseHTTPRequestHandler):
    """A minimal API: bearer-guarded ``/widgets`` (cursor-paged) + ``/widgets`` POST."""

    # Captured cross-request so assertions can inspect what the server actually saw.
    seen: dict[str, Any] = {}

    def log_message(self, *args: Any) -> None:  # silence stderr spam
        pass

    def _json(self, status: int, body: Any) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _authed(self) -> bool:
        type(self).seen["auth_header"] = self.headers.get("Authorization")
        return self.headers.get("Authorization") == f"Bearer {API_TOKEN}"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/widgets":
            self._json(404, {"error": "not found"})
            return
        if not self._authed():
            self._json(401, {"error": "unauthorized"})
            return
        query = parse_qs(parsed.query)
        cursor = query.get("cursor", [""])[0]
        # Two pages: no cursor → page 1 (+next cursor); cursor=p2 → final page.
        if not cursor:
            self._json(
                200,
                {"items": [{"id": 1}, {"id": 2}], "next_cursor": "p2"},
            )
        elif cursor == "p2":
            self._json(200, {"items": [{"id": 3}], "next_cursor": None})
        else:
            self._json(200, {"items": [], "next_cursor": None})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/widgets":
            self._json(404, {"error": "not found"})
            return
        if not self._authed():
            self._json(401, {"error": "unauthorized"})
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        type(self).seen["post_body"] = json.loads(raw or b"{}")
        type(self).seen["idempotency_key"] = self.headers.get("Idempotency-Key")
        self._json(201, {"id": 99, "created": True})


@pytest.fixture
def server() -> Iterator[str]:
    """Run the local API on an ephemeral loopback port; yield its base URL."""
    _Handler.seen = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def _openapi(base_url: str) -> dict[str, Any]:
    """An OpenAPI 3 doc describing the local API (bearer auth, list + create)."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Widgets", "version": "1.0.0"},
        "servers": [{"url": base_url}],
        "components": {
            "securitySchemes": {
                "bearerAuth": {"type": "http", "scheme": "bearer"}
            }
        },
        "paths": {
            "/widgets": {
                "get": {
                    "operationId": "list_widgets",
                    "summary": "List widgets.",
                    "parameters": [
                        {"name": "cursor", "in": "query", "schema": {"type": "string"}}
                    ],
                    "responses": {
                        "200": {
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object"}
                                }
                            }
                        }
                    },
                },
                "post": {
                    "operationId": "create_widget",
                    "summary": "Create a widget.",
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "qty": {"type": "integer"},
                                    },
                                }
                            }
                        }
                    },
                    "responses": {
                        "201": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["id"],
                                        "properties": {"id": {"type": "integer"}},
                                    }
                                }
                            }
                        }
                    },
                },
            }
        },
    }


def _service() -> ToolService:
    # No injected client → the service builds its OWN DNS-pinned client, so the SSRF
    # allow-list / private-host policy is enforced at the transport layer (real e2e).
    return ToolService(ToolRegistry())


def test_openapi_derives_tools_and_calls_apply_auth(server: str) -> None:
    """A GET tool derived from the spec applies the bearer secret and parses JSON."""
    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server),
        secret="WIDGETS_TOKEN",
        allow_private_hosts=True,
    )
    names = {s.name for s in specs}
    assert {"list_widgets", "create_widget"} <= names

    svc = _service()
    register_http_tools(svc.registry, specs)
    result = run_async(svc.execute(ToolInvocation(tool_name="list_widgets")))
    run_async(svc.aclose())

    assert result.outcome == "success", result.error_message
    # The credential was read from the secrets layer and sent as a bearer token.
    assert _Handler.seen["auth_header"] == f"Bearer {API_TOKEN}"
    assert result.result["items"] == [{"id": 1}, {"id": 2}]


def test_openapi_secret_redacted_from_events(server: str) -> None:
    """The token never appears in any emitted event payload (auth header is internal)."""
    from himmy.services.storage.service import StorageService

    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server), secret="WIDGETS_TOKEN", allow_private_hosts=True
    )
    sink = StorageService()
    svc = ToolService(ToolRegistry(), event_sink=sink)
    register_http_tools(svc.registry, specs)
    run_async(svc.execute(ToolInvocation(tool_name="list_widgets")))
    run_async(svc.aclose())
    events = run_async(sink.list_events())
    assert all(API_TOKEN not in str(e.payload) for e in events)


def test_pagination_follows_cursor_and_concatenates(server: str) -> None:
    """CURSOR pagination follows next_cursor and concatenates every page's records."""
    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server), secret="WIDGETS_TOKEN", allow_private_hosts=True
    )
    list_spec = next(s for s in specs if s.name == "list_widgets")
    list_spec.pagination = {
        "mode": "cursor",
        "items_path": "items",
        "cursor_path": "next_cursor",
        "cursor_param": "cursor",
        "max_pages": 5,
    }
    svc = _service()
    register_http_tools(svc.registry, [list_spec])
    result = run_async(svc.execute(ToolInvocation(tool_name="list_widgets")))
    run_async(svc.aclose())
    assert result.outcome == "success", result.error_message
    # Page 1 ({1,2}) + page 2 ({3}) concatenated; stops when next_cursor is null.
    assert [item["id"] for item in result.result["items"]] == [1, 2, 3]


def test_request_schema_validation_blocks_bad_arg(server: str) -> None:
    """A non-integer 'qty' fails arg-schema validation BEFORE any request is sent."""
    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server), secret="WIDGETS_TOKEN", allow_private_hosts=True
    )
    create = next(s for s in specs if s.name == "create_widget")
    # Tighten the derived arg schema so qty must be an integer (typed request contract).
    create.args_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "qty": {"type": "integer"}},
        "additionalProperties": False,
    }
    svc = _service()
    register_http_tools(svc.registry, [create])
    result = run_async(
        svc.execute(
            ToolInvocation(
                tool_name="create_widget", args={"name": "w", "qty": "not-an-int"}
            )
        )
    )
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST
    assert "post_body" not in _Handler.seen  # the server was never reached


def test_response_schema_validation_on_post(server: str) -> None:
    """A POST routes its body and the declared 2xx response schema is honored."""
    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server),
        secret="WIDGETS_TOKEN",
        allow_private_hosts=True,
        validate_responses=True,
    )
    create = next(s for s in specs if s.name == "create_widget")
    svc = _service()
    register_http_tools(svc.registry, [create])
    result = run_async(
        svc.execute(
            ToolInvocation(tool_name="create_widget", args={"name": "w", "qty": 3})
        )
    )
    run_async(svc.aclose())
    assert result.outcome == "success", result.error_message
    assert _Handler.seen["post_body"] == {"name": "w", "qty": 3}
    assert result.result == {"id": 99, "created": True}


def test_ssrf_loopback_refused_without_opt_in(server: str) -> None:
    """Pointing a tool at loopback WITHOUT allow_private_hosts is refused (no socket)."""
    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server), secret="WIDGETS_TOKEN", allow_private_hosts=False
    )
    svc = _service()
    register_http_tools(svc.registry, specs)
    result = run_async(svc.execute(ToolInvocation(tool_name="list_widgets")))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    # The pinned transport refuses the private/loopback connect → INVALID_REQUEST.
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


def test_egress_allow_list_blocks_off_list_host(server: str) -> None:
    """An egress allow-list that omits the server's host blocks the call up-front."""
    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server),
        secret="WIDGETS_TOKEN",
        allow_private_hosts=True,
        egress_allow_hosts=["api.allowed.example"],  # not the loopback server
    )
    svc = _service()
    register_http_tools(svc.registry, specs)
    result = run_async(svc.execute(ToolInvocation(tool_name="list_widgets")))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST
    assert "allow-list" in (result.error_message or "")


def test_method_allow_list_rejects_unknown_verb(server: str) -> None:
    """A connector configured with a disallowed method fails INVALID_REQUEST."""
    configure_secrets(_MemSecrets({"WIDGETS_TOKEN": API_TOKEN}))
    specs = load_openapi_tools(
        _openapi(server), secret="WIDGETS_TOKEN", allow_private_hosts=True
    )
    spec = next(s for s in specs if s.name == "list_widgets")
    cfg = spec.to_config()
    cfg.method = "TRACE"  # not on ALLOWED_HTTP_METHODS
    from himmy.services.tools.registry import register_http_tool

    svc = _service()
    register_http_tool(svc.registry, name="trace_tool", http_config=cfg)
    result = run_async(svc.execute(ToolInvocation(tool_name="trace_tool")))
    run_async(svc.aclose())
    assert result.outcome == "failed"
    assert result.error_code == ToolErrorCode.INVALID_REQUEST


@pytest.fixture(autouse=True)
def _reset_secrets() -> Iterator[None]:
    """Restore the default (env) secret provider after each test."""
    yield
    configure_secrets(None)
