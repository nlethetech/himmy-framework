"""Declarative HTTP tools: YAML spec → HttpToolConfig + auto-derived arg schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from himmy.config.agent_spec import AgentSpec
from himmy.config.http_tool_spec import HttpToolSpec, register_http_tools
from himmy.services.tools.models import (
    HttpAuthMode,
    HttpPaginationMode,
    ToolBackendKind,
)
from himmy.services.tools.registry import ToolRegistry


def test_schema_is_derived_from_path_and_args() -> None:
    spec = HttpToolSpec(
        name="get_user_repos",
        path="/users/{user}/repos",
        query=["sort", "per_page"],
    )
    schema = spec._derived_schema()
    assert set(schema["properties"]) == {"user", "sort", "per_page"}
    # path placeholders are required; query args are optional
    assert schema["required"] == ["user"]
    assert schema["additionalProperties"] is False


def test_explicit_args_schema_wins() -> None:
    custom = {"type": "object", "properties": {"x": {"type": "integer"}}}
    spec = HttpToolSpec(name="t", path="/{x}", args_schema=custom)
    assert spec._derived_schema() == custom  # the override is used verbatim


def test_to_config_maps_every_field() -> None:
    spec = HttpToolSpec(
        name="t",
        base_url="https://api.example.com",
        method="post",
        path="/items/{id}",
        query=["q"],
        body=["payload"],
        headers=["x_trace"],
        timeout_seconds=9.0,
    )
    cfg = spec.to_config()
    assert cfg.base_url == "https://api.example.com"
    assert cfg.method == "POST"  # upper-cased
    assert cfg.path_template == "/items/{id}"
    assert cfg.query_arg_names == ["q"]
    assert cfg.body_arg_names == ["payload"]
    assert cfg.header_arg_names == ["x_trace"]
    assert cfg.timeout_seconds == 9.0


def test_auth_bearer_mapping() -> None:
    spec = HttpToolSpec(
        name="t", base_url="https://x", auth={"type": "bearer", "env_var": "API_KEY"}
    )
    auth = spec.to_config().auth
    assert auth.mode is HttpAuthMode.BEARER
    assert auth.env_var == "API_KEY"


def test_unknown_auth_type_falls_back_to_none() -> None:
    spec = HttpToolSpec(name="t", base_url="https://x", auth={"type": "magic"})
    assert spec.to_config().auth.mode is HttpAuthMode.NONE


def test_register_adds_an_http_tool() -> None:
    registry = ToolRegistry()
    names = register_http_tools(
        registry,
        [HttpToolSpec(name="rate", base_url="https://x", path="/r", query=["c"])],
    )
    assert names == ["rate"]
    tool = registry.get("rate")
    assert tool is not None
    assert tool.kind is ToolBackendKind.HTTP
    assert tool.http_config is not None
    assert "c" in tool.args_json_schema["properties"]


def test_agent_spec_loads_http_tools_from_yaml() -> None:
    spec = AgentSpec.model_validate(
        {
            "name": "fx",
            "http_tools": [
                {
                    "name": "exchange_rate",
                    "description": "fx",
                    "base_url": "https://api.frankfurter.dev",
                    "path": "/v1/latest",
                    "query": ["base", "symbols"],
                }
            ],
        }
    )
    assert len(spec.http_tools) == 1
    assert spec.http_tools[0].name == "exchange_rate"


def test_typo_in_http_tool_field_is_rejected() -> None:
    # extra="forbid" — a mistyped field fails loudly instead of being ignored.
    with pytest.raises(ValidationError):
        HttpToolSpec(name="t", base_url="https://x", quary=["oops"])  # type: ignore[call-arg]


# ------------------------------------------------------------- hardened auth modes
def test_auth_secret_key_preferred_over_env_var() -> None:
    """``secret`` is the canonical credential-NAME key; ``env_var`` stays an alias."""
    spec = HttpToolSpec(
        name="t", base_url="https://x", auth={"type": "bearer", "secret": "TOK"}
    )
    assert spec.to_config().auth.env_var == "TOK"
    legacy = HttpToolSpec(
        name="t", base_url="https://x", auth={"type": "bearer", "env_var": "TOK"}
    )
    assert legacy.to_config().auth.env_var == "TOK"


def test_auth_api_key_header_alias_maps_to_header_mode() -> None:
    spec = HttpToolSpec(
        name="t",
        base_url="https://x",
        auth={"type": "api_key", "secret": "K", "header_name": "X-API-Key"},
    )
    auth = spec.to_config().auth
    assert auth.mode is HttpAuthMode.HEADER
    assert auth.header_name == "X-API-Key"


def test_auth_api_key_query_mode() -> None:
    spec = HttpToolSpec(
        name="t",
        base_url="https://x",
        auth={"type": "api_key_query", "secret": "K", "query_param": "apikey"},
    )
    auth = spec.to_config().auth
    assert auth.mode is HttpAuthMode.API_KEY_QUERY
    assert auth.query_param == "apikey"


def test_auth_basic_carries_username() -> None:
    spec = HttpToolSpec(
        name="t",
        base_url="https://x",
        auth={"type": "basic", "secret": "PW", "username": "svc"},
    )
    auth = spec.to_config().auth
    assert auth.mode is HttpAuthMode.BASIC
    assert auth.username == "svc"


# --------------------------------------------------------------------- egress + pagination
def test_egress_and_private_host_flags_flow_to_config() -> None:
    spec = HttpToolSpec(
        name="t",
        base_url="https://x",
        egress_allow_hosts=["api.internal"],
        allow_private_hosts=True,
        static_query={"v": "2024-01"},
    )
    cfg = spec.to_config()
    assert cfg.egress_allow_hosts == ["api.internal"]
    assert cfg.allow_private_hosts is True
    assert cfg.static_query == {"v": "2024-01"}


def test_pagination_mapping() -> None:
    spec = HttpToolSpec(
        name="t",
        base_url="https://x",
        pagination={
            "mode": "cursor",
            "items_path": "data.items",
            "cursor_path": "meta.next",
            "cursor_param": "after",
            "max_pages": 7,
        },
    )
    page = spec.to_config().pagination
    assert page.mode is HttpPaginationMode.CURSOR
    assert page.items_path == "data.items"
    assert page.cursor_path == "meta.next"
    assert page.cursor_param == "after"
    assert page.max_pages == 7


def test_idempotency_arg_added_to_schema_and_config() -> None:
    spec = HttpToolSpec(
        name="t",
        base_url="https://x",
        method="post",
        path="/orders",
        body=["sku"],
        idempotency_arg="request_id",
    )
    assert "request_id" in spec._derived_schema()["properties"]
    assert spec.to_config().idempotency_arg == "request_id"


def test_response_schema_becomes_output_schema_on_definition() -> None:
    schema = {"type": "object", "properties": {"id": {"type": "integer"}}}
    registry = ToolRegistry()
    HttpToolSpec(
        name="t", base_url="https://x", path="/r", response_schema=schema
    ).register(registry)
    definition = registry.get("t")
    assert definition is not None
    assert definition.output_json_schema == schema
