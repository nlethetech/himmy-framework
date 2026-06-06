"""Declarative HTTP tools: YAML spec → HttpToolConfig + auto-derived arg schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from himmy.config.agent_spec import AgentSpec
from himmy.config.http_tool_spec import HttpToolSpec, register_http_tools
from himmy.services.tools.models import HttpAuthMode, ToolBackendKind
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
