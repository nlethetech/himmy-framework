"""Declarative HTTP tools: define a REST call an agent can make, in ``agent.yaml``.

``HttpToolConfig`` already lets the framework call a REST endpoint with no per-tool code;
:class:`HttpToolSpec` is the friendly, YAML-shaped façade over it. A user writes::

    http_tools:
      - name: get_weather
        description: Current weather for a city.
        base_url: https://api.example.com
        path: /weather/{city}
        query: [units]
        auth: { type: bearer, env_var: WEATHER_API_KEY }

and the agent gains a ``get_weather`` tool — no Python. The args JSON Schema is derived
automatically from the path placeholders + query/body/header names (path args required),
so the model knows exactly what to pass.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from himmy.services.tools.models import HttpAuthConfig, HttpAuthMode, HttpToolConfig
from himmy.services.tools.registry import ToolRegistry, register_http_tool

_PLACEHOLDER = re.compile(r"\{(\w+)\}")


class HttpToolSpec(BaseModel):
    """One declarative HTTP/REST tool, authored in YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    base_url: str = ""  # literal base URL …
    base_url_env_var: str = ""  # … or read it from this env var (preferred for config)
    method: str = "GET"
    path: str = "/"  # supports {placeholders} filled from the model's args
    query: list[str] = []  # arg names sent as query params
    body: list[str] = []  # arg names sent in the JSON body
    headers: list[str] = []  # arg names sent as headers
    auth: dict[str, Any] | None = (
        None  # {type: none|bearer|header|basic, env_var, header_name}
    )
    timeout_seconds: float = 15.0
    requires_approval: bool = False
    args_schema: dict[str, Any] | None = None  # optional explicit JSON Schema override

    def _auth(self) -> HttpAuthConfig:
        if not self.auth:
            return HttpAuthConfig()
        raw = str(self.auth.get("type") or self.auth.get("mode") or "none").upper()
        try:
            mode = HttpAuthMode(raw)
        except ValueError:
            mode = HttpAuthMode.NONE
        return HttpAuthConfig(
            mode=mode,
            env_var=self.auth.get("env_var"),
            header_name=self.auth.get("header_name"),
        )

    def _derived_schema(self) -> dict[str, Any]:
        if self.args_schema is not None:
            return self.args_schema
        path_args = _PLACEHOLDER.findall(self.path)
        names = list(
            dict.fromkeys([*path_args, *self.query, *self.body, *self.headers])
        )
        return {
            "type": "object",
            "properties": {n: {"type": "string"} for n in names},
            "required": path_args,  # path placeholders must be supplied
            "additionalProperties": False,
        }

    def to_config(self) -> HttpToolConfig:
        return HttpToolConfig(
            base_url_env_var=self.base_url_env_var,
            base_url=self.base_url,
            method=self.method.upper(),
            path_template=self.path,
            auth=self._auth(),
            query_arg_names=list(self.query),
            body_arg_names=list(self.body),
            header_arg_names=list(self.headers),
            timeout_seconds=self.timeout_seconds,
        )

    def register(self, registry: ToolRegistry) -> None:
        """Register this spec as an HTTP tool on ``registry``."""
        register_http_tool(
            registry,
            name=self.name,
            http_config=self.to_config(),
            description=self.description,
            args_json_schema=self._derived_schema(),
            requires_approval=self.requires_approval,
            timeout_seconds=self.timeout_seconds,
        )


def register_http_tools(registry: ToolRegistry, specs: list[HttpToolSpec]) -> list[str]:
    """Register every declarative HTTP tool; returns the names."""
    for spec in specs:
        spec.register(registry)
    return [spec.name for spec in specs]


__all__ = ["HttpToolSpec", "register_http_tools"]
