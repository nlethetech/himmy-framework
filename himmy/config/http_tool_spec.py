"""Declarative HTTP tools: define a REST call an agent can make, in ``agent.yaml``.

``HttpToolConfig`` already lets the framework call a REST endpoint with no per-tool code;
:class:`HttpToolSpec` is the friendly, YAML-shaped façade over it. A user writes::

    http_tools:
      - name: get_weather
        description: Current weather for a city.
        base_url: https://api.example.com
        path: /weather/{city}
        query: [units]
        auth: { type: bearer, secret: WEATHER_API_KEY }

and the agent gains a ``get_weather`` tool — no Python. The args JSON Schema is derived
automatically from the path placeholders + query/body/header names (path args required),
so the model knows exactly what to pass.

This is the **generic OpenAPI/REST connector**: point it at any internal or third-party
API and the agent gets safe tools. Every call is hardened by the tool runtime — the
credential is read from the secrets layer (never a literal, never logged), the URL is
SSRF-guarded and dialed through a DNS-pinned transport (optionally restricted to an
``egress_allow_hosts`` allow-list), the method is on an allow-list, request/response are
schema-validated, pagination is followed within a bounded cap, and a non-idempotent
mutation is never blind-retried. See :mod:`himmy.config.openapi_spec` to derive a whole
list of these specs from an OpenAPI 3 document automatically.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from himmy.services.tools.models import (
    HttpAuthConfig,
    HttpAuthMode,
    HttpPaginationConfig,
    HttpPaginationMode,
    HttpToolConfig,
)
from himmy.services.tools.registry import ToolRegistry, register_http_tool

_PLACEHOLDER = re.compile(r"\{(\w+)\}")

#: Friendly YAML ``auth.type`` aliases → the canonical :class:`HttpAuthMode`.
_AUTH_ALIASES: dict[str, HttpAuthMode] = {
    "API_KEY": HttpAuthMode.HEADER,  # api-key header is the common case
    "APIKEY": HttpAuthMode.HEADER,
    "API-KEY": HttpAuthMode.HEADER,
    "API_KEY_HEADER": HttpAuthMode.HEADER,
    "API_KEY_QUERY": HttpAuthMode.API_KEY_QUERY,
    "QUERY": HttpAuthMode.API_KEY_QUERY,
}


class HttpToolSpec(BaseModel):
    """One declarative HTTP/REST tool, authored in YAML."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    base_url: str = ""  # literal base URL …
    base_url_env_var: str = ""  # … or read it from this secret name (preferred)
    method: str = "GET"
    path: str = "/"  # supports {placeholders} filled from the model's args
    query: list[str] = []  # arg names sent as query params
    body: list[str] = []  # arg names sent in the JSON body
    headers: list[str] = []  # arg names sent as headers
    #: {type: none|bearer|header|api_key|api_key_query|basic, secret, header_name,
    #: query_param, username}. ``secret`` (or ``env_var``) names a credential resolved
    #: through the secrets layer — NEVER a literal value.
    auth: dict[str, Any] | None = None
    #: Static query params always sent (e.g. an API version): {"v": "2023-01"}.
    static_query: dict[str, str] = {}
    #: Egress allow-list (host names); empty ⇒ any PUBLIC host (private/loopback still
    #: blocked). Pin to the API's host for defence in depth.
    egress_allow_hosts: list[str] = []
    #: Only for a trusted internal API behind a private address — skips the public-IP
    #: SSRF check. Leave False for anything reachable from untrusted prompts.
    allow_private_hosts: bool = False
    #: Pagination: {mode: cursor|page|link_header, items_path, cursor_path, ...}.
    pagination: dict[str, Any] | None = None
    #: Arg whose value is an idempotency key for a side-effecting call (sent as
    #: ``Idempotency-Key``), making a safe retry a no-op upstream.
    idempotency_arg: str | None = None
    timeout_seconds: float = 15.0
    requires_approval: bool = False
    args_schema: dict[str, Any] | None = None  # optional explicit JSON Schema override
    #: Optional JSON Schema the response is validated against (typed errors on mismatch).
    response_schema: dict[str, Any] | None = None

    def _auth(self) -> HttpAuthConfig:
        if not self.auth:
            return HttpAuthConfig()
        raw = str(self.auth.get("type") or self.auth.get("mode") or "none").upper()
        mode = _AUTH_ALIASES.get(raw)
        if mode is None:
            try:
                mode = HttpAuthMode(raw)
            except ValueError:
                mode = HttpAuthMode.NONE
        # ``secret`` is the preferred key (it's a secret NAME); ``env_var`` stays as an
        # accepted alias so existing YAML keeps working.
        secret_name = self.auth.get("secret") or self.auth.get("env_var")
        return HttpAuthConfig(
            mode=mode,
            env_var=secret_name,
            header_name=self.auth.get("header_name"),
            query_param=self.auth.get("query_param"),
            username=self.auth.get("username"),
        )

    def _pagination(self) -> HttpPaginationConfig:
        if not self.pagination:
            return HttpPaginationConfig()
        raw = str(self.pagination.get("mode") or "none").upper()
        try:
            mode = HttpPaginationMode(raw)
        except ValueError:
            mode = HttpPaginationMode.NONE
        page = self.pagination
        return HttpPaginationConfig(
            mode=mode,
            items_path=str(page.get("items_path") or ""),
            cursor_path=str(page.get("cursor_path") or ""),
            cursor_param=str(page.get("cursor_param") or "cursor"),
            page_param=str(page.get("page_param") or "page"),
            max_pages=int(page.get("max_pages") or 20),
        )

    def _derived_schema(self) -> dict[str, Any]:
        if self.args_schema is not None:
            return self.args_schema
        path_args = _PLACEHOLDER.findall(self.path)
        names = list(
            dict.fromkeys([*path_args, *self.query, *self.body, *self.headers])
        )
        if self.idempotency_arg:
            names.append(self.idempotency_arg)
            names = list(dict.fromkeys(names))
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
            egress_allow_hosts=list(self.egress_allow_hosts),
            allow_private_hosts=self.allow_private_hosts,
            static_query=dict(self.static_query),
            pagination=self._pagination(),
            idempotency_arg=self.idempotency_arg,
        )

    def register(self, registry: ToolRegistry) -> None:
        """Register this spec as an HTTP tool on ``registry``."""
        register_http_tool(
            registry,
            name=self.name,
            http_config=self.to_config(),
            description=self.description,
            args_json_schema=self._derived_schema(),
            output_json_schema=self.response_schema,
            requires_approval=self.requires_approval,
            timeout_seconds=self.timeout_seconds,
        )


def register_http_tools(registry: ToolRegistry, specs: list[HttpToolSpec]) -> list[str]:
    """Register every declarative HTTP tool; returns the names."""
    for spec in specs:
        spec.register(registry)
    return [spec.name for spec in specs]


__all__ = ["HttpToolSpec", "register_http_tools"]
