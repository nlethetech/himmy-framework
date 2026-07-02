"""Declarative connector spec: describe an HTTP outbound connector in YAML.

Most outbound connectors are "call this third-party REST API with a bearer token"
(GitHub, Slack, a search provider). :class:`ConnectorSpec` lets you declare that whole
connector — its tools, its credential, its egress policy, its rate limit — in YAML, with
no Python, consistent with the existing :class:`~himmy.config.http_tool_spec.HttpToolSpec`
and :class:`~himmy.config.mcp_spec.MCPServerConfig` façades::

    connectors:
      - name: github
        description: GitHub REST API.
        base_url: https://api.github.com
        auth: { type: bearer, secret: GITHUB_TOKEN }
        rate_limit: { rate: 30, per_seconds: 60 }
        tools:
          - name: github_get_issue
            method: GET
            path: /repos/{owner}/{repo}/issues/{number}
          - name: github_create_issue
            method: POST
            path: /repos/{owner}/{repo}/issues
            body: [title, body]
            requires_approval: true      # side-effecting → human-in-the-loop
            idempotent: false            # never blind-retry a create

The credential is read from the **secrets layer** (``secret:`` names a secret, never a
literal). The connector reports *unavailable* (and is skipped cleanly by discovery) when
that secret is absent. Each declared tool becomes a native :class:`SpecToolConnector`
whose calls are SSRF-guarded, capability-gated, audited, rate-limited, and — for the
ones marked ``idempotent`` — deduped on retry, exactly like a hand-written connector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from pydantic import BaseModel, ConfigDict, Field

from himmy.config.http_tool_spec import _PLACEHOLDER
from himmy.connectors.fetcher import Fetcher, get_json
from himmy.connectors.sdk import (
    Capability,
    ConnectorContext,
    ConnectorError,
    OutboundToolConnector,
    RateLimiter,
    capability,
    redact_args,
)
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit._net import build_pinned_transport

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Collection

_SAFE_METHODS = frozenset({"GET", "HEAD"})


class ConnectorAuthSpec(BaseModel):
    """How a connector authenticates: a credential NAME resolved via the secrets layer."""

    model_config = ConfigDict(extra="forbid")

    type: str = "none"  # none | bearer | header | basic
    secret: str | None = None  # secret name (resolved via get_secret) — NEVER a literal
    header_name: str = "Authorization"  # for type=header
    username: str | None = None  # for type=basic (the password is `secret`)

    def header(self, value: str | None) -> dict[str, str]:
        """Build the auth header from the resolved secret ``value`` (empty if none)."""
        if self.type == "none" or not value:
            return {}
        if self.type == "bearer":
            return {"Authorization": f"Bearer {value}"}
        if self.type == "header":
            return {self.header_name: value}
        if self.type == "basic":
            import base64

            raw = f"{self.username or ''}:{value}".encode()
            return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}
        raise ConnectorError(f"unsupported connector auth type {self.type!r}")


class RateLimitSpec(BaseModel):
    """A declarative token-bucket rate limit for a connector."""

    model_config = ConfigDict(extra="forbid")

    rate: float = Field(gt=0)
    per_seconds: float = Field(default=1.0, gt=0)
    burst: float | None = None

    def build(self) -> RateLimiter:
        return RateLimiter(
            rate=self.rate, window=self.per_seconds, capacity=self.burst
        )


class ConnectorToolSpec(BaseModel):
    """One declarative tool inside a connector: an HTTP call the agent can make."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    method: str = "GET"
    path: str = "/"  # supports {placeholders} filled from the model's args
    query: list[str] = []  # arg names sent as query params
    body: list[str] = []  # arg names sent in the JSON body
    requires_approval: bool = False
    #: Whether the call is safe to repeat. Defaults to True for GET/HEAD (read-only) and
    #: False otherwise — a non-idempotent mutation is never blind-retried, and (when
    #: an ``idempotency_arg`` is supplied) its effect is deduped per key.
    idempotent: bool | None = None
    #: Arg name whose value is the idempotency key for a side-effecting call. When set,
    #: a repeat with the same key returns the first result without re-firing the effect.
    idempotency_arg: str | None = None

    def is_idempotent(self) -> bool:
        """Resolved idempotency: explicit override, else inferred from the method."""
        if self.idempotent is not None:
            return self.idempotent
        return self.method.upper() in _SAFE_METHODS

    def args_schema(self) -> dict[str, Any]:
        """Derive the args JSON Schema from path placeholders + query/body names."""
        path_args = _PLACEHOLDER.findall(self.path)
        names = list(dict.fromkeys([*path_args, *self.query, *self.body]))
        if self.idempotency_arg:
            names.append(self.idempotency_arg)
            names = list(dict.fromkeys(names))
        return {
            "type": "object",
            "properties": {n: {"type": "string"} for n in names},
            "required": path_args,
            "additionalProperties": False,
        }


class ConnectorSpec(BaseModel):
    """A declarative HTTP outbound connector: a credential, an egress policy, tools."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    base_url: str = Field(min_length=1)
    auth: ConnectorAuthSpec = Field(default_factory=ConnectorAuthSpec)
    #: Hosts the connector may reach (egress allow-list). Empty ⇒ any PUBLIC host
    #: (the SSRF guard still blocks private/loopback/reserved). Set this to pin the
    #: connector to its API's host for defence in depth.
    egress_allow_hosts: list[str] = Field(default_factory=list)
    allow_private_hosts: bool = False  # only for a trusted internal API
    rate_limit: RateLimitSpec | None = None
    timeout_seconds: float = 20.0
    tools: list[ConnectorToolSpec] = Field(default_factory=list)

    def build(
        self,
        *,
        fetcher: Fetcher | None = None,
        context: ConnectorContext | None = None,
    ) -> SpecToolConnector:
        """Instantiate the live :class:`SpecToolConnector` this spec describes."""
        return SpecToolConnector(self, fetcher=fetcher, context=context)


class SpecToolConnector(OutboundToolConnector):
    """An :class:`OutboundToolConnector` driven by a :class:`ConnectorSpec` (no code).

    Inherits the full security posture: the credential is read from the secrets layer,
    every request URL is :meth:`guard`-ed before fetching, and a side-effecting tool
    with an ``idempotency_arg`` runs at most once per key. The connector is *unavailable*
    (skipped cleanly by discovery and by :meth:`register_tools`) when its auth secret is
    configured-but-absent.
    """

    def __init__(
        self,
        spec: ConnectorSpec,
        *,
        fetcher: Fetcher | None = None,
        context: ConnectorContext | None = None,
    ) -> None:
        self.name = spec.name
        self._spec = spec
        if spec.auth.secret:
            self.required_secrets = (spec.auth.secret,)
        ctx = context or ConnectorContext()
        if spec.rate_limit is not None and ctx.rate_limiter is None:
            ctx = ConnectorContext(
                audit=ctx.audit,
                rate_limiter=spec.rate_limit.build(),
                retry=ctx.retry,
            )
        super().__init__(
            fetcher=fetcher,
            allow_private_hosts=spec.allow_private_hosts,
            egress_allow_hosts=spec.egress_allow_hosts or None,
            context=ctx,
        )

    def capability(self) -> Capability:
        """Available when the (optional) auth secret resolves."""
        return capability(requires_secrets=self.required_secrets)

    def _auth_headers(self) -> dict[str, str]:
        """Resolve the credential from the secrets layer and build the auth header."""
        secret_name = self._spec.auth.secret
        value = self.secret(secret_name) if secret_name else None
        return self._spec.auth.header(value)

    def _build_url(self, tool: ConnectorToolSpec, args: dict[str, Any]) -> str:
        """Assemble + SSRF-guard the request URL from the path template and args.

        Path placeholders are percent-encoded (so a model-supplied value can't escape
        the path / inject a host) and query args are URL-encoded; the final URL is then
        run through the connector's :meth:`guard` (public-only, plus the egress
        allow-list) before it is ever fetched.
        """
        path = tool.path
        for name in _PLACEHOLDER.findall(tool.path):
            if name not in args:
                raise ConnectorError(f"{tool.name} requires path arg {name!r}")
            path = path.replace("{" + name + "}", quote(str(args[name]), safe=""))
        url = self._spec.base_url.rstrip("/") + "/" + path.lstrip("/")
        params = {k: str(args[k]) for k in tool.query if args.get(k) is not None}
        if params:
            url = f"{url}?{urlencode(params)}"
        return self.guard(url)

    def _make_handler(self, tool: ConnectorToolSpec) -> Any:
        """Build the async handler that performs one tool's guarded HTTP call."""
        import asyncio

        method = tool.method.upper()

        def _do(args: dict[str, Any]) -> Any:
            url = self._build_url(tool, args)
            headers = self._auth_headers()
            return self._request(method, url, tool, args, headers)

        async def handler(args: dict[str, Any]) -> dict[str, Any]:
            self._throttle()
            if tool.idempotency_arg:
                # An idempotency key makes even a mutating call safe to repeat: the
                # store fires the effect once per key and returns the cached result on
                # a duplicate (so a re-issued tool call / retry can't double-send).
                key = str(args.get(tool.idempotency_arg) or "")
                if not key:
                    raise ConnectorError(
                        f"{tool.name} requires the idempotency arg "
                        f"{tool.idempotency_arg!r}"
                    )
                result = await asyncio.to_thread(
                    self.call_idempotent, f"{tool.name}:{key}", lambda: _do(args)
                )
            elif tool.is_idempotent():
                # A read (or an explicitly idempotent call) with no key may be retried.
                result = await asyncio.to_thread(
                    self._ctx.retry.run, lambda: _do(args), idempotent=True
                )
            else:
                # Non-idempotent: exactly one attempt (a blind retry could double-send).
                result = await asyncio.to_thread(lambda: _do(args))
            # The audit detail summarizes the (secret-redacted) args, never a credential.
            self._audit(
                tool.name, "allow", detail=f"{method} {dict(redact_args(args))}"
            )
            return {"ok": True, "data": result}

        return handler

    def _request(
        self,
        method: str,
        url: str,
        tool: ConnectorToolSpec,
        args: dict[str, Any],
        headers: dict[str, str],
    ) -> Any:
        """Perform the actual HTTP call through the guarded fetcher / pinned transport.

        GET/HEAD go through the connector's :class:`Fetcher` (which already guards every
        redirect hop) carrying the auth ``headers`` — so a declarative GET tool under
        ``auth:`` is actually authenticated. The fetcher drops the ``Authorization``
        header before any redirect that crosses to a different origin, so the credential
        is never replayed to a third-party host. A body-bearing method (POST/PUT/PATCH)
        uses a pinned httpx client — same DNS-pinning transport as the comms webhook tool
        — with redirects OFF so a 3xx can't bounce the request to an internal host.
        """
        if method in _SAFE_METHODS:
            return get_json(self.fetcher, url, headers=headers or None)
        import httpx

        payload = {k: args[k] for k in tool.body if k in args}
        with httpx.Client(
            timeout=self._spec.timeout_seconds,
            follow_redirects=False,
            transport=build_pinned_transport(
                allow_private=self._allow_private, allow_hosts=self._allow_hosts
            ),
        ) as client:
            resp = client.request(method, url, json=payload, headers=headers)
            resp.raise_for_status()
            if resp.headers.get("content-type", "").startswith("application/json"):
                return resp.json()
            return {"status_code": resp.status_code}

    def register(self, registry: ToolRegistry) -> list[str]:
        """Register every declared tool as a native local tool on ``registry``."""
        names: list[str] = []
        for tool in self._spec.tools:
            register_local_tool(
                registry,
                name=tool.name,
                handler=self._make_handler(tool),
                description=tool.description,
                args_json_schema=tool.args_schema(),
                read_only=tool.method.upper() in _SAFE_METHODS,
                # Method-DERIVED intent, NOT a hand-authored assertion: a GET/HEAD
                # endpoint may still have server-side side effects (``GET /trigger``,
                # an analytics beacon). Mark it non-authoritative so it never waives the
                # parallelism barrier, the timeout re-fire, or the ``:write`` grant —
                # mirroring :func:`register_http_tool`'s method-derived read-only.
                read_only_authoritative=False,
                requires_approval=tool.requires_approval,
                metadata={"connector": self.name},
            )
            names.append(tool.name)
        return names


def register_connector_specs(
    registry: ToolRegistry,
    specs: list[ConnectorSpec],
    *,
    context: ConnectorContext | None = None,
    fetchers: Collection[Fetcher] | None = None,
) -> list[str]:
    """Build + register every declarative connector, skipping unavailable ones.

    Each connector whose credential is absent is *skipped cleanly* (its tools are not
    registered) rather than failing the whole batch, so a partially-configured
    deployment still gets the connectors it CAN run. Returns every registered tool name.
    """
    names: list[str] = []
    fetcher_list = list(fetchers or [])
    for index, spec in enumerate(specs):
        fetcher = fetcher_list[index] if index < len(fetcher_list) else None
        connector = spec.build(fetcher=fetcher, context=context)
        names.extend(connector.register_tools(registry))
    return names


__all__ = [
    "ConnectorSpec",
    "ConnectorToolSpec",
    "ConnectorAuthSpec",
    "RateLimitSpec",
    "SpecToolConnector",
    "register_connector_specs",
]
