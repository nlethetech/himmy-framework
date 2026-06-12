"""Point the generic REST connector at an OpenAPI 3 document → safe agent tools.

This is the *infinite-leverage* connector: an operator hands the framework an OpenAPI
(Swagger) spec for ANY internal or third-party API and every operation becomes a native,
hardened :class:`~himmy.config.http_tool_spec.HttpToolSpec`. No per-endpoint Python is
written; the resulting tools inherit the whole tool-runtime posture — secrets-layer
credentials (never literals/logs), SSRF-guarded + DNS-pinned egress (optionally pinned to
an allow-list), a method allow-list, request/response schema validation, bounded
pagination, and no-blind-retry for non-idempotent mutations.

Usage::

    from himmy.config.openapi_spec import load_openapi_tools
    specs = load_openapi_tools(
        document,                      # a parsed dict, JSON/YAML text, or a file path
        secret="ACME_API_TOKEN",       # secrets-layer NAME for the bearer/api-key value
        egress_allow_hosts=["api.acme.internal"],
    )
    register_http_tools(registry, specs)

Only the methods on :data:`~himmy.services.tools.security.ALLOWED_HTTP_METHODS` are
imported; an operation can be marked ``requires_approval`` for every mutating verb via
``approve_mutations`` so a side-effecting tool is human-gated by default.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from himmy.config.http_tool_spec import _PLACEHOLDER, HttpToolSpec
from himmy.services.tools.security import ALLOWED_HTTP_METHODS

#: HTTP verbs OpenAPI puts under a path item (lower-cased operation keys).
_OPERATION_METHODS: tuple[str, ...] = (
    "get",
    "put",
    "post",
    "delete",
    "patch",
    "head",
    "options",
)

#: Methods that mutate state — gated behind approval when ``approve_mutations`` is set.
_MUTATING_METHODS: frozenset[str] = frozenset({"POST", "PUT", "PATCH", "DELETE"})

_NON_IDENTIFIER = re.compile(r"[^0-9a-zA-Z_]+")


def _slugify(text: str) -> str:
    """Turn a path / operationId into a safe snake_case tool-name fragment."""
    slug = _NON_IDENTIFIER.sub("_", text).strip("_").lower()
    return slug or "op"


def _load_document(document: Any) -> dict[str, Any]:
    """Coerce a dict, JSON/YAML text, or a file path into a parsed OpenAPI dict."""
    if isinstance(document, dict):
        return document
    if isinstance(document, (str, Path)):
        text = str(document)
        candidate = Path(text)
        # Treat short, single-line strings that name an existing file as a path; longer
        # blobs are parsed as inline JSON/YAML.
        if "\n" not in text and len(text) < 4096 and candidate.is_file():
            text = candidate.read_text(encoding="utf-8")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            import yaml

            parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise ValueError("OpenAPI document did not parse to a mapping")
        return parsed
    raise TypeError(f"unsupported OpenAPI document type: {type(document).__name__}")


def _resolve_ref(node: Any, root: dict[str, Any]) -> Any:
    """Resolve a single local ``$ref`` (``#/...``) one hop; non-refs pass through."""
    if isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if isinstance(ref, str) and ref.startswith("#/"):
            target: Any = root
            for part in ref[2:].split("/"):
                if not isinstance(target, dict) or part not in target:
                    return {}
                target = target[part]
            return target
    return node


def _base_url(document: dict[str, Any], override: str | None) -> str:
    """Pick the base URL: an explicit override wins, else the first declared server."""
    if override:
        return override.rstrip("/")
    servers = document.get("servers")
    if isinstance(servers, list) and servers:
        url = servers[0].get("url") if isinstance(servers[0], dict) else None
        if isinstance(url, str) and url:
            return url.rstrip("/")
    return ""


def _auth_block(
    document: dict[str, Any], secret: str | None
) -> dict[str, Any] | None:
    """Map the document's first usable security scheme to an ``auth`` spec block.

    Only honored when the caller supplies a ``secret`` NAME (we never invent or embed a
    credential). HTTP bearer → bearer; HTTP basic → basic; apiKey in header → header;
    apiKey in query → api_key_query. Anything else (OAuth2/OpenID flows) is left to the
    caller, who can pass a bearer token under ``secret``.
    """
    if not secret:
        return None
    schemes = (
        (document.get("components") or {}).get("securitySchemes")
        if isinstance(document.get("components"), dict)
        else None
    ) or {}
    for scheme in schemes.values():
        if not isinstance(scheme, dict):
            continue
        kind = str(scheme.get("type") or "").lower()
        if kind == "http":
            sub = str(scheme.get("scheme") or "").lower()
            if sub == "basic":
                return {"type": "basic", "secret": secret}
            return {"type": "bearer", "secret": secret}
        if kind == "apikey":
            location = str(scheme.get("in") or "header").lower()
            name = str(scheme.get("name") or "")
            if location == "query":
                return {"type": "api_key_query", "secret": secret, "query_param": name}
            return {"type": "header", "secret": secret, "header_name": name or "Authorization"}
    # No declared scheme but a secret was given → default to a bearer token.
    return {"type": "bearer", "secret": secret}


def _operation_name(method: str, path: str, operation: dict[str, Any]) -> str:
    """A deterministic, readable tool name (operationId preferred)."""
    op_id = operation.get("operationId")
    if isinstance(op_id, str) and op_id.strip():
        return _slugify(op_id)
    return f"{method.lower()}_{_slugify(path)}"


def _collect_params(
    method_obj: dict[str, Any], path_obj: dict[str, Any], root: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return (query arg names, header arg names) declared for this operation.

    Path-level and operation-level parameters are merged (operation wins on name).
    ``in: path`` params are NOT returned here — they come from the ``{placeholder}``
    template, which :class:`HttpToolSpec` already derives and marks required.
    """
    seen: dict[str, str] = {}  # name -> location
    for source in (path_obj.get("parameters"), method_obj.get("parameters")):
        if not isinstance(source, list):
            continue
        for raw in source:
            param = _resolve_ref(raw, root)
            if not isinstance(param, dict):
                continue
            name = param.get("name")
            location = str(param.get("in") or "").lower()
            if isinstance(name, str) and location in ("query", "header"):
                seen[name] = location
    query = [n for n, loc in seen.items() if loc == "query"]
    headers = [n for n, loc in seen.items() if loc == "header"]
    return query, headers


def _body_args(
    method_obj: dict[str, Any], root: dict[str, Any]
) -> tuple[list[str], dict[str, Any] | None]:
    """Return (top-level JSON body field names, the resolved request body schema)."""
    request_body = _resolve_ref(method_obj.get("requestBody"), root)
    if not isinstance(request_body, dict):
        return [], None
    content = request_body.get("content")
    if not isinstance(content, dict):
        return [], None
    media = content.get("application/json")
    if not isinstance(media, dict):
        return [], None
    schema = _resolve_ref(media.get("schema"), root)
    if not isinstance(schema, dict):
        return [], None
    props = schema.get("properties")
    names = list(props.keys()) if isinstance(props, dict) else []
    return names, schema


#: JSON-schema keywords the derived args schema preserves per property (so a typed
#: request contract — int/enum/bounds — survives from the spec to the tool).
_KEPT_SCHEMA_KEYS: frozenset[str] = frozenset(
    {
        "type",
        "enum",
        "format",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "pattern",
        "description",
    }
)


def _prune_property(schema: Any) -> dict[str, Any]:
    """Reduce a property schema to the constraint keywords the validator honors."""
    if not isinstance(schema, dict):
        return {"type": "string"}
    kept = {k: v for k, v in schema.items() if k in _KEPT_SCHEMA_KEYS}
    return kept or {"type": "string"}


def _args_schema(
    path: str,
    method_obj: dict[str, Any],
    path_obj: dict[str, Any],
    body_schema: dict[str, Any] | None,
    root: dict[str, Any],
) -> dict[str, Any]:
    """Build a TYPED args JSON Schema (path/query/header params + body fields).

    Path placeholders and any ``required`` query/header param or required body field are
    marked required; each property carries the spec's type/constraints (pruned to the
    keywords the runtime validator enforces) so the model gets a real, checkable contract
    rather than every arg degraded to a string.
    """
    properties: dict[str, Any] = {}
    required: list[str] = list(_PLACEHOLDER.findall(path))
    for name in required:
        properties.setdefault(name, {"type": "string"})

    for source in (path_obj.get("parameters"), method_obj.get("parameters")):
        if not isinstance(source, list):
            continue
        for raw in source:
            param = _resolve_ref(raw, root)
            if not isinstance(param, dict):
                continue
            param_name = param.get("name")
            location = str(param.get("in") or "").lower()
            if not isinstance(param_name, str) or location not in (
                "path",
                "query",
                "header",
            ):
                continue
            properties[param_name] = _prune_property(
                _resolve_ref(param.get("schema"), root) or {}
            )
            if param.get("required") and param_name not in required:
                required.append(param_name)

    if isinstance(body_schema, dict):
        body_props = body_schema.get("properties")
        if isinstance(body_props, dict):
            for name, sub in body_props.items():
                properties[name] = _prune_property(_resolve_ref(sub, root))
        for name in body_schema.get("required") or []:
            if isinstance(name, str) and name not in required:
                required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _response_schema(
    method_obj: dict[str, Any], root: dict[str, Any]
) -> dict[str, Any] | None:
    """The JSON schema of the first 2xx ``application/json`` response, if declared."""
    responses = method_obj.get("responses")
    if not isinstance(responses, dict):
        return None
    for status in ("200", "201", "default"):
        entry = _resolve_ref(responses.get(status), root)
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if not isinstance(content, dict):
            continue
        media = content.get("application/json")
        if isinstance(media, dict):
            schema = _resolve_ref(media.get("schema"), root)
            if isinstance(schema, dict):
                return schema
    return None


def load_openapi_tools(
    document: Any,
    *,
    base_url: str | None = None,
    base_url_env_var: str = "",
    secret: str | None = None,
    egress_allow_hosts: list[str] | None = None,
    allow_private_hosts: bool = False,
    approve_mutations: bool = False,
    validate_responses: bool = False,
    include_methods: frozenset[str] | None = None,
    timeout_seconds: float = 20.0,
) -> list[HttpToolSpec]:
    """Build one :class:`HttpToolSpec` per OpenAPI operation (hardened by construction).

    ``document`` is a parsed dict, JSON/YAML text, or a file path. ``base_url`` overrides
    the document's first server; ``base_url_env_var`` instead reads it from the secrets
    layer at call time (preferred for per-env config). ``secret`` is the secrets-layer
    NAME of the credential — the security scheme is mapped to the right auth mode. Set
    ``egress_allow_hosts`` to pin the connector to the API's host(s); ``approve_mutations``
    gates every POST/PUT/PATCH/DELETE behind approval; ``validate_responses`` attaches the
    declared 2xx response schema so a malformed body surfaces a typed error.
    ``include_methods`` narrows which verbs are imported (default: every allowed method).
    """
    root = _load_document(document)
    resolved_base = _base_url(root, base_url)
    auth = _auth_block(root, secret)
    allowed = (include_methods or ALLOWED_HTTP_METHODS) & ALLOWED_HTTP_METHODS
    paths = root.get("paths")
    if not isinstance(paths, dict):
        return []

    specs: list[HttpToolSpec] = []
    used_names: set[str] = set()
    for path, path_obj in paths.items():
        if not isinstance(path_obj, dict):
            continue
        for method in _OPERATION_METHODS:
            method_obj = path_obj.get(method)
            if not isinstance(method_obj, dict):
                continue
            upper = method.upper()
            if upper not in allowed:
                continue

            name = _operation_name(method, str(path), method_obj)
            # De-dupe collisions deterministically (two ops slugging to one name).
            if name in used_names:
                suffix = 2
                while f"{name}_{suffix}" in used_names:
                    suffix += 1
                name = f"{name}_{suffix}"
            used_names.add(name)

            query, headers = _collect_params(method_obj, path_obj, root)
            body_names, body_schema = _body_args(method_obj, root)
            args_schema = _args_schema(
                str(path), method_obj, path_obj, body_schema, root
            )
            response_schema = (
                _response_schema(method_obj, root) if validate_responses else None
            )
            description = str(
                method_obj.get("summary")
                or method_obj.get("description")
                or f"{upper} {path}"
            ).strip()

            specs.append(
                HttpToolSpec(
                    name=name,
                    description=description,
                    base_url=resolved_base,
                    base_url_env_var=base_url_env_var,
                    method=upper,
                    path=str(path),
                    query=query,
                    headers=headers,
                    body=body_names,
                    auth=auth,
                    egress_allow_hosts=list(egress_allow_hosts or []),
                    allow_private_hosts=allow_private_hosts,
                    requires_approval=approve_mutations and upper in _MUTATING_METHODS,
                    args_schema=args_schema,
                    response_schema=response_schema,
                    timeout_seconds=timeout_seconds,
                )
            )
    return specs


__all__ = ["load_openapi_tools"]
