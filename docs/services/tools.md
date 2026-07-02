# Tools service

> The tools kernel: one registry, two execution backends (`LOCAL` / `HTTP`), policy hooks, JSON-Schema validation, SSRF hardening, and event emission — the single seam through which a model's tool calls actually run.

## Overview

`himmy/services/tools/` is the dispatcher that turns a model-requested tool call
into a validated, policy-gated, observable execution. It owns:

- a serializable catalog of tool definitions (`ToolRegistry` + `ToolDefinition`),
- a dispatcher (`ToolService`) that runs every documented execution guarantee
  end-to-end (approval gate → input validation → pre-hook → timeout/retry dispatch
  → post-hook → output validation → events),
- security helpers for the HTTP backend (`security.py`: SSRF/path-traversal guards,
  secret redaction),
- a stdlib-only JSON-Schema validator with optional `jsonschema` upgrade
  (`validation.py`),
- a binding seam to the inference layer (`runtime_adapter.py` plus
  `ToolService.bound_tools` / `ToolService.tool_executor`),
- small-model ergonomics: arg coercion, schema hints, tool-name repair
  (`repair.py`), and read/write intent tagging (`access.py`).

The tool layer and the inference layer are deliberately decoupled: inference
receives **pure-data** `BoundTool`s (schemas only) plus one `ToolExecutor`
callback. No tool-layer Python callable ever crosses into the inference kernel.

## Module map

| File | Responsibility |
| --- | --- |
| `service.py` | `ToolService` — the dispatcher. Approval gate, validation, hooks, per-tool timeout + bounded retry, `LOCAL`/`HTTP` dispatch, event emission, `bound_tools()` / `tool_executor()` binding seams. |
| `registry.py` | `ToolRegistry` (name→definition + name→handler maps) and the `register_local_tool` / `register_http_tool` helpers. |
| `models.py` | Data shapes: `ToolDefinition`, `ToolInvocation`, `ToolExecutionResult`, `ToolPolicyDecision`, `ToolBackendKind`, `ToolErrorCode`, `HttpToolConfig`, `HttpAuthConfig`, `HttpAuthMode`. |
| `security.py` | HTTP-backend hardening: `safe_path_arg` / `build_safe_path` / `assemble_url` (path-traversal + host pinning), `ALLOWED_HTTP_METHODS`, `redact_mapping` / `is_sensitive_key` (secret redaction), `ToolSecurityError`. |
| `validation.py` | `validate_against_schema` — offline JSON-Schema subset validator, delegating to `jsonschema` when installed. |
| `runtime_adapter.py` | Optional pydantic-ai binding: `build_arg_model` (JSON Schema → pydantic model) and `ToolServiceToolset.as_pydantic_ai_toolset`. Import-safe without `pydantic_ai`. |
| `repair.py` | Tool-name near-miss recovery: `resolve_tool_name` (typo auto-fix), `unknown_tool_message` (model-facing correction). |
| `access.py` | Read/write intent: `classify_read_only` (infer from verb), `describe_for_model` (append a read-only/WRITE hint to a description). |
| `__init__.py` | Public surface re-exporting the above. |

## Key abstractions

### `ToolDefinition` (`models.py`)

The serializable catalog entry for one tool. The local Python handler is kept
*out* of this model (the registry owns a separate `name → handler` map) so a
definition stays JSON-serializable and projectable to an `EntityRecord`
(`to_record()`, kind `tool_definition`). Notable fields:

- `name`, `kind` (`ToolBackendKind.LOCAL | HTTP`), `description`
- `args_json_schema` / `output_json_schema` — validated before/after execution
- `requires_approval: bool` — gated before any user pre-hook
- `read_only: bool | None` — read vs mutate intent surfaced to the model. Declaring
  `read_only=True` on `register_local_tool` / `register_http_tool` is AUTHORITATIVE
  (`read_only_authoritative=True`): a first-class STRUCTURAL assertion of no side effect that
  lets the tool join a concurrent read-batch and be re-fired on a bare timeout. `read_only=False`
  is a strict write barrier (never parallelised, never timeout-retried) — set it on a
  side-effecting GET or a read-named mutator (`get_then_charge`). Leaving it unset (`None`) is
  fail-closed: intent falls back to the strict name gate and, when the name is ambiguous, the
  tool is treated as a sequential WRITE barrier. A method-DERIVED read-only (an HTTP `GET`) is a
  parallelism hint only, NOT authoritative — it too routes through the fail-closed name gate, so
  a `GET /trigger` with a server-side side effect is never auto-parallelised.
- `timeout_seconds`, `retry_hints` (dict), `sequential`
- `http_config: HttpToolConfig | None` — declarative REST connector
- `sensitive_arg_names: list[str]` — arg keys redacted from emitted events

### `ToolRegistry` (`registry.py`)

In-memory catalog. `register(definition, handler=None)` stores the definition and
(for `LOCAL`) its callable; when an `EntityRegistry` is supplied, each
registration is also projected to a `tool_definition` record. `remove(name)`
drops a definition + handler (used by short-lived per-run bindings) but does **not**
retract the append-only entity record. `register_local_tool` / `register_http_tool`
are the friendly constructors.

### `ToolService` (`service.py`)

The dispatcher. Constructed with a registry plus optional `pre_execution_hook`,
`post_execution_hook`, `event_sink`, and HTTP-client knobs. Owns a lazily-built,
**shared** `httpx.AsyncClient` (redirects disabled) for all HTTP tools; `aclose()`
closes it if owned.

### Pre/post-execution hooks

```python
PreExecutionHook  = Callable[[ToolInvocation, ToolDefinition], Awaitable[ToolPolicyDecision]]
PostExecutionHook = Callable[[ToolExecutionResult, ToolDefinition], Awaitable[Any]]
```

- **Pre-hook** returns a `ToolPolicyDecision`. `allow=False` → `POLICY_BLOCKED`
  (outcome `denied`). A non-`None` `transformed_args` replaces the args and is
  **re-validated** against the schema, so a hook cannot smuggle in an invalid
  payload.
- **Post-hook** may reshape the result (a non-`None` return replaces it). Crucially,
  **output validation runs AFTER the post-hook**, so a reshaping hook cannot bypass
  the schema check. A raising post-hook fails the call with `EXECUTION_ERROR`.

### Binding to inference: `BoundTool` and `ToolExecutor`

These types live in `himmy/services/inference/models.py` and are imported lazily
by the tool service:

- `ToolService.bound_tools(names=None)` → `list[BoundTool]`. A `BoundTool` is
  **pure data** (`name`, `description`, `args_json_schema`, `output_json_schema`,
  `read_only`) — no callable. The model advertises these.
- `ToolService.tool_executor()` → `ToolExecutor`, i.e.
  `Callable[[str, dict], Awaitable[ToolReturnRecord]]`. This is the one explicit
  seam: it routes `(tool_name, args)` through `ToolService.execute` (so every hook,
  timeout, retry, and event still applies) and normalizes the result into a
  `ToolReturnRecord`. The runtime attaches both `bound_tools` and this executor to
  the `InferenceRequest`.

The optional pydantic-ai path (`runtime_adapter.py`) instead builds real `Tool`s
with a generated typed argument model (`build_arg_model`) so pydantic-ai validates
args and can raise `ModelRetry`; the proxy still calls `ToolService.execute`. This
requires the `[providers]` extra and is import-safe without it.

## How it works / data flow

A single call (`ToolService.execute(invocation)`) flows through these stages
(see `service.py`):

1. **Lookup + `TOOL_CALLED` event.** The definition is fetched; a `TOOL_CALLED`
   event is emitted with args redacted via `redact_mapping` (sensitive keys +
   `definition.sensitive_arg_names`). Unknown tool → `NOT_FOUND`.

2. **Approval gate.** If `requires_approval` and the invocation metadata lacks
   `"approved"`, fail closed with `POLICY_BLOCKED` (outcome `denied`). This runs
   **before** the user pre-hook.

3. **Input validation.** Against `args_json_schema` (when it's an object schema).
   If `lenient_args` is on (default), args are first coerced (`_coerce_lenient_args`):
   hallucinated keys are dropped when `additionalProperties: false`, and `null`-valued
   optionals are dropped (treated as omitted). Required/typed checks stay strict.
   On failure → `INVALID_REQUEST`, and the error message appends a compact
   `_schema_hint` so the model can self-correct next turn.

4. **Pre-execution hook.** (if configured) → `POLICY_BLOCKED` on deny;
   transformed args are re-validated.

5. **Dispatch with timeout + bounded retry.** The effective timeout is
   `definition.timeout_seconds` → `http_config.timeout_seconds` → service default
   (30s). `_RetryPolicy.from_hints` builds an exponential-backoff policy from
   `retry_hints`; only `RATE_LIMITED`, `TIMEOUT`, `PROVIDER_UNAVAILABLE` are retried.
   - `LOCAL` → `_dispatch_local`: calls the registered handler (sync or async; an
     awaitable is awaited).
   - `HTTP` → `_dispatch_http`: resolves base URL (env var preferred), builds a
     safe path, pins the host, applies query/body/header args + env-backed auth,
     and calls the shared client with `follow_redirects=False`. Status handling:
     `429` → `RATE_LIMITED`, `>=500` → `PROVIDER_UNAVAILABLE`, `3xx` →
     `INVALID_REQUEST` (redirects are surfaced, never followed), `>=400` →
     `INVALID_REQUEST`. Body is parsed as JSON, falling back to `{"text": ...}`.

6. **Post-execution hook** (may reshape the result).

7. **Output validation** against `output_json_schema` (if set), **after** the
   post-hook → `OUTPUT_VALIDATION` on failure.

8. **`TOOL_COMPLETED` / `TOOL_FAILED` event** with latency, and the
   `ToolExecutionResult` is returned. Failures are built and emitted by `_fail`
   (which never echoes secrets in its message).

### SSRF / security hardening (`security.py`)

The HTTP connector executes paths assembled from model-synthesized arguments, so
it's the most security-sensitive surface. Guards:

- `safe_path_arg` URL-encodes one path arg and rejects path separators (`/ \`),
  query/fragment markers (`? #`), control chars (incl. CR/LF), and literal `.`/`..`.
- `build_safe_path` formats only the `{name}` placeholders referenced by the
  template; a missing one raises `KeyError` (→ `INVALID_REQUEST`).
- `assemble_url` verifies the resolved URL's scheme+netloc still equals the
  configured base host (so an encoded-but-malicious path cannot pivot to another
  origin) and that the scheme is http/https.
- `ALLOWED_HTTP_METHODS` is an allow-list (`GET POST PUT PATCH DELETE HEAD OPTIONS`).
- `redact_mapping` / `is_sensitive_key` redact secret-looking keys (by hint
  substring like `token`/`secret`/`authorization`, or explicit `extra_keys`)
  before args/headers ever reach an event.

> Note: this module pins HTTP tools to a *configured* base host. Tools that fetch a
> *model-supplied* URL (the `web` pack's `web_fetch` / `http_request`) use a
> different guard, `guard_url` in `himmy/toolkit/_net.py` — see
> [toolkit](../architecture/toolkit.md).

### Schema validation (`validation.py`)

`validate_against_schema(value, schema)` returns an error string or `None`. When
`jsonschema` is importable it delegates for full Draft-2020-12 fidelity; otherwise
a stdlib-only subset runs (covers `type`, `required`, nested `properties`,
`additionalProperties`, array `items`/`minItems`/`maxItems`, `enum`/`const`,
numeric and string constraints, and `oneOf`/`anyOf`/`allOf`). Unknown keywords are
ignored permissively. The import is cached/lazy, so the module is import-safe.

### Schema repair & access intent (`repair.py`, `access.py`)

- `resolve_tool_name` (repair) does exact → case-insensitive → confident-typo
  matching (`difflib`, cutoff 0.82) so `egg_total` → `egg_totals` recovers with no
  extra turn; `unknown_tool_message` composes a did-you-mean + tool list for the
  model. These are pure functions reused by text-protocol inference providers.
- `classify_read_only` / `describe_for_model` (access) tag a description with a
  read-only or WRITE hint (explicit `read_only` flag wins; otherwise inferred from
  the verb) so a small model doing a look-up doesn't pick a write tool. An
  ambiguous name yields no tag rather than a wrong one.

## Configuration

`ToolService.__init__` knobs:

- `pre_execution_hook`, `post_execution_hook`, `event_sink`
- `default_timeout_seconds` (default `30.0`)
- `http_client` (inject for tests), `http_max_connections` (100),
  `http_max_keepalive_connections` (20)
- `lenient_args` (default `True`)

`ToolDefinition` per-tool knobs: `timeout_seconds`, `retry_hints`
(`max_attempts`/`max_retries`, `base_delay_seconds`/`backoff_seconds`,
`max_delay_seconds`, `backoff_multiplier`), `requires_approval`,
`sensitive_arg_names`, `read_only`, `sequential`.

HTTP auth (`HttpAuthMode`, resolved from env at call time, never stored):
`NONE`, `BEARER`, `HEADER` (custom `header_name`), `BASIC` (env holds `user:pass`,
base64-encoded here), `PREENCODED_BASIC` (env holds an already-encoded credential).
Base URL comes from `base_url_env_var` (preferred) else the literal `base_url`.

## Extension points

- **Add a local tool:** `register_local_tool(registry, name=..., handler=..., args_json_schema=..., ...)`.
  Handler is `(args: dict) -> result` (sync or async).
- **Add an HTTP tool:** `register_http_tool(registry, name=..., http_config=HttpToolConfig(...))`.
  No per-tool client code. (Authored declaratively in YAML via `HttpToolSpec` in
  `himmy/config/http_tool_spec.py` — see [toolkit](../architecture/toolkit.md).)
- **Bridge an MCP server's tools:** `register_mcp_tools` registers each as a `LOCAL`
  tool — see [mcp](./mcp.md).
- **Policy:** supply a `pre_execution_hook` (block/transform) and/or
  `post_execution_hook` (reshape).
- **Bind to inference:** use `bound_tools()` + `tool_executor()`, or
  `ToolServiceToolset.as_pydantic_ai_toolset()` with the `[providers]` extra.

## Gotchas & invariants

- **Output validation runs after the post-hook** — by design, so a reshaping hook
  can't bypass it.
- **Approval is gated before the user pre-hook**, and a missing approval fails
  closed (`POLICY_BLOCKED` / `denied`).
- **The HTTP client never follows redirects** (SSRF guard); a `3xx` is surfaced as
  an error, not returned as content.
- **Secrets never leak**: args/headers are redacted on events; error messages for
  HTTP failures report only the exception *type* / status, never header values.
- **Transformed pre-hook args are re-validated**; never trust a hook to keep the
  schema.
- **`BoundTool` carries no callable** — execution always flows back through
  `tool_executor()` → `execute()`, even from the pydantic-ai path. Unknown names
  fail closed via `execute`.
- **`registry.remove()` does not retract entity records** (the projection is
  append-only).
- **Only transient codes retry** (`RATE_LIMITED`, `TIMEOUT`,
  `PROVIDER_UNAVAILABLE`); everything else fails on the first attempt.
- **A broken `event_sink` never fails a tool call** (`_emit` swallows sink errors).

## Recently added (modules post-dating this doc — 2026-06-08)

- **MCP is its own service** — `himmy/services/mcp/` (see [`mcp.md`](./mcp.md)); MCP tools register as LOCAL `ToolDefinition`s through this service.
- **`repair.py`** — small-model tool-call recovery: `resolve_tool_name` fuzzy-matches a near-miss (≥0.82 auto-correct, ≤3 suggestions); only remaps the name, never invents args.
- **`access.py`** — read-vs-change intent (`classify_read_only` verb heuristic; `describe_for_model` tags the description) to steer small models away from wrong-tool selection.
- **`security.py`** — HTTP-connector guards: path-encoding + base-host re-pinning (SSRF / path-traversal), cross-host pagination-`Link` rejection, recursive secret redaction.
- **`validation.py`** — input **and** output JSON-Schema validation (`jsonschema` when present, offline subset otherwise); output validated *after* the post-hook.

These small-model surfaces feed the [learning service](./learning.md) (tool reputation from `TOOL_*` events).

> Open P0 (latest tools-reconciliation report): MCP tool-input-schema normalization is absent — `services/mcp/` passes `inputSchema` as-is; provider rejections (Gemini inline-vs-Kimi-`$defs`, null-unions, strict keyword strip) need a per-target normalizer.

## Related docs

- [mcp](./mcp.md) — MCP servers as a tool source.
- [toolkit](../architecture/toolkit.md) — built-in tool packs, declarative
  `HttpToolSpec`, and the `guard_url` SSRF guard for model-supplied URLs.
- [connectors](./connectors.md) — domain connectors registered as tools.
