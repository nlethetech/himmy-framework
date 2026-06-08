# MCP service

> A transport-direct stdio JSON-RPC 2.0 client for the Model Context Protocol — no SDK — that bridges any stdio MCP server's tools into the Himmy `ToolService`.

## Overview

`himmy/services/mcp/` lets a Himmy agent consume tools from any stdio MCP server.
It is implemented **against the wire format directly** (newline-delimited JSON-RPC
2.0 over a subprocess's stdin/stdout), not the `mcp`/pydantic-ai SDKs, so the core
stays dependency-light and the entire path is exercisable offline against a mock
server.

Two pieces:

- `MCPClient` — launches and speaks to one server subprocess, with a background
  reader task that de-multiplexes responses by request id.
- `register_mcp_tools` — discovers a server's tools (`tools/list`) and registers
  each as a `LOCAL` `ToolDefinition` whose handler proxies through the client, so
  MCP tools flow through the **same** pipeline as native ones (arg validation,
  approval gating, events, lineage).

## Module map

| File | Responsibility |
| --- | --- |
| `client.py` | `MCPClient` — the stdio JSON-RPC client: `connect` (spawn + handshake), `request`/`notify`, the `_read_loop` demultiplexer, `list_tools`, `call_tool`, `aclose`. Plus `MCPError` and `DEFAULT_PROTOCOL_VERSION`. |
| `connector.py` | `register_mcp_tools` — bridge a server's tools into a `ToolRegistry` as `LOCAL` tools. |
| `models.py` | `MCPServerSpec` (how to launch a server), `MCPTool` (an advertised tool), `MCPToolResult` (a normalized `tools/call` outcome). |
| `__init__.py` | Public surface. |

## Key abstractions

### `MCPServerSpec` (`models.py`)

How to launch a stdio MCP server: `command`, `args`, `env`, `cwd`. The `env` dict
is **overlaid on the inherited parent environment** — MCP servers are
operator-configured *trusted* processes (unlike the untrusted code the sandbox
kernel runs), so they inherit the parent env by default.

### `MCPTool` / `MCPToolResult`

- `MCPTool`: `name`, `description`, `input_schema` (JSON Schema — becomes the tool's
  `args_json_schema`).
- `MCPToolResult`: `text` (the concatenation of the result's text content blocks —
  the common case), `content` (the raw block list), `structured` (the optional
  `structuredContent`), and `is_error` (mirrors the MCP `isError` flag).

### `MCPClient` (`client.py`)

Wraps one already-spawned subprocess; use the `connect` classmethod. Lifecycle:

```python
client = await MCPClient.connect(MCPServerSpec(command="my-mcp-server"))
try:
    tools  = await client.list_tools()
    result = await client.call_tool("search", {"q": "acme"})
finally:
    await client.aclose()
```

Also usable as an async context manager (`__aenter__` / `__aexit__`).

## How it works / data flow

### Connect + handshake

`MCPClient.connect` spawns the server with `asyncio.create_subprocess_exec`
(stdin/stdout/stderr as pipes, env = `{**os.environ, **spec.env}`, a 4 MiB stream
limit because large tool results exceed the 64 KiB default). It then:

1. starts the background `_read_loop` task,
2. sends the `initialize` request (advertising `protocolVersion`, empty
   `capabilities`, and `clientInfo`), capturing `serverInfo` and the negotiated
   `protocolVersion`,
3. sends the `notifications/initialized` notification.

`DEFAULT_PROTOCOL_VERSION` is `"2024-11-05"`. On any handshake failure it `aclose`s
and re-raises.

### Request/response demultiplexing (the background reader)

Each `request(method, params)` allocates a monotonically increasing integer id,
registers a `Future` in `self._pending[id]`, writes the JSON-RPC frame, and awaits
the future with a per-request timeout (default 30s; a timeout removes the pending
entry and raises `HimmyError`).

The single `_read_loop` task reads stdout line-by-line and dispatches:

- **EOF** → fail every pending future with "MCP server closed the connection" and
  return.
- non-JSON lines are ignored (stdout noise tolerance).
- a message whose `id` is `None` or not in `_pending` is skipped — this is how
  **server-initiated notifications and server→client requests are handled**: they
  simply have no matching pending future and are dropped (unhandled by design).
- otherwise the matching future is resolved: an `"error"` payload becomes an
  `MCPError(code, message, data)`; a result payload sets the future's result.

`notify(method, params)` sends a notification (no id, no response). `_fail_pending`
rejects all in-flight requests when the server dies or the client closes.

### `aclose`

Idempotent: marks closed, fails pending futures, cancels the reader task, closes
stdin, `terminate()`s the process (waiting up to 2s), and `kill()`s it if still
running.

### `list_tools` / `call_tool`

- `list_tools()` → `tools/list`, mapping each entry to an `MCPTool` (using
  `inputSchema`).
- `call_tool(name, arguments)` → `tools/call`, joining the `type == "text"` content
  blocks into `MCPToolResult.text` and carrying `content`, `structuredContent`, and
  `isError`.

### Wiring MCP servers as a tool source (`connector.py`)

`register_mcp_tools(registry, client, *, prefix="", requires_approval=False, names=None)`:

- discovers the server's tools via `client.list_tools()`,
- optionally restricts to `names`,
- registers each as a `LOCAL` tool named `f"{prefix}{tool.name}"` (the `prefix`
  namespaces tools to avoid collisions across multiple servers),
- uses the MCP `inputSchema` as the tool's `args_json_schema`,
- tags `metadata={"backend": "mcp", "mcp_tool": tool.name}`,
- returns the registered names.

The generated handler calls `client.call_tool(...)` and returns
`{"text", "is_error", "content", "structured"}`. **MCP tool errors are returned
in-band** (via `is_error`) rather than raised, matching the protocol's semantics.

Because each MCP tool is an ordinary `LOCAL` `ToolDefinition`, it inherits the full
`ToolService` pipeline: schema validation against the MCP `inputSchema`, approval
gating, the pre/post hooks, timeout/retry, events, and entity lineage. See
[tools](./tools.md).

## Configuration

- `MCPServerSpec(command, args=[], env={}, cwd=None)` — `env` overlays the parent
  environment.
- `MCPClient.connect(..., client_name="himmy", client_version="0.1.0", protocol_version=DEFAULT_PROTOCOL_VERSION, request_timeout=30.0)`.
- `register_mcp_tools(..., prefix=, requires_approval=, names=)`.

At the agent-spec layer, MCP servers are declared via `mcp_servers` on the
`AgentSpec` (`himmy/config/agent_spec.py`) and wired during runtime build.

## Extension points

- **Add a server:** construct an `MCPServerSpec`, `connect`, then
  `register_mcp_tools` into the agent's `ToolRegistry`.
- **Sandbox a server:** the client does **not** sandbox the server. Wrap one in the
  sandbox kernel / an OS isolate if isolation is required.
- **Ship a Himmy MCP server:** `himmy/connectors/news_mcp_server.py` is a worked
  example of the *server* side of this protocol — see [connectors](./connectors.md).

## Gotchas & invariants

- **Servers are trusted, not sandboxed.** They inherit the parent env and run with
  whatever privileges the host process has. Only configure servers you trust.
- **Server-initiated requests are unhandled.** The reader drops any inbound message
  without a matching pending request id (notifications and server→client requests
  alike).
- **One reader task, demultiplex by integer id.** Concurrent `request`s are safe;
  ids are client-assigned and monotonically increasing.
- **`aclose` is idempotent** and best-effort: it fails pending futures, terminates,
  and force-kills if needed.
- **Tool errors are in-band.** A failing MCP tool yields a successful
  `MCPToolResult` with `is_error=True`, not an exception — the bridged handler
  surfaces that as `is_error` in its result payload.
- **No SDK dependency.** The whole client is stdlib + asyncio; everything is
  offline-testable against a mock stdio server.

## Related docs

- [tools](./tools.md) — the pipeline every bridged MCP tool flows through.
- [toolkit](../architecture/toolkit.md) — built-in tool packs and how
  `AgentSpec.tool_packs` / `mcp_servers` resolve.
- [connectors](./connectors.md) — the Himmy news MCP *server* implementation.
