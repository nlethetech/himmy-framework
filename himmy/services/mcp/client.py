"""MCP kernel: a transport-direct stdio client (no SDK, offline-testable).

Speaks the Model Context Protocol over stdio: newline-delimited JSON-RPC 2.0 to a
server subprocess. A single background reader task de-multiplexes responses by
request id, so concurrent calls and server-initiated notifications are handled
cleanly. Implemented against the wire format directly (not the ``mcp``/pydantic-ai
SDKs) so the core stays dependency-light and the whole path is exercisable offline
against a mock server.

Lifecycle::

    client = await MCPClient.connect(MCPServerSpec(command="my-mcp-server"))
    try:
        tools = await client.list_tools()
        result = await client.call_tool("search", {"q": "acme"})
    finally:
        await client.aclose()

MCP servers are operator-configured trusted processes; this client launches one
with the inherited environment (overlaid with ``spec.env``). It does NOT sandbox
the server — wrap one in the sandbox kernel / an OS isolate if that is required.
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from typing import Any

from himmy.core.errors import HimmyError
from himmy.services.mcp.models import MCPServerSpec, MCPTool, MCPToolResult

#: Default MCP protocol version advertised in ``initialize``.
DEFAULT_PROTOCOL_VERSION = "2024-11-05"
#: StreamReader line buffer (large tool results can exceed the 64KiB default).
_STREAM_LIMIT = 4 * 1024 * 1024


class MCPError(HimmyError):
    """A JSON-RPC error returned by an MCP server (carries code + data)."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(f"MCP error {code}: {message}")
        self.code = code
        self.data = data


class MCPClient:
    """A stdio JSON-RPC client for one MCP server subprocess."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        *,
        request_timeout: float = 30.0,
    ) -> None:
        """Wrap an already-spawned server process (use :meth:`connect`)."""
        self._proc = process
        self._request_timeout = request_timeout
        self._pending: dict[int, asyncio.Future[Any]] = {}
        self._next_id = 0
        self._closed = False
        self._reader: asyncio.Task[None] | None = None
        self.server_info: dict[str, Any] = {}
        self.protocol_version: str | None = None

    # --------------------------------------------------------------- lifecycle
    @classmethod
    async def connect(
        cls,
        spec: MCPServerSpec,
        *,
        client_name: str = "himmy",
        client_version: str = "0.1.0",
        protocol_version: str = DEFAULT_PROTOCOL_VERSION,
        request_timeout: float = 30.0,
    ) -> MCPClient:
        """Spawn the server, run the MCP handshake, and return a ready client."""
        try:
            proc = await asyncio.create_subprocess_exec(
                spec.command,
                *spec.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, **spec.env},
                cwd=spec.cwd,
                limit=_STREAM_LIMIT,
            )
        except Exception as exc:
            raise HimmyError(
                f"failed to launch MCP server {spec.command!r}: {exc}"
            ) from exc

        self = cls(proc, request_timeout=request_timeout)
        self._reader = asyncio.create_task(self._read_loop())
        try:
            init = await self.request(
                "initialize",
                {
                    "protocolVersion": protocol_version,
                    "capabilities": {},
                    "clientInfo": {"name": client_name, "version": client_version},
                },
            )
            self.server_info = init.get("serverInfo", {}) if init else {}
            self.protocol_version = (init or {}).get("protocolVersion")
            await self.notify("notifications/initialized")
        except Exception:
            await self.aclose()
            raise
        return self

    async def aclose(self) -> None:
        """Terminate the server and cancel the reader (idempotent)."""
        if self._closed:
            return
        self._closed = True
        self._fail_pending(HimmyError("MCP client closed"))
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._reader
        with suppress(Exception):
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        with suppress(ProcessLookupError, Exception):
            self._proc.terminate()
        with suppress(Exception):
            await asyncio.wait_for(self._proc.wait(), timeout=2.0)
        with suppress(ProcessLookupError, Exception):
            if self._proc.returncode is None:
                self._proc.kill()

    async def __aenter__(self) -> MCPClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # ------------------------------------------------------------------ rpc io
    async def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        """Send a JSON-RPC request and await its result (or raise on error)."""
        if self._closed:
            raise HimmyError("MCP client is closed.")
        self._next_id += 1
        request_id = self._next_id
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await self._write(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            }
        )
        try:
            return await asyncio.wait_for(future, timeout=self._request_timeout)
        except TimeoutError as exc:
            self._pending.pop(request_id, None)
            raise HimmyError(
                f"MCP request {method!r} timed out after {self._request_timeout}s"
            ) from exc

    async def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no id, no response expected)."""
        await self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    async def _write(self, message: dict[str, Any]) -> None:
        """Write one newline-delimited JSON-RPC message to the server's stdin."""
        if self._proc.stdin is None:  # pragma: no cover - defensive
            raise HimmyError("MCP server stdin is not available.")
        self._proc.stdin.write((json.dumps(message) + "\n").encode("utf-8"))
        await self._proc.stdin.drain()

    async def _read_loop(self) -> None:
        """Read server stdout lines, dispatching responses to their futures."""
        assert self._proc.stdout is not None
        while True:
            line = await self._proc.stdout.readline()
            if not line:  # EOF — the server exited or closed its stream
                self._fail_pending(HimmyError("MCP server closed the connection"))
                return
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
            except json.JSONDecodeError:
                continue  # ignore non-JSON-RPC noise on stdout
            message_id = message.get("id")
            if message_id is None or message_id not in self._pending:
                continue  # a notification or a server->client request (unhandled)
            future = self._pending.pop(message_id)
            if future.done():
                continue
            if "error" in message:
                err = message["error"] or {}
                future.set_exception(
                    MCPError(
                        int(err.get("code", -32000)),
                        str(err.get("message", "unknown error")),
                        err.get("data"),
                    )
                )
            else:
                future.set_result(message.get("result"))

    def _fail_pending(self, error: Exception) -> None:
        """Reject every in-flight request (server died / client closed)."""
        for future in self._pending.values():
            if not future.done():
                future.set_exception(error)
        self._pending.clear()

    # -------------------------------------------------------------- mcp methods
    async def list_tools(self) -> list[MCPTool]:
        """Return the tools the server advertises (``tools/list``)."""
        result = await self.request("tools/list", {})
        tools = (result or {}).get("tools", [])
        return [
            MCPTool(
                name=tool["name"],
                description=tool.get("description", "") or "",
                input_schema=tool.get("inputSchema") or {},
            )
            for tool in tools
        ]

    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> MCPToolResult:
        """Invoke a tool (``tools/call``) and normalize its content blocks."""
        result = await self.request(
            "tools/call", {"name": name, "arguments": arguments or {}}
        )
        result = result or {}
        content = result.get("content") or []
        text = "\n".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        return MCPToolResult(
            text=text,
            content=content,
            structured=result.get("structuredContent"),
            is_error=bool(result.get("isError", False)),
        )


__all__ = ["MCPClient", "MCPError", "DEFAULT_PROTOCOL_VERSION"]
