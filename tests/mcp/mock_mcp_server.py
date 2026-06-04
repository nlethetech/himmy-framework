"""A minimal in-repo MCP stdio server for offline tests (NOT a real server).

Speaks just enough of the protocol to exercise ``MCPClient``: the ``initialize``
handshake, ``tools/list``, and ``tools/call`` for a couple of deterministic tools,
plus an error path. Newline-delimited JSON-RPC 2.0 on stdin/stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo the given text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
]


def _send(message: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _result(message_id: Any, result: dict[str, Any]) -> None:
    _send({"jsonrpc": "2.0", "id": message_id, "result": result})


def _error(message_id: Any, code: int, message: str) -> None:
    _send(
        {
            "jsonrpc": "2.0",
            "id": message_id,
            "error": {"code": code, "message": message},
        }
    )


def _call(message_id: Any, params: dict[str, Any]) -> None:
    name = params.get("name")
    args = params.get("arguments") or {}
    if name == "echo":
        _result(
            message_id,
            {
                "content": [{"type": "text", "text": str(args.get("text", ""))}],
                "isError": False,
            },
        )
    elif name == "add":
        total = (args.get("a", 0) or 0) + (args.get("b", 0) or 0)
        _result(
            message_id,
            {"content": [{"type": "text", "text": str(total)}], "isError": False},
        )
    elif name == "explode":
        _error(message_id, -32000, "boom")
    else:
        _result(
            message_id,
            {
                "content": [{"type": "text", "text": f"unknown tool {name}"}],
                "isError": True,
            },
        )


def main() -> None:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = message.get("method")
        message_id = message.get("id")
        if method == "initialize":
            _result(
                message_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "mock-mcp", "version": "0.0.1"},
                },
            )
        elif method == "notifications/initialized":
            continue  # a notification — no response
        elif method == "tools/list":
            _result(message_id, {"tools": TOOLS})
        elif method == "tools/call":
            _call(message_id, message.get("params") or {})
        elif message_id is not None:
            _error(message_id, -32601, f"method not found: {method}")


if __name__ == "__main__":
    main()
