"""API kernel: Studio MCP router — the connector-ecosystem manager.

Mounted under ``/api/studio/mcp`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). Three surfaces:

* a persisted **server registry** (``.himmy/mcp_servers.json``) — CRUD over
  named stdio MCP server definitions. Env entries are secret **names only**;
  values are resolved through the secrets backend at test/attach time and are
  never persisted in the registry nor echoed in any response.
* **test** — launch the server with a short timeout, run the MCP handshake,
  and return its tool list (name + description). Listing only: ``tools/call``
  is never issued from this router. The result is cached on the registry
  entry so ``GET /servers/{name}/tools`` works offline.
* **attach** — round-trip an ``agent.yaml`` through the existing
  loader/validator to add/remove the server in ``spec.mcp_servers``. The file
  is re-loaded after the write and restored byte-for-byte if the loader
  rejects it, so a spec can never be corrupted by this endpoint.

Subprocesses are spawned by :class:`~himmy.services.mcp.client.MCPClient`
(``create_subprocess_exec`` — argv, never a shell). MCP servers are
operator-configured trusted processes; see the client module for the
sandboxing caveat.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import Depends, HTTPException, Response
from pydantic import BaseModel, Field, field_validator

from himmy.api.routers.studio_common import build_studio_router, studio_permission
from himmy.api.routers.studio_notify import record_notification
from himmy.api.studio_service import list_agents, project_root, resolve_spec_path
from himmy.config.mcp_spec import MCPServerConfig
from himmy.config.secrets import get_secret
from himmy.core.ids import utc_now_iso

router = build_studio_router("mcp", tag="studio-mcp")

#: MCP server registration mutates which external tool servers an agent can reach (an
#: RCE-adjacent surface) — write/registration routes require the dedicated
#: ``studio.mcp:manage`` permission (admin-only by default), additively on top of the
#: router's ``studio.mcp:read`` baseline.
_mcp_manage = Depends(studio_permission("studio.mcp", "manage"))


# ---- bounds ----------------------------------------------------------------

_MAX_SERVERS = 100
_MAX_CACHED_TOOLS = 200
_MAX_ERROR_LEN = 500
_MAX_DESCRIPTION_LEN = 2000
#: Per-request MCP timeout for a test connect (handshake + tools/list each get
#: this budget; the whole test is additionally capped at twice this value).
_TEST_TIMEOUT_S = 10.0

_LOCK = threading.Lock()


# ---- registry models --------------------------------------------------------


class MCPToolInfo(BaseModel):
    """One tool advertised by a server (cached from the last successful test)."""

    name: str = Field(max_length=256)
    description: str = Field(default="", max_length=_MAX_DESCRIPTION_LEN)


class TestStatus(BaseModel):
    """The outcome of the most recent ``POST /servers/{name}/test``."""

    ok: bool
    at: str  # ISO timestamp
    error: str | None = Field(default=None, max_length=_MAX_ERROR_LEN)


class ServerIn(BaseModel):
    """Create/update payload for one registry entry.

    ``env_keys`` holds secret NAMES only — the values live in the secrets
    backend (Connections / env / keychain) and are resolved when the server is
    launched, never stored here.
    """

    name: str = Field(min_length=1, max_length=64)
    transport: str = Field(default="stdio", max_length=16)
    command: str = Field(min_length=1, max_length=256)
    args: list[str] = Field(default_factory=list, max_length=32)
    env_keys: list[str] = Field(default_factory=list, max_length=32)
    cwd: str | None = Field(default=None, max_length=512)
    prefix: str = Field(default="", max_length=32)
    requires_approval: bool = False
    enabled: bool = True

    @field_validator("name")
    @classmethod
    def _name_slug(cls, v: str) -> str:
        ok = v[:1].isalnum() and all(c.isalnum() or c in "_-." for c in v)
        if not ok:
            raise ValueError(
                "name must start with a letter/digit and use only "
                "letters, digits, '_', '-' or '.'"
            )
        return v

    @field_validator("transport")
    @classmethod
    def _stdio_only(cls, v: str) -> str:
        if v != "stdio":
            raise ValueError(
                "only the 'stdio' transport is supported by the himmy runtime"
            )
        return v

    @field_validator("args")
    @classmethod
    def _args_bounded(cls, v: list[str]) -> list[str]:
        for a in v:
            if len(a) > 512:
                raise ValueError("each argument must be at most 512 characters")
        return v

    @field_validator("env_keys")
    @classmethod
    def _env_names(cls, v: list[str]) -> list[str]:
        for k in v:
            valid = (
                0 < len(k) <= 128
                and (k[0].isalpha() or k[0] == "_")
                and all(c.isalnum() or c == "_" for c in k)
            )
            if not valid:
                raise ValueError(
                    f"env key {k!r} is not a valid environment variable NAME "
                    "(values belong in the secrets backend, not here)"
                )
        return v

    @field_validator("prefix")
    @classmethod
    def _prefix_chars(cls, v: str) -> str:
        if v and not all(c.isalnum() or c == "_" for c in v):
            raise ValueError("prefix may only contain letters, digits and '_'")
        return v


class ServerOut(ServerIn):
    """A registry entry as returned to the GUI (plus cached test state)."""

    created_at: str = ""
    updated_at: str = ""
    last_test: TestStatus | None = None
    tools: list[MCPToolInfo] = Field(default_factory=list)  # from last OK test
    tools_at: str | None = None


class TestResult(BaseModel):
    """What ``POST /servers/{name}/test`` returns: a verdict + the tool list."""

    ok: bool
    at: str
    error: str | None = None
    tools: list[MCPToolInfo] = Field(default_factory=list)
    server_info: dict[str, Any] = Field(default_factory=dict)


class AttachRequest(BaseModel):
    """Attach/detach one registered server to/from one agent spec file."""

    server: str = Field(min_length=1, max_length=64)
    agent_path: str = Field(min_length=1, max_length=512)
    attach: bool = True


class AttachResult(BaseModel):
    server: str
    agent_path: str
    attached: bool


class AgentAttachment(BaseModel):
    """One discovered agent + whether it carries this server in its spec."""

    path: str
    name: str
    attached: bool


# ---- persistence ------------------------------------------------------------


def _store_path() -> Path:
    d = project_root() / ".himmy"
    d.mkdir(exist_ok=True)
    return d / "mcp_servers.json"


def _load_servers() -> list[dict[str, Any]]:
    path = _store_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"MCP registry file {path.name} is unreadable: {exc}",
        ) from exc
    servers = data.get("servers") if isinstance(data, dict) else None
    if not isinstance(servers, list):
        return []
    return [s for s in servers if isinstance(s, dict)]


def _save_servers(servers: list[dict[str, Any]]) -> None:
    """Atomic write (tmp + replace) so a crash never truncates the registry."""
    path = _store_path()
    payload = json.dumps({"servers": servers}, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _find(servers: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for s in servers:
        if s.get("name") == name:
            return s
    return None


def _must_find(servers: list[dict[str, Any]], name: str) -> dict[str, Any]:
    server = _find(servers, name)
    if server is None:
        raise HTTPException(status_code=404, detail=f"unknown MCP server: {name!r}")
    return server


def _to_out(record: dict[str, Any]) -> ServerOut:
    """Project a raw store record into the response model (drops unknown keys)."""
    return ServerOut.model_validate(
        {k: record.get(k) for k in ServerOut.model_fields if k in record}
    )


# ---- secrets ----------------------------------------------------------------


def _resolve_env(env_keys: list[str]) -> tuple[dict[str, str], list[str]]:
    """Resolve each env NAME through the secrets backend; report missing names.

    The returned values are handed to the subprocess environment only — they
    must never appear in a response body or a log line.
    """
    values: dict[str, str] = {}
    missing: list[str] = []
    for key in env_keys:
        value = get_secret(key)
        if value is None:
            missing.append(key)
        else:
            values[key] = value
    return values, missing


# ---- CRUD --------------------------------------------------------------------


@router.get("/servers")
async def list_servers() -> dict[str, list[ServerOut]]:
    """Every registered MCP server, with cached test state."""
    with _LOCK:
        servers = _load_servers()
    return {"servers": [_to_out(s) for s in servers]}


@router.post("/servers", status_code=201, dependencies=[_mcp_manage])
async def create_server(body: ServerIn) -> ServerOut:
    """Register a new server definition (409 on a name collision)."""
    with _LOCK:
        servers = _load_servers()
        if _find(servers, body.name) is not None:
            raise HTTPException(
                status_code=409, detail=f"MCP server {body.name!r} already exists"
            )
        if len(servers) >= _MAX_SERVERS:
            raise HTTPException(
                status_code=400,
                detail=f"registry is full ({_MAX_SERVERS} servers max)",
            )
        now = utc_now_iso()
        record: dict[str, Any] = {
            **body.model_dump(),
            "created_at": now,
            "updated_at": now,
            "last_test": None,
            "tools": [],
            "tools_at": None,
        }
        servers.append(record)
        _save_servers(servers)
    return _to_out(record)


@router.put("/servers/{name}", dependencies=[_mcp_manage])
async def update_server(name: str, body: ServerIn) -> ServerOut:
    """Update (or rename) a server. A changed launch config invalidates the
    cached tool list — the old test result no longer describes this server."""
    with _LOCK:
        servers = _load_servers()
        record = _must_find(servers, name)
        if body.name != name and _find(servers, body.name) is not None:
            raise HTTPException(
                status_code=409, detail=f"MCP server {body.name!r} already exists"
            )
        launch_keys = ("command", "args", "env_keys", "cwd")
        launch_changed = any(record.get(k) != getattr(body, k) for k in launch_keys)
        record.update(body.model_dump())
        record["updated_at"] = utc_now_iso()
        if launch_changed:
            record["last_test"] = None
            record["tools"] = []
            record["tools_at"] = None
        _save_servers(servers)
    return _to_out(record)


@router.delete("/servers/{name}", status_code=204, dependencies=[_mcp_manage])
async def delete_server(name: str) -> Response:
    """Remove a server from the registry (specs that embed it are untouched)."""
    with _LOCK:
        servers = _load_servers()
        _must_find(servers, name)
        _save_servers([s for s in servers if s.get("name") != name])
    return Response(status_code=204)


# ---- test (listing only — never tools/call) ----------------------------------


async def _probe(record: dict[str, Any], env: dict[str, str]) -> TestResult:
    """Connect, list tools, disconnect. ``tools/call`` is never issued here."""
    from himmy.services.mcp.client import MCPClient
    from himmy.services.mcp.models import MCPServerSpec

    spec = MCPServerSpec(
        command=str(record.get("command", "")),
        args=[str(a) for a in record.get("args") or []],
        env=env,
        cwd=record.get("cwd") or None,
    )
    client = await MCPClient.connect(spec, request_timeout=_TEST_TIMEOUT_S)
    try:
        tools = await client.list_tools()
        return TestResult(
            ok=True,
            at=utc_now_iso(),
            tools=[
                MCPToolInfo(
                    name=t.name[:256],
                    description=(t.description or "")[:_MAX_DESCRIPTION_LEN],
                )
                for t in tools[:_MAX_CACHED_TOOLS]
            ],
            server_info={
                k: v
                for k, v in (client.server_info or {}).items()
                if k in ("name", "version")
            },
        )
    finally:
        await client.aclose()


@router.post("/servers/{name}/test", dependencies=[_mcp_manage])
async def test_server(name: str) -> TestResult:
    """Launch the server briefly and return its tool list (or a clear failure).

    Always answers 200 with a verdict — a server that fails to start is an
    expected, reportable outcome, not an API error. The verdict (and on
    success the tool list) is cached on the registry entry.

    Requires ``studio.mcp:manage`` (NOT the router's ``:read`` baseline): probing a
    server SPAWNS its subprocess with admin-provisioned secrets resolved into the env,
    and MUTATES persisted registry state (``last_test``/``tools``/``tools_at``) — both
    privileged side effects (subprocess execution + a write), not a read. Leaving it at
    ``:read`` let a low-privilege console role (viewer/operator/auditor) execute an
    admin-registered server on demand and poison its cached state.
    """
    with _LOCK:
        record = dict(_must_find(_load_servers(), name))

    env, missing = _resolve_env([str(k) for k in record.get("env_keys") or []])
    if missing:
        result = TestResult(
            ok=False,
            at=utc_now_iso(),
            error=(
                "missing secrets: "
                + ", ".join(missing)
                + " — add them in Connections or the environment"
            )[:_MAX_ERROR_LEN],
        )
    else:
        try:
            result = await asyncio.wait_for(
                _probe(record, env), timeout=_TEST_TIMEOUT_S * 2
            )
        except TimeoutError:
            result = TestResult(
                ok=False,
                at=utc_now_iso(),
                error=f"test timed out after {int(_TEST_TIMEOUT_S * 2)}s",
            )
        except Exception as exc:  # noqa: BLE001 - verdict, not a 500
            result = TestResult(
                ok=False, at=utc_now_iso(), error=str(exc)[:_MAX_ERROR_LEN]
            )

    with _LOCK:
        servers = _load_servers()
        live = _find(servers, name)
        if live is not None:  # may have been deleted mid-test
            live["last_test"] = TestStatus(
                ok=result.ok, at=result.at, error=result.error
            ).model_dump()
            if result.ok:
                live["tools"] = [t.model_dump() for t in result.tools]
                live["tools_at"] = result.at
            _save_servers(servers)

    if result.ok:
        record_notification(
            "mcp",
            f"MCP server '{name}' connected",
            body=f"{len(result.tools)} tools discovered",
            link="/mcp",
        )
    else:
        record_notification(
            "mcp",
            f"MCP server '{name}' test failed",
            body=result.error or "",
            link="/mcp",
        )
    return result


@router.get("/servers/{name}/tools")
async def server_tools(name: str) -> dict[str, Any]:
    """The tool list cached from the last successful test (empty if untested)."""
    with _LOCK:
        record = _must_find(_load_servers(), name)
    return {
        "tools": [
            MCPToolInfo.model_validate(t)
            for t in record.get("tools") or []
            if isinstance(t, dict)
        ],
        "tested_at": record.get("tools_at"),
    }


# ---- agent attachment ---------------------------------------------------------


def _entry_matches(entry: Any, record: dict[str, Any]) -> bool:
    """Does one ``spec.mcp_servers`` entry correspond to this registry server?

    Identity is structural — command + args + prefix — because the spec format
    (:class:`MCPServerConfig`) has no name field of its own.
    """
    if not isinstance(entry, dict):
        return False
    return (
        entry.get("command") == record.get("command")
        and [str(a) for a in entry.get("args") or []]
        == [str(a) for a in record.get("args") or []]
        and (entry.get("prefix") or "") == (record.get("prefix") or "")
    )


def _spec_entry(record: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """The minimal ``mcp_servers`` item to write into an agent spec."""
    entry: dict[str, Any] = {"command": record["command"]}
    if record.get("args"):
        entry["args"] = [str(a) for a in record["args"]]
    if env:
        entry["env"] = env
    if record.get("cwd"):
        entry["cwd"] = record["cwd"]
    if record.get("prefix"):
        entry["prefix"] = record["prefix"]
    if record.get("requires_approval"):
        entry["requires_approval"] = True
    return entry


def _load_raw_spec(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise HTTPException(
            status_code=400, detail=f"agent file is not valid YAML: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="agent file is not a YAML mapping")
    if raw.get("entry") and isinstance(raw.get("members"), list):
        raise HTTPException(
            status_code=400,
            detail="that file is a team spec — attach MCP servers to single agents",
        )
    return raw


@router.post("/attach", dependencies=[_mcp_manage])
async def attach_server(body: AttachRequest) -> AttachResult:
    """Add/remove one registered server in an agent spec's ``mcp_servers``.

    The write is validated twice: each proposed entry against
    :class:`MCPServerConfig`, and the whole file by re-running the real spec
    loader afterwards — on any failure the original bytes are restored, so the
    spec on disk is never corrupted. Env secret values are resolved here (at
    attach time) and written into the spec entry the runtime launches with.
    """
    from himmy.config.agent_spec import load_agent_spec

    with _LOCK:
        record = dict(_must_find(_load_servers(), body.server))

    if body.attach and not record.get("enabled", True):
        raise HTTPException(
            status_code=409,
            detail=f"MCP server {body.server!r} is disabled — enable it first",
        )

    try:
        path = resolve_spec_path(body.agent_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw = _load_raw_spec(path)
    existing = raw.get("mcp_servers")
    entries = (
        [e for e in existing if isinstance(e, dict)]
        if isinstance(existing, list)
        else []
    )
    entries = [e for e in entries if not _entry_matches(e, record)]

    if body.attach:
        env, missing = _resolve_env([str(k) for k in record.get("env_keys") or []])
        if missing:
            raise HTTPException(
                status_code=400,
                detail=(
                    "cannot attach — missing secrets: "
                    + ", ".join(missing)
                    + " (add them in Connections or the environment)"
                ),
            )
        entries.append(_spec_entry(record, env))

    # Validate every proposed entry against the exact model the loader uses.
    try:
        for e in entries:
            MCPServerConfig.model_validate(e)
    except Exception as exc:  # noqa: BLE001 - surface pydantic messages
        raise HTTPException(
            status_code=400, detail=f"invalid mcp_servers entry: {exc}"
        ) from exc

    if entries:
        raw["mcp_servers"] = entries
    else:
        raw.pop("mcp_servers", None)

    original = path.read_text(encoding="utf-8")
    path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    try:
        load_agent_spec(str(path))  # the authoritative round-trip check
    except Exception as exc:  # noqa: BLE001 - restore + report, never corrupt
        path.write_text(original, encoding="utf-8")
        raise HTTPException(
            status_code=400,
            detail=f"refused to write a spec the loader rejects: {exc}",
        ) from exc

    record_notification(
        "mcp",
        f"MCP server '{body.server}' "
        + ("attached to" if body.attach else "detached from")
        + f" {body.agent_path}",
        link="/mcp",
    )
    return AttachResult(
        server=body.server, agent_path=body.agent_path, attached=body.attach
    )


@router.get("/servers/{name}/agents")
async def server_agents(name: str) -> dict[str, list[AgentAttachment]]:
    """Every discovered agent, flagged with whether this server is attached."""
    with _LOCK:
        record = dict(_must_find(_load_servers(), name))
    root = project_root().resolve()
    out: list[AgentAttachment] = []
    for summary in list_agents(root):
        if summary.error:
            continue
        try:
            raw = yaml.safe_load((root / summary.path).read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - unreadable file → just not attached
            raw = None
        entries = raw.get("mcp_servers") if isinstance(raw, dict) else None
        attached = isinstance(entries, list) and any(
            _entry_matches(e, record) for e in entries
        )
        out.append(
            AgentAttachment(path=summary.path, name=summary.name, attached=attached)
        )
    return {"agents": out}


__all__ = ["router"]
