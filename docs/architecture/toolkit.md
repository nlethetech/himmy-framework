# Toolkit (tool packs)

> Named bundles of built-in tools — `tool_packs: [web, files]` in `agent.yaml` wires a working toolset onto a `ToolRegistry` with no driver code. Offline- and keyless-by-default.

## Overview

`himmy/toolkit/` is the catalog of ready-made tools an agent can be given by name.
A `ToolPack` couples a registrar function with the names of the tools it provides,
so a spec can list pack names and `register_packs` wires them all in one call.

Design principles visible across the packs:

- **Offline/keyless defaults.** Web search defaults to a keyless DuckDuckGo HTML
  scrape; data sources use keyless public APIs; the news pack honors a fixture file.
- **Injectable network seams.** I/O goes through `Fetcher` / `HttpCaller` protocols
  so every pack is testable without the network.
- **SSRF safety for model-supplied URLs.** `web_fetch` / `http_request` run every
  URL through `_net.guard_url` (no private hosts, http/https only, no embedded
  credentials, DNS-resolution checked).
- **Approval-gating of sensitive ("act") tools.** Send-email/webhook/telegram and
  the code sandbox are approval-gated by default.

The base tool machinery (registry, execution, validation, security) lives in
`himmy/services/tools/` — see [tools](../services/tools.md).

## Module map

| File | Responsibility |
| --- | --- |
| `pack.py` | The pack abstraction: `ToolPack` dataclass, `BUILTIN_PACKS` catalog, `resolve_packs`, `register_packs`, `UnknownToolPackError`. |
| `config.py` | `ToolkitConfig` — non-secret settings for every pack (fs root, search backend, DSNs, sandbox envelope, network knobs); `from_env` / `from_sources`. |
| `_net.py` | `guard_url` — SSRF guard for tools that fetch a **model-supplied** URL (vs the host-pinned check in `services/tools/security.py`). |
| `web.py` | `web` pack: `web_search`, `web_fetch`, `http_request`; pluggable search backends (DuckDuckGo/Tavily/Brave). |
| `files.py` | `files` pack: `read_file`, `write_file`, `list_dir` under a sandboxed root. |
| `data.py` | `data` pack: `sql_query`, `sql_schema` (read-only SQLite/Postgres). |
| `code.py` | `code` pack: `run_python` in a resource-limited sandbox (approval-gated). |
| `utils.py` | `utils` pack: `calculator`, `current_time`. |
| `knowledge.py` | `knowledge` pack: `kb_ingest`, `kb_search` (RAG over the agent's own KB). |
| `documents.py` | `documents` pack: `read_document` (PDF/text/MD/CSV/Excel extraction). |
| `comms.py` | `comms` pack: `send_email`, `send_webhook` (approval-gated). |
| `datasources.py` | `data-sources` pack: `weather`, `geocode`, `wikipedia` (keyless public APIs). |
| `memory.py` | `memory` pack: `remember`, `recall` (+ optional `consolidate`). |
| `notes.py` | `notes` pack: `list_notes`, `read_note`, `write_note`. |
| `tasks_pack.py` | `tasks` pack: `list_tasks`, `add_task`, `complete_task`. |
| `google_pack.py` | `google` pack: `gmail_inbox`, `gmail_send`, `gcal_events`, `gcal_create`. |
| `nepal.py` | `nepal` pack: `nepali_date`, `nepali_format`, `nepali_transliterate` + NRB tools. |
| `agentic.py` | `agentic` pack: `ask_human`, scratchpad, and todo tools. |
| `telegram.py` | `telegram` pack: `send_telegram` (approval-gated). |
| `spawn.py` | `spawn_agent` sub-agent tool (wired specially, not an ordinary pack). |

The news pack's registrar is defined *inside* `pack.py` (`_register_news_pack`,
lazily importing `himmy.connectors.news_tools`) so the `connectors` package stays
off the import path — see [connectors](../services/connectors.md).

## Key abstractions

### `ToolPack` (`pack.py`)

```python
@dataclass(frozen=True)
class ToolPack:
    name: str
    description: str
    tool_names: tuple[str, ...]
    register: Callable[[ToolRegistry, ToolkitConfig], None]
```

A named bundle plus its registrar. `BUILTIN_PACKS` is the single source-of-truth
dict (keyed by pack name) listed by the `himmy tools` CLI.

### `ToolkitConfig` (`config.py`)

The non-secret settings every pack reads — filesystem jail root, search backend
choice, DB DSNs, sandbox envelope, embedder config, comms/telegram settings, and
network safety knobs (`allow_private_hosts`, `egress_allow_hosts`, `http_timeout`,
`http_max_bytes`). `from_env()` reads `HIMMY_*` env vars; `from_sources()` overlays
`himmy.toml`'s `[toolkit]` with the env (precedence: **env > toml > default**).
Secrets (search API keys, DB/SMTP passwords) are read from the environment at call
time and never persisted on the model or projected onto entities.

### `guard_url` (`_net.py`)

The SSRF guard for tools that fetch a **model-supplied** URL. Rejects non-http(s)
schemes, embedded userinfo credentials, and a missing host; resolves the host via
DNS and rejects any address that is loopback/private/link-local/reserved/multicast/
unspecified (unless `allow_private`). An optional `allow_hosts` egress allow-list
narrows further (exact or subdomain match). This complements — and is distinct
from — the *host-pinned* check in `services/tools/security.py`, which protects
declarative HTTP tools whose base host is fixed by config.

## How it works / data flow

### Registration flow

```python
register_packs(registry, ["web", "files"], ToolkitConfig.from_env())
```

`register_packs` → `resolve_packs(names)` (raises `UnknownToolPackError` on an
unknown name) → for each pack, `pack.register(registry, cfg)`. Each registrar calls
`register_local_tool` (or `register_http_tool`) on the registry; the tools are then
dispatchable by `ToolService`.

### How `AgentSpec.tool_packs` resolves

In `himmy/runtime/from_spec.py`, a `ToolRegistry` is built when the spec declares
any of `tool_packs`, `tools_module`, `http_tools`, `knowledge`, `mcp_servers`,
`allow_spawn`, or `allow_skill_dispatch`. Then:

- `spec.tool_packs` → `register_packs(registry, spec.tool_packs, ToolkitConfig.from_sources(...))`.
- `spec.knowledge` auto-registers the `knowledge` pack (if not already listed) and
  ingests the declared docs.
- `spec.http_tools` → `register_http_tools` (declarative REST; see below).
- `spec.tools_module` → a user-supplied registrar function.
- `spec.allow_spawn` → `register_spawn_tool` (sub-agent; see below).
- `spec.allow_skill_dispatch` → `register_skill_dispatch_tool`.

Skill bundles can also contribute `tool_packs` (merged, de-duplicated) — see
`agent_spec.py`'s skill resolution.

### Built-in pack catalog

| Pack | Tools | Description |
| --- | --- | --- |
| `web` | `web_search`, `web_fetch`, `http_request` | Reach the open web: search, fetch a page's readable text, or make an arbitrary HTTP request. SSRF-guarded; redirects not followed. |
| `files` | `read_file`, `write_file`, `list_dir` | Read/write/list files under a sandboxed root. `write_file` is approval-gated unless `fs_allow_write`. |
| `data` | `sql_query`, `sql_schema` | Run read-only SQL against SQLite or Postgres; inspect tables/columns. |
| `code` | `run_python` | Execute Python in a resource-limited sandbox. **Approval-gated.** |
| `utils` | `calculator`, `current_time` | A safe arithmetic calculator and the current time (timezone-aware). |
| `knowledge` | `kb_ingest`, `kb_search` | Build and semantically search the agent's own knowledge base (RAG). |
| `documents` | `read_document` | Extract text from PDF/text/Markdown/CSV/Excel files under the root. |
| `comms` | `send_email`, `send_webhook` | Reach the outside world: send email (SMTP) or POST JSON to a webhook. **Approval-gated** unless `comms_allow_send`. |
| `data-sources` | `weather`, `geocode`, `wikipedia` | Keyless public data: Open-Meteo weather, OSM Nominatim geocoding, Wikipedia summaries. |
| `news` | `news_sources`, `news_search`, `news_fetch` | Read RSS/Atom news feeds: list sources, fetch a feed, keyword-search headlines. Honors `HIMMY_NEWS_FIXTURE` for offline use. |
| `memory` | `remember`, `recall` | Durable long-term memory: store facts and recall them semantically. (Adds `consolidate` when `memory_consolidate` is set.) |
| `notes` | `list_notes`, `read_note`, `write_note` | Shared markdown notes (same store as the GUI). |
| `tasks` | `list_tasks`, `add_task`, `complete_task` | Shared task board (same store as the GUI). |
| `google` | `gmail_inbox`, `gmail_send`, `gcal_events`, `gcal_create` | Gmail + Google Calendar for the connected account: read inbox, send mail, list/create events. |
| `nepal` | `nepali_date`, `nepali_format`, `nepali_transliterate`, `nrb_forex`, `nrb_macro_reports`, `nrb_macro_workbook` | Bikram Sambat dates, NPR/Devanagari formatting, transliteration, and NRB forex/macro data. See [connectors](../services/connectors.md). |
| `agentic` | `ask_human`, `scratchpad_set`, `scratchpad_get`, `todo_write`, `todo_complete`, `todo_read` | Be agentic: ask the human, keep a scratchpad, manage a todo list. |
| `telegram` | `send_telegram` | Send a message to a Telegram chat via a bot. **Approval-gated** unless `comms_allow_send`. |

### Read/write intent on tools

Most pack tools set `read_only=True/False` explicitly (e.g. `web_search`,
`weather`, `recall` are read-only; `write_file`, `send_email`, `remember` are
writes). `http_request` deliberately sets no `read_only` claim (a generic
GET/POST/... escape hatch is method-dependent). The flag is surfaced to the model
so look-ups don't land on a write tool — see `access.py` in [tools](../services/tools.md).

### Declarative REST tools (`http_tools`)

Defined not in `toolkit/` but in `himmy/config/http_tool_spec.py`. `HttpToolSpec`
is the YAML-shaped façade over `HttpToolConfig`:

```yaml
http_tools:
  - name: get_weather
    description: Current weather for a city.
    base_url: https://api.example.com
    path: /weather/{city}
    query: [units]
    auth: { type: bearer, env_var: WEATHER_API_KEY }
```

`HttpToolSpec` derives the args JSON Schema automatically from the path
placeholders + query/body/header names (path args required), then
`register_http_tools(registry, specs)` registers each via `register_http_tool` — no
Python. Execution flows through `ToolService`'s host-pinned, SSRF-hardened HTTP
backend (see [tools](../services/tools.md)). An explicit `args_schema` overrides the
derived one.

### Sub-agents (`spawn.py`)

`register_spawn_tool(registry, *, inference=...)` adds `spawn_agent`, turning any
agent into an ad-hoc orchestrator: it hands a sub-task to a fresh single-agent (its
own persona, optionally its own `tool_packs`) that runs to completion and returns
its answer. The sub-agent shares the parent's `InferenceService` but runs in a
*fresh* runtime whose registry has **no `spawn_agent`** — that one-level cap is the
recursion guard. It's wired where the inference backend is known
(`allow_spawn: true` on the spec), not as an ordinary pack, because a pack
registrar only receives a `ToolkitConfig`. Unknown requested packs are dropped and
reported rather than failing the whole spawn.

## Configuration

Key `ToolkitConfig` / `HIMMY_*` knobs (see `config.py` for the full list):

| Concern | Field / env |
| --- | --- |
| Filesystem jail | `fs_root` / `HIMMY_FS_ROOT`, `fs_allow_write` / `HIMMY_FS_ALLOW_WRITE` |
| Web search | `search_backend` (`duckduckgo`/`tavily`/`brave`), `search_api_key` (`HIMMY_SEARCH_API_KEY` or `TAVILY_API_KEY`/`BRAVE_API_KEY`) |
| Network safety | `http_timeout`, `http_max_bytes`, `allow_private_hosts`, `egress_allow_hosts` (`HIMMY_EGRESS_ALLOW`) |
| SQL | `sqlite_path`, `sql_dsn`, `sql_read_only` |
| Knowledge/memory | `kb_dsn`, `embedder`, `memory_path`, `memory_consolidate`, `memory_min_similarity` |
| Comms | `comms_allow_send`, `smtp_*` |
| Telegram | `telegram_bot_token`, `telegram_default_chat_id` |
| Code sandbox | `code_exec` (`off`/`subprocess`/`container`), `sandbox_limits`, `sandbox_image`, `sandbox_engine` |
| News (offline) | `HIMMY_NEWS_FIXTURE` (a local RSS file) |

## Extension points

- **Add a pack:** write a `register_<name>_pack(registry, config)` registrar, then
  add a `ToolPack` entry to `BUILTIN_PACKS`.
- **Add a one-off REST tool with no Python:** use `http_tools` in `agent.yaml`
  (`HttpToolSpec`).
- **Add custom Python tools:** `tools_module` pointing at a registrar.
- **Bridge an MCP server:** `mcp_servers` — see [mcp](../services/mcp.md).
- **Swap a network seam:** most registrars accept an injectable `Fetcher` /
  `HttpCaller` / `SearchBackend` for tests.

## Gotchas & invariants

- **`guard_url` (model-supplied URLs) ≠ host pinning (declarative tools).** The web
  pack uses the former; declarative `http_tools` use the latter. Don't conflate
  them.
- **Sensitive tools are approval-gated by default:** `code`'s `run_python`,
  `comms` (unless `comms_allow_send`), `telegram` (unless `comms_allow_send`), and
  `files`' `write_file` (unless `fs_allow_write`).
- **`spawn_agent` is one-level only** — a spawned worker can't spawn again.
- **Keyless by default:** DuckDuckGo search, Open-Meteo/Nominatim/Wikipedia, and
  fixture-backed news all work with no API keys.
- **The news registrar lives in `pack.py`**, not in a separate pack module, to keep
  `himmy.connectors` off the import path until needed.
- **`spawn` and `http_tools` are NOT in `BUILTIN_PACKS`** — they're wired by the
  runtime/spec layer, not via `tool_packs`.

## Related docs

- [tools](../services/tools.md) — the execution kernel every pack tool runs on.
- [mcp](../services/mcp.md) — MCP servers as a tool source.
- [connectors](../services/connectors.md) — domain connectors behind the `news` and
  `nepal` packs.
