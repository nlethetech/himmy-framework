# CLI & Studio

> The two front doors to Himmy: a `himmy` command-line tool and Himmy Studio, a local web GUI served by the same FastAPI BFF.

## Overview

Himmy ships two human-facing surfaces, both offline-first:

- **The `himmy` CLI** — a console script (also `python -m himmy`) that wires the
  offline-first runtime so the common case needs no keys: `init`, `run`, `chat`,
  `team`, `eval`, `bench`, `doctor`, `tools`, `skills`, `prices`, `trace`, `serve`,
  `studio`, `telegram`.
- **Himmy Studio** — a Vite/React SPA backed by the FastAPI BFF (the same app
  `himmy serve` runs). It is the no-code front door: chat, a visual agent builder,
  run/trace browsing, an approvals inbox, and local "personal-app" surfaces (tasks,
  notes, calendar, memory, knowledge). Served on loopback by `himmy studio`.

Both go through the same spec→runtime wiring (`himmy/runtime/from_spec.py`), so a
Studio run is configured exactly like a `himmy run`.

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/cli/__main__.py` | `main()` entry point; argparse parser + subcommand dispatch. |
| `himmy/cli/commands.py` | `cmd_*` handlers (one per subcommand). |
| `himmy/cli/provider.py` | `build_inference_for` / `build_manager_for` — provider selection. |
| `himmy/api/app.py` | `create_app()` — the FastAPI BFF; mounts `/v1` + `/api/studio` + the built SPA. |
| `himmy/api/routers/studio.py` | The `/api/studio` router (GUI-shaped endpoints). |
| `himmy/api/studio_*.py` | Per-feature Studio backends (agents, approvals, tasks, notes, …). |
| `studio/` | The Vite/React frontend (built into `himmy/api/_studio_static`). |

## Key abstractions

### CLI subcommands (`__main__.py` + `commands.py`)

| Command | Handler | What it does |
| --- | --- | --- |
| `run` | `cmd_run` | One-shot: run a prompt, print the answer. Flags: `--stream`, `--trace`, `--plan`, `--json`, `--record FILE`, `--replay FILE`. |
| `chat` | `cmd_chat` | Interactive REPL on one thread (`/exit`, `/reset`, `/help`); `--message` for one turn; `--session` persists to `.himmy/sessions.db`. |
| `telegram` | `cmd_telegram` | Run an agent as a live Telegram bot (one thread per chat). |
| `team` | `cmd_team` | Run a multi-agent team from `team.yaml`; print the handoff route. |
| `eval` | `cmd_eval` | Evaluate an agent/team against a `suite.yaml`; print the scorecard. |
| `bench` | `cmd_bench` | Benchmark `provider:model` pairs on a task suite; `--fail-under` is a CI regression gate. |
| `init` | `cmd_init` | Scaffold an agent (`--template helpdesk/analyst/researcher`, or `--team`). |
| `serve` | `cmd_serve` | Boot the FastAPI BFF via uvicorn (needs the `api` extra). |
| `studio` | `cmd_studio` | Serve Himmy Studio (needs the `studio` extra). |
| `doctor` | `cmd_doctor` | Report installed extras, local providers on PATH, provider keys, guardrails. |
| `tools` | `cmd_tools` | List built-in tool packs and their tools. |
| `skills` | `cmd_skills` | List skills (built-in + project-local), or detail one. |
| `prices` | `cmd_prices` | Model price table: `sync` / `show <model>` / list. |
| `trace` | `cmd_trace` | Inspect saved run traces (list recent runs, or show one timeline). |

> Naming notes vs. the prompt's wording: the benchmark command is `bench` (not
> `benchmark`); there is **no** standalone `replay` subcommand — record/replay is the
> `--record` / `--replay` flag pair on `run`.

`main()` builds the parser, dispatches `args.func(args)`, and maps a raised
`HimmyError` to a clean `error: ...` line + exit code 1.

### Provider selection (`provider.py`)

`PROVIDERS = ("stub", "claude-cli", "ollama", "pydantic-ai", "openrouter")`.
`build_manager_for(provider, model)` returns a `ClientManager`:

- `None` (default) → the framework auto-selection
  (`himmy.runtime.builder.build_inference` — a real pydantic-ai manager when a
  provider key + the `providers` extra + a model are present, otherwise the offline
  `StubClientManager`).
- `stub` → `StubClientManager` (deterministic, offline, canned output).
- `claude-cli` → `ClaudeCliClientManager` (the local `claude` CLI — Claude Max;
  default model `haiku`).
- `ollama` → `OllamaClientManager` (a local Ollama server).
- `pydantic-ai` → `PydanticAIClientManager` (gateway/cloud; needs the `providers`
  extra, else a `ProviderError`).
- `openrouter` → `PydanticAIClientManager` pointed at `https://openrouter.ai/api/v1`
  (needs `OPENROUTER_API_KEY` + the `providers` extra).

`resolves_to_stub(...)` powers the CLI's one-line "you're offline, here's how to use a
real model" hint, printed only on an interactive TTY and not when the stub was chosen
explicitly (`HIMMY_NO_HINTS` suppresses it).

### Local trace DB (`.himmy/trace.db`)

`run --trace` attaches a `_TraceCollector` that persists run events to a SQLite event
store (`himmy.services.observability.trace.SqliteEventStore`) at `.himmy/trace.db`
(the `.himmy/` dir is created on demand). `himmy trace` reads it back:
`himmy trace` lists recent runs; `himmy trace <thread_id>` renders one run's timeline
via `format_timeline`. Other local SQLite stores under `.himmy/` include
`sessions.db` (chat `--session`), `approvals.db`, `tasks.db`, `chats.db`, etc.

### Studio (the local web GUI)

**Frontend** (`studio/`): a Vite + React 18 + react-router SPA
(`react-markdown`/`remark-gfm` for rendering). Screens map to the BFF endpoints —
Home, Chat, Chats, Builder, Tools, Teams, Activity, RunDetail, Lineage, Approvals,
Memory, Brain, Knowledge, Evaluation, Workflows, Compare, Models, Doctor,
Connections, Email, Calendar, Tasks, Notes, Cookbook, Research, Theme. `vite build`
emits the SPA **into the Python package** at `himmy/api/_studio_static`, so it ships
as package data; in dev, Vite runs on `:5173` and proxies `/api` + `/health` to the
BFF on `:8000` (one origin to the browser).

**BFF** (`himmy/api/app.py`): `create_app()` wires an `ApiContainer` onto
`app.state`, mounts the `/v1` routers (context, runs, recommendations, dashboard,
evaluation, audit) and the `/api/studio` router, then mounts the built SPA *last* via
a 404-fallback so a real API route always matches first (unknown API paths keep a
JSON 404; unknown non-API GETs serve `index.html` for client-side routing).
`studio_is_built()` checks for `index.html` in `_studio_static`.

**Agent auto-discovery** (`studio_service.py`): `list_agents()` / `list_teams()` scan
the project root (the directory `himmy studio` was launched in), skipping
`.git`/`node_modules`/`.venv`/`__pycache__`/`.himmy`. Each discovered `agent.yaml`
becomes an `AgentSummary` keyed by its project-relative path.

**The `studio_*` routers / backends** — the `/api/studio` surface (all in
`himmy/api/routers/studio.py`, delegating to the per-feature `studio_*` modules):

| Area | Endpoints (under `/api/studio`) | Backend module |
| --- | --- | --- |
| Runs | `POST /run` (SSE), `/run-team` (SSE), `/research` (SSE), `GET /runs`, `/runs/analytics`, `/runs/{id}`, `/runs/{id}/lineage` | `studio_service`, `studio_runs`, `studio_lineage` |
| Agent authoring | `GET /agents`, `/teams`, `/agent`, `/tools`, `/skills`; `POST /agents/validate`; `PUT /agents` | `studio_agents`, `studio_service` |
| Approvals | `GET /approvals`, `/approvals/{id}`; `POST /approvals/{id}/approve|reject` (SSE) | `studio_approvals` |
| Connections | `GET/PUT/DELETE /connections[/{ctype}]`, `/test`, `/send` | `studio_connections` |
| Google | `/google`, `/google/client`, `/google/auth-url`, `/google/callback`, `/google/gmail*`, `/google/calendar*` | `studio_google` |
| Models / compare | `GET /models`; `POST /compare` | `studio_bench`, `cli.provider` |
| Benchmarks | `GET /benchmarks`; `POST /benchmarks/probe` | `studio_bench` |
| Tasks | `GET/POST/PATCH/DELETE /tasks` | `studio_tasks` |
| Chats | `GET/POST/PATCH/DELETE /chats` | `studio_chats` |
| Cookbook | `GET/PUT/DELETE /cookbook` | `studio_cookbook` |
| Notes | `GET/PUT/DELETE /notes` | `studio_notes` |
| Calendar | `GET/POST/DELETE /calendar` | `studio_calendar` |
| Memory | `GET /memory`, `/memory/subjects`; `POST /memory`, `/memory/recall`; `DELETE /memory/{id}` | `studio_memory` |
| Knowledge | `GET/POST /knowledge`, `/{id}/ingest`, `/{id}/search`, `DELETE /{id}` | `studio_knowledge` |
| Evaluation | `GET /evals`; `POST /evals/run` | `studio_eval` |
| Workflows | `GET /workflows`; `POST /workflows/run` | `studio_workflows` |
| MCP servers | `GET/POST/PUT/DELETE /mcp/servers`, `/servers/{n}/test|tools|agents`, `POST /mcp/attach` | `routers/studio_mcp` |
| Guardrails (read-only) | `GET /guardrails` | `routers/studio_guardrails` |
| Security log (read-only) | `GET /seclog` | `routers/studio_seclog` |
| Privacy / governance | `GET /privacy/subjects|consents`, `POST /privacy/erase`, `/privacy/audit/export|verify` | `routers/studio_privacy` |
| Doctor / health | `GET /health`, `/doctor` | `runtime.diagnostics` |

Run/team/research/approval-resume endpoints stream **Server-Sent Events**
(`start`/`token`/`tool`/`message`/`done`, or a terminal `error`). Approvals are the
human-in-the-loop inbox: an approval-gated tool call pauses the run into an
`AgentCheckpoint` (durable at `.himmy/approvals.db`); the GUI lists pending
checkpoints, shows one with secret args redacted, and approve/reject resumes the run.

### MCP store model (CLI vs Studio — two stores, one runtime seam)

MCP server configuration is reachable from both front doors, and the two write to
**different** stores by design — there is no single shared MCP registry, and that is
intentional, not a gap:

| Surface | What it writes | Store |
| --- | --- | --- |
| CLI `himmy mcp add/remove` | the server directly into a spec's `mcp_servers` | `agent.yaml` (`mcp_cmd.py::_write_back`) |
| Studio **registry** (CRUD/test) | named stdio server *definitions* (env = secret **names** only) | `.himmy/mcp_servers.json` (`studio_mcp.py::_store_path`) |
| Studio **attach** | a registry server into a chosen spec's `mcp_servers` | `agent.yaml` (same loader round-trip as the CLI) |

The `.himmy/mcp_servers.json` registry is an **additive Studio-only project-global
staging catalog** with no CLI equivalent — it lets the GUI define/test a server once
and attach it to many agents. The **runtime seam both honour is `agent.yaml`'s
`mcp_servers`** (read by `from_spec.py`): a server *attached* via Studio appears in
`himmy mcp list` for that `agent.yaml`, and a server added via `himmy mcp` is visible
to a Studio attach view of the same file. So the catalog/spec split is: the catalog
stages, the spec binds — and the spec is the one shared truth the runtime launches
with. (On `/v1` the spec's `mcp_servers`/`tools_module`/`http_tools` are tenant-driven
subprocess/RCE/SSRF surfaces, so they are stripped from tenant-submitted specs unless
operator-provisioned — see the tenant-spec sanitizer.)

### Read-only governance viewers (data sources)

Two Studio surfaces mirror CLI commands and are pure reads over **existing** durable
stores — no new state of their own:

- **Guardrails viewer** (`GET /api/studio/guardrails`, `routers/studio_guardrails`) —
  the GUI sibling of `himmy guardrails`. The built-in catalog is the SAME
  `builtin_rows()` the CLI prints (`cli/guardrails_view.py`); recent **firings** are
  read from the Studio run store (`.himmy/studio.db`), where every `GUARDRAIL_APPLIED`
  event is recorded as a `CognitionStep` of kind `safety` carrying the verdict's
  `stage`/`blocked`/`redacted`/`flags`/`reasons`, each linked back to its `run_id`.
- **Security-log viewer** (`GET /api/studio/seclog`, `routers/studio_seclog`) — the GUI
  sibling of `himmy seclog`, rendered on the Privacy screen. It reads
  `app.state.security_audit` (a `SecurityAuditLog` over the container's entity
  registry), which since the `SpineFactory` refactor IS the durable shared
  `.himmy/spine.db` the CLI also reads — so the viewer's rows are **identical** to
  `himmy seclog --limit N --type T` for the same project. Single-user-local: no
  workspace filter; the `limit`/`type` filters mirror `render_seclog`.

## How it works / launching

```bash
# CLI (offline, no keys needed)
himmy init my-agent
himmy run -f my-agent/agent.yaml -p "hello"
himmy chat -f my-agent/agent.yaml --session work
himmy run -p "..." --trace          # then: himmy trace
himmy doctor

# pick a backend
himmy run -p "..." --provider ollama --model qwen2.5:3b-instruct
himmy run -p "..." --provider claude-cli

# API only
himmy serve                          # FastAPI BFF on 127.0.0.1:8000 (needs api extra)

# Studio (needs the studio extra + a built frontend)
cd studio && npm install && npm run build   # emits into himmy/api/_studio_static
himmy studio                                # 127.0.0.1:8765, opens a browser
# dev with hot reload (two terminals):
himmy serve                                 # API on :8000
cd studio && npm run dev                    # UI on :5173, proxies /api → :8000
```

`himmy studio` boots the same `create_app()` BFF bound to loopback (it can run agents
and write files); if the SPA hasn't been built it prints the exact build/dev commands
instead of an empty shell. A deployable image is provided by the repo `Dockerfile`
(`himmy studio --host 0.0.0.0 --port 8765 --no-browser`).

## Configuration

| Var | Effect |
| --- | --- |
| `HIMMY_NO_HINTS` | Suppress the "running on the stub" CLI hint. |
| `OPENROUTER_API_KEY` | Required for `--provider openrouter`. |
| `HIMMY_STUDIO_GUARD` / `HIMMY_STUDIO_ALLOW_HOSTS` | Loopback / same-site guard on `/api/studio` (on by default). |
| `HIMMY_INTERNAL_API_KEY` | Guards every BFF route behind a trusted-boundary header. |

Provider/model also come from `agent.yaml` (`provider:` / `model:`) and project
defaults in `himmy.toml`; CLI flags override both.

## Extension points

- Add a provider branch in `provider.py::build_manager_for` (and to `PROVIDERS`).
- Add a CLI subcommand: a `cmd_*` in `commands.py` + a subparser in `__main__.py`.
- Add a Studio feature: a `studio_<feature>.py` backend + routes in
  `routers/studio.py` + a screen in `studio/src/screens/`.
- `init --template` starters live in `_TEMPLATES` in `commands.py`.

## Gotchas & invariants

- The offline stub is the default — print real-model hints, don't fail.
- `himmy serve` / `himmy studio` need the `api` / `studio` extras (a clean error if
  missing).
- The SPA mount must come last and use a 404-fallback so it never shadows an API
  route; unknown API paths keep a JSON 404.
- The `/api/studio` surface is loopback-only by design (it runs agents + stores
  credentials); the host/origin guard defends against DNS-rebinding.
- Studio local stores live under `.himmy/` (per project / cwd).

## Related docs

- [Local-first architecture](local-first.md) — the offline-by-default philosophy and the optional extras.
- [Audit](../services/audit.md) — the BFF's security wiring (auth, RBAC, rate-limit).
- [Record/replay & compaction](../design/record_replay_and_compaction.md)
