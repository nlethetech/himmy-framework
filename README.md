# Himmy

**Build AI agents that run entirely on your own machine — no API keys, no cloud, every
run replayable.**

Himmy is an **offline-first** Python agent framework. The default needs zero network and
zero keys; real local models run through **Ollama** or the **Claude CLI**; cloud providers
(OpenAI / Anthropic) are an opt-in extra, not a requirement. Under the hood every persona,
prompt, tool, message, and run event is an immutable, versioned `EntityRecord` — so any
answer can be traced, or **replayed exactly**, months later.

### See it in 30 seconds — chat with your own docs, fully offline

```bash
pip install -e ".[knowledge]"
ollama pull qwen2.5:3b-instruct
cd examples/local-doc-chat
python chat.py "How many PTO days do I get?"
```

```
✓ ingested 3 documents into a local knowledge base — offline, no keys.

you   › How many PTO days do I get?
agent › Full-time employees accrue 20 days of paid time off per year, rolling over up
        to 5 days. (source: handbook-pto.md)
```

No keys. No cloud. Your documents never leave the machine. → **[full example](examples/local-doc-chat/)**

### Why himmy

- **Offline-first** — Ollama / Claude CLI / a deterministic stub; cloud is optional, never required.
- **Auditable + replayable** — re-run a failed agent run *exactly* from recorded model
  responses, with the provider turned off. Most frameworks can't do this.
- **Batteries included** — declarative agents in YAML, a [skills](#extend-it-tools-knowledge-and-skills)
  layer, 13 tool packs, MCP servers, memory, RAG, and multi-agent — all offline-capable.
- **Honest about quality** — a standing benchmark runs in CI against a real local model, so
  "did my change make agents better or worse?" is actually answerable.

> **Everything that matters is an entity.** Personas, prompts, tools, threads, messages,
> context snapshots, and run events are all immutable, versioned `EntityRecord`s with typed
> links — the audit/lineage spine the rest of the framework is built on.

### Prefer a GUI? Himmy Studio

A local web app for everything below — no terminal required:

```bash
cd studio && npm install && npm run build   # build the GUI once (Node 18+)
himmy studio                                 # opens http://127.0.0.1:8765
```

- **Chat** with any agent (live token streaming, multi-turn, tool calls).
- **Agents** — build/edit an `agent.yaml` from a form: pick skills, tool packs,
  knowledge, guardrails — no YAML by hand.
- **Runs** — browse every past run with its transcript and step-by-step trace.
- **Doctor** — what providers/keys/extras are available, and your next step.

Served by the same FastAPI BFF, bound to loopback. For hot-reload development run
`himmy serve` + `cd studio && npm run dev` (Vite proxies the API).

## Install

```bash
# Core (offline) install — pydantic, pyyaml, httpx only.
pip install -e .

# With the local API server (FastAPI + uvicorn).
pip install -e ".[api]"

# Everything (providers, postgres, knowledge, observability, dev tooling).
pip install -e ".[all]"
```

Requires Python **3.12+**. The full list of [optional extras is in the advanced
guide](docs/advanced.md#optional-extras).

---

# Three concepts, start to finish

You only need three things to ship an agent with Himmy: **define it**, **run it**, and
**extend it** with tools and knowledge. Everything else is optional and lives in the
**[advanced guide](docs/advanced.md)**.

## 1. Define it — `agent.yaml`

An agent is a small YAML file. Scaffold one (with an example `tools.py` and `skill`):

```bash
himmy init my-agent
```

```yaml
# my-agent/agent.yaml
name: market-analyst
description: A market research analyst specializing in tech.
role: Research Analyst
instructions:
  - Provide actionable insights backed by clear reasoning.
model: default
# provider: ollama   # stub | claude-cli | ollama | pydantic-ai (default: auto)
```

`agent.yaml` is a thin declarative façade over `Persona` + `Task` + `LLMConfig` — the same
spec loads in Python via `from himmy import load_agent_spec`.

## 2. Run it

Everything below runs **offline against the deterministic stub** — no keys needed. Add a
real model with `--provider ollama` (or set a key); see [choosing a
provider](docs/advanced.md#choosing-a-provider).

```bash
# One-shot: run a prompt and print the answer
himmy run -f my-agent/agent.yaml -p "Say hello in one sentence."

# Interactive chat that keeps a single thread
himmy chat -f my-agent/agent.yaml

# Is a real model available on this machine? What's my next step?
himmy doctor
```

The smallest possible run, in pure Python:

```python
import asyncio

from himmy import Persona, Task
from himmy.runtime import SingleAgentRuntime
from himmy.services.inference import InferenceService, StubClientManager


async def main() -> None:
    runtime = SingleAgentRuntime(
        inference_service=InferenceService(StubClientManager()),
    )
    persona = Persona(name="analyst", description="A concise research analyst.")
    task = Task(title="ping", prompt="Say hello in one sentence.")
    thread = await runtime.run_task(persona, task)
    print(thread.last_message.content)


asyncio.run(main())
```

## 3. Extend it — tools, knowledge, and skills

This is the part people confuse. Three words, one ladder:

```
tool        a single function the model can call          web_search
tool_pack   a named bundle of related tools               web → web_search, web_fetch, http_request
skill       a tool_pack + the know-how to use it well     web_research → the web tools + how-to + examples
```

**Rule of thumb:** reach for a **skill** first — you get the tools *and* the guidance to use
them. Drop to **`tool_packs`** when you want the raw tools without the know-how. Name
individual **`tools`** only to bind a subset.

```yaml
# capabilities, the easy way: tools + know-how in one line
skills: [web_research]

# or wire raw tool packs yourself
tool_packs: [web, utils]            # register these built-in packs
tools: [web_search, web_fetch]      # bind a subset to the model (omit = all)

# ground the agent in your own docs — auto-ingested into a local KB → kb_search
knowledge: [./docs]
```

List what's available: `himmy tools` (packs) and `himmy skills` (capabilities).

| Pack | Tools |
|------|-------|
| `web` | `web_search`, `web_fetch`, `http_request` |
| `files` | `read_file`, `write_file`, `list_dir` (jailed to a sandbox root) |
| `data` | `sql_query` (read-only; SQLite or Postgres) |
| `code` | `run_python` (resource-limited sandbox, approval-gated) |
| `utils` | `calculator`, `current_time` |
| `knowledge` | `kb_ingest`, `kb_search` (the agent's own RAG memory) |
| `documents` | `read_document` (PDF / text / Markdown → text) |
| `comms` | `send_email`, `send_webhook` (outbound; approval-gated) |
| `data-sources` | `weather`, `geocode`, `wikipedia` (keyless public APIs) |
| `memory` | `remember`, `recall` (durable long-term memory, SQLite-backed) |
| `nepal` | `nepali_date` (Bikram Sambat), `nepali_format` (NPR/Devanagari), `nepali_transliterate`, NRB forex |
| `agentic` | `ask_human` (human-in-the-loop), `scratchpad_set`/`get`, `todo_write`/`read` (self-managed task list) |
| `telegram` | `send_telegram` (message a Telegram chat via a bot; approval-gated) |

Built-in skills: `web_research`, `data_analysis`, `file_ops`, `python_compute`,
`knowledge_base`, `nepal_brief`, `summarize`. Author your own as a one-file YAML in
`skills/` (auto-discovered; project files shadow built-ins):

```yaml
# skills/exact_math.yaml
name: exact_math
description: Compute exact arithmetic by running Python.
tool_packs: [code]
instructions:
  - Never estimate arithmetic — run Python and print() the result.
```

### Connect your own API, no Python

Declare REST endpoints right in `agent.yaml` and they become arg-validated, traced tools:

```yaml
http_tools:
  - name: get_order
    base_url_env_var: MYAPI_URL
    path: /orders/{order_id}
    auth: { type: bearer, env_var: MYAPI_KEY }
```

### Start from a working specialised agent

```bash
himmy init my-agent --template helpdesk    # docs-grounded Q&A (knowledge)
himmy init my-agent --template analyst     # live API lookups (http_tools)
himmy init my-agent --template researcher  # web research (skills)
```

---

## CLI reference

```bash
himmy init my-agent          # scaffold an agent.yaml + example tools.py + skill
himmy run   -f agent.yaml -p "…"   # one-shot prompt → answer
himmy chat  -f agent.yaml          # interactive single-thread chat
himmy doctor                 # what extras / providers / keys are available + next step
himmy tools                  # list built-in tool packs
himmy skills                 # list skills (built-in + project-local)
himmy serve                  # serve the FastAPI BFF (needs the `api` extra)
```

Other commands — `team`, `telegram`, `eval`, `bench`, `prices`, `trace` — are covered in
the **[advanced guide](docs/advanced.md)**.

## Going further

When you outgrow a single agent, the [advanced guide](docs/advanced.md) covers:

- **[Multi-agent teams](docs/advanced.md#multi-agent-teams)** — handoff (swarm) and delegation (manager-worker).
- **[MCP servers](docs/advanced.md#mcp-servers)** — point an agent at any Model Context Protocol server; its tools become native himmy tools.
- **[Recursive sub-agents](docs/advanced.md#recursive-sub-agents)** — `allow_spawn` gives an agent a `spawn_agent` tool.
- **[Live on Telegram](docs/advanced.md#live-on-telegram)** — turn any `agent.yaml` into a Telegram bot.
- **[Embeddings](docs/advanced.md#embeddings)** — swap the offline embedder for real semantic recall (Ollama / fastembed / OpenAI).
- **[Record & replay](docs/advanced.md#record--replay)** — capture a run to a cassette and re-run it exactly, offline.
- **[Production capabilities](docs/advanced.md#production-capabilities)** — failure handling, streaming, caching, Postgres/pgvector, security, observability, evaluation.
- **[Architecture overview](docs/advanced.md#architecture-overview)** — the kernel layout.

## Running the examples

Every example runs offline against the stub:

```bash
python examples/01_basic_chat.py
python examples/02_tool_calling.py
python examples/03_structured_output.py
python examples/04_orchestration_team.py
python examples/05_workflow.py
python examples/06_postgres_storage.py   # skips cleanly unless HIMMY_TEST_POSTGRES_DSN is set
python examples/07_streaming.py          # token-delta streaming via InferenceService.run_stream
```

## Running the tests

```bash
pip install -e ".[dev]"
pytest -q
```

Tests use `asyncio.run` inside synchronous test functions, so `pytest-asyncio` is
**not** required. The full suite passes offline with the stub. Real-provider,
Postgres/pgvector, and Logfire tests **skip cleanly** when their deps/DB are absent
(set `HIMMY_TEST_POSTGRES_DSN` and install the relevant extra to exercise them);
run `pytest -q -rs` to see why each skip fired.

## Project layout

```
himmy/
  core/            ids, errors, events
  entities/        records, registry, postgres scaffold
  agents/          personas, base_agent (task, thread, agent)
  services/        inference, context, prompts, tools, storage,
                   knowledge, evaluation, observability
  runtime/         single_agent runtime
  application/     app-level services + models
  orchestrators/   workflow engine
  api/             FastAPI app + routers
examples/          runnable, offline-first demos
tests/             pytest suite (asyncio.run based)
docs/              advanced guide + design notes
docker/            pgvector docker-compose for local Postgres
```
