# Himmy recipes — running agents on a real model

The unit tests run against an offline deterministic stub. These recipes run the *same*
framework against a **real local model** so you can see tools actually fire. They need no
cloud keys — just [Ollama](https://ollama.com) with a tool-capable model pulled
(`qwen2.5`, `llama3.1`, `mistral-nemo`, …):

```bash
ollama pull qwen2.5:3b-instruct      # small + tool-capable
ollama serve                          # if not already running
```

All examples are provider-selectable via env vars (default = offline stub):

```bash
export HIMMY_EXAMPLE_PROVIDER=ollama
export HIMMY_EXAMPLE_MODEL=qwen2.5:3b-instruct
```

## 1. A tool-using agent (`examples/10_real_tool_agent.py`)

```bash
HIMMY_EXAMPLE_PROVIDER=ollama HIMMY_EXAMPLE_MODEL=qwen2.5:3b-instruct \
    python examples/10_real_tool_agent.py
```

Verified output — the model calls the tool, the runtime executes it, the model answers:

```
stopped_reason : final   turns: 2
  turn 0: tool_calls=[('calculator', {'expression': '17*23'})]
  turn 1: tool_calls=[]
answer         : The result of multiplying 17 by 23 is 391.0.
```

## 2. Web research (`examples/11_web_research.py`)

Real model + keyless DuckDuckGo search + live web (needs network):

```bash
HIMMY_EXAMPLE_PROVIDER=ollama HIMMY_EXAMPLE_MODEL=qwen2.5:3b-instruct \
    python examples/11_web_research.py
```

Verified: the model calls `web_search("permaculture food forest")`, the runtime hits the
live web, and the model summarizes —
*"A permaculture food forest is a type of garden where you grow many different fruits,
nuts, herbs, and vegetables, mimicking a natural forest with various plant layers."*

## 3. Durable semantic memory (`examples/12_memory_recall.py`)

```bash
python examples/12_memory_recall.py                          # deterministic (exact-overlap)
# real semantic recall (pull the embed model first: ollama pull nomic-embed-text):
HIMMY_EMBEDDER=ollama HIMMY_EMBEDDER_MODEL=nomic-embed-text \
    python examples/12_memory_recall.py
```

The deterministic embedder only matches shared words; a real embedder (`nomic-embed-text`)
recalls by meaning — try the query *"how is honey produced"* and watch the bee fact rank
first only with the real embedder.

## Via the CLI

```bash
himmy init my-agent
# edit my-agent/agent.yaml: provider: ollama, model: qwen2.5:3b-instruct, tool_packs: [web]
himmy run -f my-agent/agent.yaml -p "Research permaculture and summarize."
```

## Mixed-provider team — a strong brain + cheap local workers

Each team member can declare **its own `provider` + `model`**, so a strong model
orchestrates while free local models do the grunt work. The CLI builds a multi-provider
dispatcher automatically:

```yaml
# team.yaml
entry: brain
members:
  - name: brain            # the orchestrator / decision-maker
    description: Decide the approach and delegate the work.
    provider: claude-cli   # Claude Max (Opus) as the brain
    model: opus
    delegates: [researcher, writer]
  - name: researcher       # cheap local worker
    description: Gather facts from the web.
    provider: ollama
    model: qwen2.5:3b-instruct
    tool_packs: [web]
    tools: [web_search]
  - name: writer           # another local worker
    description: Write the final answer in Nepali.
    provider: ollama
    model: qwen2.5:3b-instruct
    language: ne
```

```bash
himmy team -f team.yaml -p "Research permaculture and write a summary."
```

Members without a `provider` fall back to the CLI's `--provider`/`--model` (or the
framework default). Under the hood each `model_key` routes to its own backend via
`MultiProviderClientManager` — so you pay for the strong model only where it reasons, and
run everything else locally for free.

## Plug in MCP servers — the whole ecosystem as agent tools

Instead of hand-writing a connector for every service, point himmy at any **stdio MCP
server** and its tools become native agent tools. List them under `mcp_servers:` in
`agent.yaml` (or `team.yaml`, shared by the team):

```yaml
# agent.yaml
name: dev-assistant
provider: ollama
model: qwen2.5:3b-instruct
mcp_servers:
  - command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"]
    prefix: fs_                 # tools become fs_read_file, fs_list_directory, …
  - command: uvx
    args: ["mcp-server-git", "--repository", "."]
    requires_approval: true     # gate this server's tools behind approval
    tools: [git_log, git_diff]  # bind only a subset (empty = all)
```

```bash
himmy run -f agent.yaml -p "Summarize the recent git history and list /tmp/workspace."
```

The CLI launches each server, registers its tools (arg-validated against the server's
own JSON Schema, with approval gating + events + lineage like any native tool), runs the
agent, then tears the servers down. Leave the agent's `tools:` empty to advertise every
MCP tool to the model, or name the (prefixed) ones you want bound. Works the same in
`himmy chat`, `himmy team`, and `himmy eval`.

> Verified offline against an in-repo mock MCP server: `himmy run` connected it,
> registered `mock_echo`/`mock_add`, and the model called them through the full pipeline.

## Notes from real-model testing

- **Tool calling works** on Ollama (native `/api/chat` tools) and, best-effort, on the
  Claude CLI (a text ReAct protocol). The runtime executes the tool and feeds the result
  back, so the model answers instead of looping.
- **Model size matters for orchestration.** Small models (3B) reliably call tools but may
  *not* choose to hand off in a team — they answer directly when they can. The handoff
  machinery is correct; reliable multi-agent routing wants a larger/instruct model.
- **Structured output** and **guardrails** behave the same on real models as on the stub.
