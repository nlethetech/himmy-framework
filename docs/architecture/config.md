# Configuration

> Spec-driven, declarative config: define agents, teams, workflows, eval suites, HTTP/MCP tools, secrets, and residency without code.

## Overview

`himmy/config/` is the declarative façade over the framework's runtime primitives. A user
describes an agent (or team, workflow, eval suite) in a YAML file; the spec classes
project that description onto `Persona` / `Task` / `LLMConfig` / `AgentTeam` / `Workflow`
so it can be run from the CLI or Himmy Studio with no Python. Imports stay light/offline:
the heavier dependencies (`LLMConfig`, orchestrators, tool registry) are pulled in lazily
inside the methods that need them.

Alongside the specs, this package owns the cross-cutting operational config: the
`SecretProvider` abstraction (env / file / vault / cloud), data-residency pinning, and
the `himmy.toml` project defaults.

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/config/agent_spec.py` | `AgentSpec` (the `agent.yaml` model) + `to_persona()` / `to_llm_config()` / `make_task()` projections, `apply_skills`, `load_agent_spec`. |
| `himmy/config/team_spec.py` | `TeamSpec` / `TeamMemberSpec`, `build_team`, `build_team_inference` (mixed-provider teams). |
| `himmy/config/workflow_spec.py` | `load_workflow_spec` — parse a `workflow.yaml` into a `Workflow`. |
| `himmy/config/eval_spec.py` | `load_eval_suite` — parse a `suite.yaml` into an `EvaluationSuite`. |
| `himmy/config/http_tool_spec.py` | `HttpToolSpec` — a declarative REST tool (no Python) + `register_http_tools`. |
| `himmy/config/mcp_spec.py` | `MCPServerConfig` — a stdio MCP server as agent tools; `attach_mcp_servers` / `close_mcp_clients`. |
| `himmy/config/secrets.py` | `SecretProvider` abstraction + Env/File/Keychain/Vault/AWS/GCP/Azure backends; `get_secret`. |
| `himmy/config/residency.py` | Data-residency / region pinning: `enforce_region`, `region_allowed`. |
| `himmy/config/project.py` | `himmy.toml` loader: `find_project_config`, `load_project_config`. |

## Key abstractions

### `AgentSpec` (`agent_spec.py`)

The full `agent.yaml` model. Fields include identity (`name`, `description`,
`instructions`, `role`), model selection (`model` = the model_key, `provider`,
`temperature`, `max_tokens`), capabilities (`skills`, `tools`, `tool_packs`,
`tools_module`, `http_tools`, `mcp_servers`, `allow_spawn`, `allow_skill_dispatch`,
`tool_router`), behavior (`guardrails`, `memory` + `memory_top_k`, `knowledge`,
`compact_context` + `compact_after_tokens` + `compact_keep_recent`, `language`,
`output_schema`), and `metadata`.

Projections:

- `to_persona()` → a `Persona` (role folded into metadata; `language: ne` appends a
  "respond in Nepali (Devanagari)" instruction).
- `to_llm_config()` → an `LLMConfig` (model_key + `output_json_schema` + generation
  knobs; `response_format` is auto-derived — a schema yields `STRUCTURED_OUTPUT`).
- `make_task(prompt, *, title=None)` → a `Task` whose `context` mirrors the LLM config so
  a run still binds the declared tools/schema even without an explicit `llm_config`. It
  wires `model_key`, `tool_names`, `skill_routing_hints`, `compaction_spec` (from the
  `compact_*` fields), `output_schema`, and — when `memory` is on — a
  `context_build_spec` + `context_prompt_map_spec` that recall long-term memory into the
  system prompt (needs a runtime wired with a memory `ContextAdapter`).

`apply_skills(spec, registry)` expands `spec.skills` into the spec (see
[skills](./skills.md)). `load_agent_spec(path)` reads the YAML and inlines an
`output_schema` given as a path to a `.json` file (resolved relative to the YAML).

### `TeamSpec` / `TeamMemberSpec` (`team_spec.py`)

`TeamMemberSpec` describes one member (name, description, instructions, role, per-member
`provider`/`model`, `tools`, `tool_packs`, `tools_module`, `handoffs`, `delegates`).
`TeamSpec` is `members` + an `entry` member (+ shared `mcp_servers`).

- `build_team(spec, …)` → an `AgentTeam` + a `ToolRegistry` pre-loaded with each member's
  packs / custom tools (and where the orchestrator later registers synthetic
  handoff/delegate tools). Each member's `model_key` is `<provider>:<model>` when it sets
  a provider, else its plain `model` (`_dispatch_key`).
- `build_team_inference(spec, …)` → an `InferenceService`. When any member declares its
  own `provider`, the team runs on a `MultiProviderClientManager` that dispatches each
  member's `model_key` to its own backend (a strong brain + cheap local workers — see
  `RECIPES.md`); otherwise a single backend is used.

### `WorkflowSpec` / `EvalSpec`

`load_workflow_spec(path)` parses a `workflow.yaml` straight into a
`himmy.orchestrators.workflow.Workflow` (the YAML shape mirrors the model: `name`,
`description`, `steps[]`). `load_eval_suite(path)` parses a `suite.yaml` into a
`himmy.services.evaluation.models.EvaluationSuite` (cases with `input`,
`expected_output`, `metric_weights`) for `himmy eval`. Both default `name` to the file
stem.

### `HttpToolSpec` (`http_tool_spec.py`)

A declarative REST tool authored in YAML (`extra="forbid"`). Fields: `name`,
`description`, `base_url` or `base_url_env_var`, `method`, `path` (with `{placeholders}`),
`query` / `body` / `headers` (arg-name lists), `auth` (`{type: none|bearer|header|basic,
env_var, header_name}`), `timeout_seconds`, `requires_approval`, optional `args_schema`.
The args JSON Schema is derived automatically from the path placeholders + query/body/
header names (path args required) unless overridden. `register(registry)` /
`register_http_tools(registry, specs)` wire them as native tools (arg validation,
approval gating, events, lineage).

### `MCPServerConfig` (`mcp_spec.py`)

Declarative launch + binding config for one stdio MCP server: `command`, `args`, `env`,
`cwd`, `prefix` (tool-name namespace), `requires_approval`, `tools` (subset to bind;
empty = all). `attach_mcp_servers(registry, configs)` connects each server and registers
its tools as native `ToolDefinition`s (returning the live clients so the caller can
`close_mcp_clients` after the run). Each MCP tool flows through the same pipeline as a
built-in tool. The CLI launches the servers inside one `asyncio.run` so the reader tasks
stay on a live loop.

### `SecretProvider` (`secrets.py`)

A Protocol — `get(name) -> str | None` — so every secret is read through `get_secret`
rather than straight from `os.environ`. `build_secret_provider()` selects the backend
from `HIMMY_SECRETS`; all non-`env` backends chain an env fallback, so a partially
migrated deployment keeps working.

| `HIMMY_SECRETS` | Backend | Notes |
| --- | --- | --- |
| `env` *(default)* | `EnvSecrets` | identical to reading `os.environ` — offline/zero-config unchanged. |
| `file` | `FileSecrets` | `<NAME>_FILE` pointer or `<NAME>` under `HIMMY_SECRETS_DIR` (Docker/K8s secrets). Writable. |
| `keychain` | `KeychainSecrets` (macOS) | login keychain via the `security` CLI; falls back to writable `FileSecrets` off macOS. |
| `vault` | `VaultSecrets` | HashiCorp Vault KV v2 over plain HTTP (no SDK). |
| `aws` | `AwsSecretsManager` | boto3, lazily imported. |
| `gcp` | `GcpSecretManager` | google-cloud-secret-manager, lazily imported. |
| `azure` | `AzureKeyVault` | azure-keyvault-secrets, lazily imported. |

`WritableSecretProvider` (set/delete) is implemented by the file/keychain backends so the
Studio Connections UI can persist a pasted token; `get_writable_provider()` unwraps the
active chain to find one. `configure_secrets(provider)` installs a process-wide provider
(for embedding/tests).

### Residency (`residency.py`)

When `HIMMY_REGION` is set, the deployment is pinned: an operation targeting a region not
in `HIMMY_ALLOWED_REGIONS` (defaulting to the home region) is refused by
`enforce_region(region)`, which raises `ResidencyError`. Unset ⇒ no pinning. Wire
`enforce_region` where data leaves a region (storage/inference builders, cross-region
connectors). `current_region`, `allowed_regions`, `region_allowed` are the helpers.

### Project defaults (`project.py`)

`himmy.toml` (cwd) or `~/.himmy/config.toml` sets per-project defaults once.
`[defaults]` (`provider`, `model`, `tool_packs`, `guardrails`) is applied to the spec by
`apply_project_defaults` ([runtime](./runtime.md)); `[toolkit]` feeds `ToolkitConfig`.
Precedence: **CLI flag > env > `himmy.toml` > built-in default**.

## How it works / data flow

```
agent.yaml ──load_agent_spec──▶ AgentSpec
             apply_project_defaults (himmy.toml [defaults])
             apply_skills (skills → tools/packs/guardrails/know-how)
                            │
            ┌───────────────┼────────────────────┐
   to_persona()      make_task(prompt)      to_llm_config()
        │                   │                     │
     Persona             Task(context=…)       LLMConfig
        └─────────── build_runtime_for_spec ──────┘
                    (tools, packs, http_tools, mcp_servers,
                     knowledge, memory, guardrails, spawn,
                     skill dispatch) → SingleAgentRuntime
```

Secrets and residency cut across this: any backend credential is fetched via
`get_secret`, and region-leaving operations call `enforce_region`.

## Configuration

### Annotated `agent.yaml`

```yaml
name: market-analyst                 # required
description: A market research analyst specializing in tech.
role: Research Analyst               # folded into persona metadata
instructions:
  - Provide actionable insights backed by clear reasoning.

model: sonnet                        # the model_key handed to the provider
provider: claude-cli                 # stub | claude-cli | ollama | pydantic-ai | openrouter
temperature: 0.3
max_tokens: null

skills: [web_research]               # capability bundles → tools + injected know-how
tools: [web_search, web_fetch]       # explicit tool names to bind
tool_packs: [web, data]              # built-in toolkit packs
tools_module: tools:register         # dotted path to a register(registry) function
tool_router: true                    # route to the few relevant tools per turn

http_tools:                          # declarative REST tools (no Python)
  - name: get_weather
    description: Current weather for a city.
    base_url: https://api.example.com
    path: /weather/{city}
    query: [units]
    auth: { type: bearer, env_var: WEATHER_API_KEY }

mcp_servers:                         # stdio MCP servers → agent tools
  - command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/tmp/workspace"]
    prefix: fs_                      # tools become fs_read_file, fs_list_directory, …
  - command: uvx
    args: ["mcp-server-git", "--repository", "."]
    requires_approval: true          # gate this server's tools behind approval
    tools: [git_log, git_diff]       # bind only a subset (empty = all)

allow_spawn: true                    # give the agent a spawn_agent tool (1-level recursion)
allow_skill_dispatch: true           # give the agent a dispatch_skill tool

guardrails: [pii]                    # input + tool-arg redaction/blocking
memory: true                         # auto-recall durable memory into each prompt
memory_top_k: 5

knowledge: [./docs]                  # auto-ingest text docs into the agent's KB (RAG)

compact_context: true                # summarize old turns past the budget
compact_after_tokens: 3000
compact_keep_recent: 6

language: en                         # "ne" → respond in Nepali (Devanagari)

output_schema: null                  # inline JSON Schema dict, or a path to a .json file
metadata: {}
```

(The shipped `default.agent.yaml` is a smaller real example; `default-team.yaml` shows a
coordinator-plus-specialists `team.yaml`.)

## Extension points

- **New declarative tool** → `HttpToolSpec` (REST) or `MCPServerConfig` (MCP) in the
  spec; or a `tools_module` dotted path to a `register(registry)` function.
- **New secret backend** → implement the `SecretProvider` Protocol and select it via
  `HIMMY_SECRETS` (or `configure_secrets`).
- **Mixed-provider team** → give members their own `provider`/`model`; `build_team_inference`
  builds the multi-provider dispatcher.
- **Project-wide defaults** → set them once in `himmy.toml` instead of repeating flags.

## Gotchas & invariants

- `HttpToolSpec` uses `extra="forbid"` — a typo'd field fails at load, not silently.
- `model` is a **model_key** string (default `"default"`), not a literal model name; the
  active provider resolves it. A member's `model_key` becomes `<provider>:<model>` only
  when it sets a provider.
- Non-`env` secret backends always chain an env fallback, so migration is incremental.
- Residency only enforces when `HIMMY_REGION` is set; otherwise everything is allowed.
- Spec defaults: a spec value still wins over `himmy.toml` (`apply_project_defaults` only
  fills *unset* fields).

## Related docs

- [overview](./overview.md) — where config sits in the layered stack.
- [runtime](./runtime.md) — `build_runtime_for_spec` consumes these specs.
- [orchestrators](./orchestrators.md) — `TeamSpec` / `WorkflowSpec` feed the orchestrators.
- [skills](./skills.md) — `AgentSpec.skills` and `apply_skills`.
