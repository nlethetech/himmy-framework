# Prompts Service

> Render system and task prompts from typed variables and versioned YAML templates, and project a context snapshot's fields into delimited prompt blocks — never hardcoded strings in Python.

## Overview

`himmy/services/prompts/` turns structured inputs (persona, objectives, skills, task,
output format) into the two strings a run sends to the model: a **system prompt** and a
**task prompt**. Two collaborators:

- **`PromptManager`** renders prompts from a sectioned YAML template (`configs/prompts/
  default_prompt.yaml`) and typed variable objects. Business/wording owners can edit a
  template without a code deploy.
- **`ContextPromptMapper`** (the context→prompt mapping) projects selected
  `ContextSnapshot` fields into fenced, attributed blocks appended to the system/task
  prompts, with per-key truncation and redaction.

The runtime (`himmy/runtime/single_agent.py`, `_render_prompts`) wires these together:
it builds `SystemPromptVariables`/`TaskPromptVariables` from the persona, task, and run
context, renders via `PromptManager`, then appends snapshot blocks via
`ContextPromptMapper`.

## Module map

| File | Responsibility |
| --- | --- |
| `manager.py` | `PromptManager`, `PromptTemplate`, `SystemPromptVariables`, `TaskPromptVariables`, the `{name}`-only renderer, template loading + cache. |
| `mapper.py` | `ContextPromptMapper`, `ContextPromptMapSpec`, `ContextPromptKey` — snapshot-field → prompt-block projection. |
| `configs/prompts/default_prompt.yaml` | The bundled default template (`persona`, `conduct`, `task` sections). |
| `__init__.py` | Public re-exports. |

## Key abstractions

### `PromptManager` & templates (`manager.py`)

- **`SystemPromptVariables`** — `role`, `persona` (background/description),
  `objectives: list[str]`, `skills: list[str]`, `datetime`.
- **`TaskPromptVariables`** — `task`, `output_format`, `output_schema: dict | None`.
- **`PromptTemplate`** — loads one or more sectioned YAML files and merges them
  deterministically (later files win per `(section, key)`). Parsed files are cached by
  `(resolved path, mtime)`; malformed structure raises a clear `HimmyError`.
- **`PromptManager(template_paths=...)`** — defaults to the bundled
  `default_prompt.yaml`. Two render methods (below).

The renderer (`_render`) substitutes **only** `{name}` placeholders that map to a
supplied value; unknown placeholders, other braces, and `$` are left verbatim (so
`Use set {a, b}` or `Budget is $total` survive intact). It is a real `{name}`-only
formatter, not a global brace rewrite.

### The template (`default_prompt.yaml`)

Sections used by the manager:

- **`persona`**: `role`, `background`, `objectives`, `skills` sub-templates.
- **`conduct`**: a `default` block of framework-wide operating principles (ground
  answers, call tools for current facts, use the user's exact terms, cite source URLs)
  — **rendered unconditionally** on every system prompt.
- **`task`**: `task`, `output`, `schema` sub-templates.

### `ContextPromptMapper` (`mapper.py`)

- **`ContextPromptKey`** — `key`, `required`, `max_chars`, `redact`. A bare string
  coerces to `{"key": value}`.
- **`ContextPromptMapSpec`** — `system_keys`, `task_keys`, `default_max_chars` (a cap
  for keys that don't set their own; `None` = unlimited).
- **`ContextPromptMapper.project(snapshot, spec) -> (system_block, task_block,
  missing)`** — renders each selected snapshot key as a delimited block; required keys
  absent from the snapshot return in `missing`.

## How it works / data flow

### Rendering (`PromptManager`)

- `get_system_prompt(vars)` renders the `persona` section blocks **only for non-empty
  values** (role, background, objectives, skills), then **always** appends the
  `conduct.default` block. Joined with blank lines.
- `get_task_prompt(vars)` renders the `task` section blocks for the task, output
  format, and (JSON-serialized) output schema, including only non-empty ones.

### Context → prompt projection (`ContextPromptMapper`)

- Each selected key renders as a fenced, attributed block:
  `<context key="...">\n<value>\n</context>` — deliberately *not* a raw `###` heading,
  so an arbitrary key name can't be misread by the model as instruction structure.
- Values are rendered as text (JSON for non-string structures), truncated to the
  per-key `max_chars` (or the spec's `default_max_chars`) with a `... [truncated N
  chars]` marker, or replaced with `[REDACTED]` when `redact=True`.
- Required keys missing from the snapshot are collected and returned.

### Composition: persona instructions + skills + context (runtime)

In `_render_prompts` (`himmy/runtime/single_agent.py`):

1. **Objectives** = persona `instructions` + persona `objectives` + any
   `ctx["objectives"]`. (Persona instructions are rendered as objectives so they reach
   the model even when a description is set.)
2. **Skills** = `ctx["skills"]` if present, else `persona.metadata["skills"]` or
   `persona.required_skills`.
3. `SystemPromptVariables(role=ctx role or persona.role, persona=persona.description,
   objectives, skills, datetime)` → `get_system_prompt`.
4. `TaskPromptVariables(task=task.prompt, output_format, output_schema)` →
   `get_task_prompt` (falls back to `task.prompt`).
5. An optional `ctx["system_prefix"]` is prepended.
6. When a `context_prompt_map_spec` and a `snapshot` are present, the mapper projects
   snapshot fields into `sys_block`/`task_block`, appended to the respective prompts;
   projection errors degrade to an empty `missing` list rather than failing the run.

So a final system prompt composes: persona identity + instructions/objectives + skills
+ framework conduct + (optional) projected context blocks; the task prompt composes:
the task + output format/schema + (optional) projected context blocks.

## Configuration

- **Custom templates:** `PromptManager(template_paths=[...])`. Multiple files merge
  with later-wins per `(section, key)`, enabling layered/versioned overrides on top of
  the bundled default.
- **Conduct:** edit the `conduct.default` block in the template to change the
  framework-wide operating principles (applied to every agent).
- **Context projection:** supply a `ContextPromptMapSpec` (as `context_prompt_map_spec`
  in run context) selecting which snapshot keys flow into system vs task blocks, with
  per-key `max_chars`/`redact` and a `default_max_chars`.
- Both `PromptManager` and `ContextPromptMapper` are overridable in the runtime
  builder (`himmy/runtime/builder.py`) and default to fresh instances.

## Extension points

- **New prompt wording / sections:** edit or add YAML template files; no code change.
- **New context block policy:** construct `ContextPromptMapSpec` with the desired
  `system_keys`/`task_keys` and caps.
- **Versioning:** template paths + later-wins merge support layered overrides; the
  parsed-template cache keys on `(path, mtime)` so edits are picked up.

## Gotchas & invariants

- The renderer is `{name}`-only: literal braces and `$` are preserved; unknown
  placeholders stay verbatim (they are not errors).
- The `conduct` block is rendered **unconditionally** on every system prompt — empty
  persona values are skipped, conduct is not.
- Context blocks use a `<context key="...">` fence, never a markdown heading, to avoid
  the model misreading a key name as instruction structure.
- Templates are validated as a mapping-of-mapping-of-string; malformed structure raises
  `HimmyError`.
- Persona `instructions` are surfaced as objectives (not as the background/description)
  so they always reach the model.

## Related docs

- [Context](context.md) — produces the `ContextSnapshot` the mapper projects.
- [Knowledge](knowledge.md) — retrieved chunks arrive as context fields and can be
  projected into prompt blocks.
