# Skills

> First-class, typed agent capabilities: know-how + the tools it needs + the guardrails it implies.

## Overview

A skill is the capability layer between tools and agents. A tool is a callable; a skill
is *the know-how to wield a set of tools toward a job* — a prompt fragment
(`instructions`) bundled with the `tools` / `tool_packs` it needs and any `guardrails` it
implies. Skills are authored in YAML (or seeded in code), discovered from the project,
resolved into an agent (binding their tools and injecting their guidance), and projected
to versioned `EntityRecord(kind="skill")` rows like every other domain artefact.

Declaring `skills: [web_research]` in an `agent.yaml` grants both the capability's tools
**and** the guidance for using them well. The design rationale (why skills became a
first-class entity) lives in [`../design/skills_system.md`](../design/skills_system.md).

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/skills/models.py` | The `Skill` and `SkillExample` pydantic models (`extra="forbid"` so a typo'd YAML field fails loudly). |
| `himmy/skills/loader.py` | YAML authoring + project discovery: `load_skill_file`/`load_skill_dir`, `discover_skill_dirs`, `build_skill_registry`. |
| `himmy/skills/registry.py` | `SkillRegistry` — an in-memory name→`Skill` catalog (mirrors `ToolRegistry`). |
| `himmy/skills/resolve.py` | `resolve_skills` — DFS expansion of `requires_skills`, cycle detection, dedup → a `ResolvedSkills` bundle. |
| `himmy/skills/dispatch.py` | `SkillDispatcher` + the `dispatch_skill` tool — run one skill as an isolated sub-agent. |
| `himmy/skills/errors.py` | `UnknownSkillError`, `CyclicSkillError`, `SkillToolError`, `SkillCollisionError`. |
| `himmy/skills/builtin/__init__.py` | `BUILTIN_SKILLS` — the seed catalog (validated `Skill`s defined in code). |

## Key abstractions

### `Skill` (`models.py`)

```python
class Skill(BaseModel):
    name: str                       # min_length=1
    version: int = 1
    description: str = ""
    instructions: list[str] = []    # know-how appended to the system prompt
    tools: list[str] = []           # explicit tool names to bind
    tool_packs: list[str] = []      # toolkit packs to register
    guardrails: list[str] = []      # guardrails to merge
    when_to_use: str = ""           # routing hint (feeds the tool router)
    examples: list[SkillExample] = []   # few-shot examples
    requires_skills: list[str] = []     # composition; cycle-guarded at resolve
    metadata: dict[str, Any] = {}
```

`SkillExample` is `{input, action, note}`. `Skill.to_record()` projects to an
`EntityRecord(kind="skill")`; `Skill.from_dict()` validates a parsed-YAML mapping.

### `SkillRegistry` (`registry.py`)

A catalog keyed by name: `register(skill, *, overwrite=False)` (raises
`SkillCollisionError` on a duplicate unless `overwrite`), `get`, `list`, `names`,
`__contains__`, `__len__`. Pass an `entity_registry` to auto-project every registered
skill into the entity spine. `SkillRegistry.with_builtins()` returns one pre-loaded with
`BUILTIN_SKILLS`.

### `ResolvedSkills` (`resolve.py`)

The frozen, order-stable aggregate effect of a set of skills: `skills` (contributing
names in application order), `instruction_blocks` (one labeled know-how block per skill),
`tools`, `tool_packs`, `guardrails` (each deduped), `examples`, and `when_to_use`
(routing hints). Pure data — the spec/runtime turn it into an agent.

## How it works / data flow

### Discovery

`build_skill_registry()` loads the built-in catalog (`BUILTIN_SKILLS`) and overlays
project-local skills on top. `discover_skill_dirs()` scans `./skills` plus every
directory in the `HIMMY_SKILLS_PATH` env (a `PATH`-style, `os.pathsep`-separated list),
later-wins. A project file named like a built-in **intentionally shadows** it
(`overwrite=True`, logged). `load_skill_file` defaults the skill `name` to the file stem,
so `web_research.yaml` need not repeat `name:`.

> The built-in catalog ships as validated `Skill` objects in
> `himmy/skills/builtin/__init__.py` (`web_research`, `data_analysis`, `research_writer`,
> `file_ops`, `summarize`, `knowledge_base`, `python_compute`, `nepal_brief`, `clarify`),
> not as separate YAML files. YAML authoring is for **project** skills.

### Resolution (DFS + cycle detection + dedup)

`resolve_skills(names, registry)` walks the requested skills **depth-first** so a skill's
`requires_skills` prerequisites are applied *before* it (`_ordered_skills`):

```
research_writer → requires [web_research, summarize]
  visit research_writer
    visit web_research   (no deps) → append web_research
    visit summarize      (no deps) → append summarize
  append research_writer
ordered = [web_research, summarize, research_writer]
```

A name already on the current `path` raises `CyclicSkillError` (the message names the
offending path); an unknown name raises `UnknownSkillError` with a `difflib`
did-you-mean hint. Every contribution (tools, packs, guardrails) is order-preservingly
de-duplicated (`_dedup`). Each contributing skill emits one labeled instruction block
(`Skill — <name>: <description>` + `Use this when …` + bulleted instructions).

### Merging skills into an agent (`apply_skills`)

`himmy.config.agent_spec.apply_skills(spec, registry)` consumes `spec.skills` into the
spec (returning a new spec):

- unions the resolved `tool_packs` / `tools` / `guardrails` into the spec (order-stable);
- appends the instruction blocks (and rendered examples) to the spec `description` so
  they reliably render as the agent's background;
- records the contributing names in `metadata['skills']` (which the runtime's prompt
  renderer surfaces as the agent's skills list);
- stashes explicit skill-required tool names in `metadata['resolved_skill_tools']` and
  routing hints in `metadata['skill_routing_hints']`.

`build_runtime_for_spec` ([runtime](./runtime.md)) then validates `resolved_skill_tools`
against the live tool registry and raises `SkillToolError` if a skill names a tool no
wired pack/module provides — so a missing capability fails loudly, not silently. (The
`from_spec` loader calls `apply_skills` automatically via `load_spec_file(...,
expand_skills=True)`.)

### Skill dispatch (`dispatch_skill` tool)

Where `spawn_agent` hands a sub-task to a free-form worker, `dispatch_skill` runs a
**named capability** as a focused, isolated sub-agent. `SkillDispatcher.run(skill_name,
prompt)`:

1. resolves the skill bundle;
2. builds a fresh sub-runtime via `build_runtime`, registering **only** that skill's
   `tool_packs` (when it has any);
3. builds a persona from the skill's instruction blocks + examples and binds **only** the
   skill's tools (`ctx['tool_names']`);
4. runs `run_agent_loop` (with tools) or `run_task_detailed` (no tools);
5. returns `{answer, skill, applied_skills, tool_packs, status}`.

`register_skill_dispatch_tool(registry, *, inference, skill_registry, …)` binds the
`dispatch_skill(skill, prompt)` tool whose description lists the available skills. An
`UnknownSkillError` is tolerated (reported, not raised, so it doesn't fail the parent
run). Because the sub-runtime has no `dispatch_skill` tool, recursion is capped at one
level. An agent opts in via `AgentSpec.allow_skill_dispatch`.

### Projection to EntityRecords

A `Skill` is a versioned domain artefact: `to_record()` projects it to
`EntityRecord(kind="skill")` (content-addressed, namespace `skill`, stable value =
skill name). When a `SkillRegistry` is constructed with an `entity_registry`, every
registered skill is auto-projected, so the skill catalog itself is part of the audit
spine.

## Configuration

- `AgentSpec.skills: list[str]` — declare capabilities in `agent.yaml`.
- `AgentSpec.allow_skill_dispatch: bool` — give the agent the `dispatch_skill` tool.
- `HIMMY_SKILLS_PATH` — extra project-skill directories (`PATH`-style, later-wins);
  `./skills` is always scanned.

A skill YAML file is just the `Skill` model serialized:

```yaml
# skills/web_research.yaml
name: web_research            # optional — defaults to the file stem
description: Find and synthesize facts from the open web, with citations.
when_to_use: the question needs current or external facts you cannot recall.
tool_packs: [web]
guardrails: []
requires_skills: []
instructions:
  - Search before answering; never guess a fact you can look up.
  - Cite the URL(s) you used so the answer is verifiable.
examples:
  - input: "What's the latest on X?"
    action: "web_search then open the top source and quote it"
    note: "ground the claim in the source"
```

## Extension points

- **Author a skill** → drop a YAML file in `./skills` (or a `HIMMY_SKILLS_PATH` dir);
  shadow a built-in by reusing its name.
- **Compose skills** → use `requires_skills` (DFS-resolved, cycle-guarded).
- **Register in code** → `SkillRegistry.register(Skill(...))` or add to
  `BUILTIN_SKILLS`.
- **Run a skill as a sub-agent** → enable `dispatch_skill`, or call `SkillDispatcher`
  directly.

## Gotchas & invariants

- `Skill` uses `extra="forbid"`: a typo'd field in a hand-written YAML fails at load
  time, not silently.
- A skill naming a tool that no wired pack/module provides raises `SkillToolError` at
  wiring time (existence is validated against the **live** tool registry, separate from
  resolution — a skill may name a tool that only exists once its pack is registered).
- `requires_skills` cycles raise `CyclicSkillError`; the prerequisite is always applied
  before the skill that requires it.
- A dispatched skill cannot dispatch again (one-level recursion cap by construction).

## Related docs

- [config](./config.md) — `AgentSpec.skills` / `allow_skill_dispatch` and `apply_skills`.
- [runtime](./runtime.md) — where skills are validated against the tool registry and
  routing hints feed the tool router.
- [overview](./overview.md) — the toolkit packs skills bind.
- [`../design/skills_system.md`](../design/skills_system.md) — the design rationale.
