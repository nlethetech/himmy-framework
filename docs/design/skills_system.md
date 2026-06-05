# Himmy Skills — production-grade, first-class capability system

## Problem

`skills` is the one agent-domain concept that is **not** a first-class entity. Today:

- `AgentSpec` has **no `skills` field at all** — you cannot declare skills in `agent.yaml`.
- The only carriers are `Agent.skills: list[str]`, `RolePersona.required_skills: list[str]`,
  and `SystemPromptVariables.skills: list[str]`.
- `_render_prompts` (runtime/single_agent.py) reads `ctx["skills"]` / `persona.metadata["skills"]`
  / `persona.required_skills` and renders them as a **bare bullet list** via the `skills:`
  section of `default_prompt.yaml`. Nothing populates it from a spec, validates it, binds
  tools to it, or stores it.

So a skill is a decorative string. We make it a **typed, validated, versioned, registry-backed
capability** that bundles *instructions (know-how) + tools + guardrails*, authored as a small
YAML file, and resolved into an agent automatically.

## Design principles (carried from the rest of himmy)

1. **Mirror the proven `ToolPack` pattern** — frozen catalog (`BUILTIN_SKILLS`), `resolve_*`,
   `register_*`, `Unknown*Error` with did-you-mean.
2. **Pydantic v2 everywhere**, `extra="forbid"` on the YAML model so typos fail loudly.
3. **Project onto `EntityRecord`** (`kind="skill"`) via `.to_record()`, opt-in like the rest.
4. **Offline-first, additive, opt-in** — zero new deps; agents without skills behave identically.
5. **Composition over inheritance** — a skill is data; resolution is a pure function.
6. **Deterministic** — declared order preserved, tools deduped order-stable, precedence explicit.

## The model — `Skill` (himmy/skills/models.py)

```text
Skill(BaseModel, extra="forbid")
  name:            str                      # stable id
  version:         int = 1
  description:     str = ""                 # one-liner (shown by `himmy skills`)
  instructions:    list[str] = []           # the prompt fragment / know-how
  tools:           list[str] = []           # tool names this skill needs
  tool_packs:      list[str] = []           # whole packs to pull in (reuses resolve_packs)
  guardrails:      list[str] = []           # merged into the agent's guardrail set
  when_to_use:     str = ""                 # routing hint (feeds tool_router later)
  examples:        list[SkillExample] = []  # few-shot (Tier 3)
  requires_skills: list[str] = []           # composition (cycle-guarded)
  metadata:        dict = {}

  to_record(version, metadata) -> EntityRecord(kind="skill", stable_id=name, payload=...)
```

`SkillExample(BaseModel)`: `{ input: str, action: str, note: str = "" }`.

## Resolution — `resolve_skills(names, registry) -> ResolvedSkills` (himmy/skills/resolve.py)

Pure, registry-only (no ToolRegistry needed):

- look up each name → `UnknownSkillError` (difflib did-you-mean, like our SQL hint) on miss;
- DFS-expand `requires_skills`, dedup, **cycle detection** → `CyclicSkillError(path)`;
- aggregate into `ResolvedSkills`: ordered `instruction_blocks` (one labeled block per skill),
  order-stable `tools`, `tool_packs`, `guardrails`, `examples`, and the contributing `skills`
  (names, for the prompt's skills section + entity lineage).

Tool-existence validation is a **second phase** in `_build_runtime_for` (where the live
`ToolRegistry` exists): every `ResolvedSkills.tools` name must resolve, else a clear
`SkillToolError` naming the skill + missing tool.

## Registry & catalog (himmy/skills/registry.py, builtin/)

- `SkillRegistry` — `name -> Skill`, `register`, `get`, `list`, optional `entity_registry`
  for auto-projection (mirrors `ToolRegistry`).
- `BUILTIN_SKILLS: dict[str, Skill]` loaded from `himmy/skills/builtin/*.yaml`.
- Project-local discovery: `skills/*.yaml` under cwd (path from `ToolkitConfig`/`HIMMY_SKILLS_PATH`).
- **Precedence**: project-local overrides built-in by name; collision → deterministic
  (project wins) + a one-line warning.

---

## Tiers

### Tier 0 — Foundation (pure, no wiring)
New package `himmy/skills/`: `Skill` + `SkillExample` models, `SkillRegistry`,
`UnknownSkillError`/`CyclicSkillError`/`SkillToolError`, `resolve_skills` + `ResolvedSkills`,
`.to_record()` projection, `BUILTIN_SKILLS` (empty/seed). Fully unit-tested in isolation
(model validation, did-you-mean, cycle detection, ordering/dedup, entity projection).
**Deliverable:** skills are a typed, validated, resolvable, versioned entity — usable from
Python — before any agent touches them.

### Tier 1 — AgentSpec + runtime wiring (the payoff)
- Add `skills: list[str] = []` to `AgentSpec`.
- `apply_skills(spec, skill_registry) -> AgentSpec` (pure): expand skills → append
  `instruction_blocks` to `instructions`, union `tools`/`tool_packs`, merge `guardrails`,
  carry skill names for the prompt's skills section.
- `_build_runtime_for`: call `apply_skills`, then validate resolved tools against the built
  `ToolRegistry` (`SkillToolError`).
- `make_task`: put resolved skill names into `context["skills"]` so the existing
  `_render_prompts` path renders them; richer per-skill instructions ride through `instructions`.
- **Verify end-to-end:** an `agent.yaml` with `skills: [web_research]` binds web_search/web_fetch
  and gets the guidance — on the stub **and** on Ollama.

### Tier 2 — Authoring & discovery (the "easy")
- `Skill.from_yaml` + directory loader; `skills/*.yaml` auto-discovery + precedence rules.
- `himmy skills` CLI (mirror `cmd_tools`): list skills, their tools, description.
- `himmy init` scaffolds an example `skills/example.yaml`.
- Seed 4–6 high-value built-in skills (e.g. `web_research`, `data_analysis`, `file_ops`,
  `summarize`, `nepal_brief`). README + CHANGELOG sections.

### Tier 3 — Composition & few-shot
- `requires_skills` recursive composition surfaced in resolution (cycle-guarded already).
- `examples` injected as few-shot into the task/system prompt (new prompt section).
- `when_to_use` fed into `select_tools`/`tool_router` so skills inform routing.

### Tier 4 — Advanced (optional)
- **Skill-as-subagent**: invoke a skill as an isolated sub-run bound to only its tools
  (reuses spawn machinery) — a dispatchable capability, not just prompt shaping.
- **Versioning surfaced**: `himmy skills --history <name>` via the entity registry trace.
- **Skill-level eval**: a skill suite for the Tier 3.8 benchmark harness (benchmark a skill
  in isolation, measure tool-binding + accuracy).

## Cross-cutting
- Each tier is **CI-mirror-gated** (fresh venv, ruff+format+mypy+pytest) and committed
  separately as `nlethetech` (no attribution).
- `himmy/__init__.py` lazily exports `Skill`/`SkillRegistry`; `himmy doctor`/`tools`/`skills`
  list the new surface. No new dependencies. Offline-first preserved.
- Test blast radius is additive; the one touch-point is `_render_prompts` already reading
  `ctx["skills"]`, so Tier 1 slots in without changing existing behavior when `skills` is empty.
