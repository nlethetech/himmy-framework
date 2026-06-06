"""Himmy Studio agent authoring: list capabilities, read/validate/save agent specs.

Powers the GUI's no-code agent builder. Saving is a *merge* onto the existing file,
so fields the form doesn't expose (http_tools, mcp_servers, tools_module, …) survive
an edit. Validation is explicit (the underlying ``AgentSpec`` is permissive), so a
bad skill/pack name or an empty agent name is a clean 400, not a silent runtime gap.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from himmy.api.studio_service import AgentSummary, _summarize, project_root

# Fields the GUI form may set. Anything else in an existing file is preserved.
EDITABLE_FIELDS = (
    "name",
    "description",
    "role",
    "instructions",
    "model",
    "provider",
    "temperature",
    "max_tokens",
    "skills",
    "tools",
    "tool_packs",
    "knowledge",
    "guardrails",
    "memory",
    "memory_top_k",
    "language",
    "tool_router",
    "allow_spawn",
    "allow_skill_dispatch",
)

# Preferred key order when writing YAML (readability); unknown keys appended after.
_WRITE_ORDER = (
    "name",
    "description",
    "role",
    "instructions",
    "provider",
    "model",
    "skills",
    "tool_packs",
    "tools",
    "knowledge",
    "guardrails",
    "memory",
    "memory_top_k",
    "language",
    "temperature",
    "max_tokens",
    "tool_router",
    "allow_spawn",
    "allow_skill_dispatch",
)

# Values dropped when writing (defaults / empties) to keep the file tidy. name and
# description are always kept.
_DEFAULTS = {
    "model": "default",
    "language": "en",
    "memory": False,
    "memory_top_k": 5,
    "tool_router": False,
    "allow_spawn": False,
    "allow_skill_dispatch": False,
    "role": None,
    "provider": None,
    "temperature": None,
    "max_tokens": None,
}


class PackInfo(BaseModel):
    name: str
    description: str
    tools: list[str]


class SkillInfo(BaseModel):
    name: str
    description: str
    when_to_use: str | None = None
    tool_packs: list[str] = []
    tools: list[str] = []
    requires_skills: list[str] = []
    builtin: bool = True


class AgentDetail(BaseModel):
    """The full editable spec for one agent, plus its path."""

    path: str
    spec: dict[str, Any]
    has_advanced: bool = False  # carries fields the form doesn't expose


class SaveAgentRequest(BaseModel):
    """Create/update an agent. ``path`` is relative to the project root.

    ``overwrite`` must be true to write over an existing file — the GUI sends it for
    an intentional edit, but withholds it when creating a new agent so a name that
    collides with an existing file is a 409, not a silent clobber.
    """

    path: str
    spec: dict[str, Any]
    overwrite: bool = False


class ValidationResult(BaseModel):
    ok: bool
    errors: list[str] = []


def list_tool_packs() -> list[PackInfo]:
    """The built-in tool packs (name, description, tools)."""
    from himmy.toolkit import BUILTIN_PACKS

    return [
        PackInfo(name=p.name, description=p.description, tools=list(p.tool_names))
        for p in BUILTIN_PACKS.values()
    ]


def list_skill_infos() -> list[SkillInfo]:
    """Available skills (built-in + project-local)."""
    from himmy.skills import BUILTIN_SKILLS, build_skill_registry

    registry = build_skill_registry()
    out: list[SkillInfo] = []
    for s in registry.list():
        out.append(
            SkillInfo(
                name=s.name,
                description=s.description,
                when_to_use=s.when_to_use,
                tool_packs=list(s.tool_packs),
                tools=list(s.tools),
                requires_skills=list(s.requires_skills),
                builtin=s.name in BUILTIN_SKILLS,
            )
        )
    return out


def _resolve_writable_path(rel_path: str, root: Path | None = None) -> Path:
    """Resolve a save path safely under the project root (must be a .yaml)."""
    root = (root or project_root()).resolve()
    if not rel_path or rel_path.startswith("/") or ".." in Path(rel_path).parts:
        raise ValueError(f"invalid agent path: {rel_path!r}")
    if not rel_path.endswith((".yaml", ".yml")):
        raise ValueError("agent path must end in .yaml")
    target = (root / rel_path).resolve()
    if root != target.parent and root not in target.parents:
        raise ValueError(f"path escapes project root: {rel_path!r}")
    return target


def load_agent_detail(rel_path: str, root: Path | None = None) -> AgentDetail:
    """Load an agent file into its full editable field set."""
    from himmy.api.studio_service import resolve_spec_path
    from himmy.config.agent_spec import load_agent_spec

    path = resolve_spec_path(rel_path, root)
    spec = load_agent_spec(str(path))
    full = spec.model_dump(mode="json", exclude_none=False)
    has_advanced = bool(
        full.get("http_tools")
        or full.get("mcp_servers")
        or full.get("tools_module")
        or full.get("output_schema")
    )
    # Return only the editable fields to the form; advanced ones are preserved on save.
    editable = {k: full.get(k) for k in EDITABLE_FIELDS if k in full}
    return AgentDetail(
        path=str(path.relative_to((root or project_root()).resolve())),
        spec=editable,
        has_advanced=has_advanced,
    )


def validate_spec(data: dict[str, Any]) -> list[str]:
    """Return human-readable validation errors for a proposed spec (empty = valid)."""
    from himmy.config.agent_spec import AgentSpec
    from himmy.services.guardrails import BUILTIN_GUARDRAILS
    from himmy.skills import build_skill_registry
    from himmy.toolkit import BUILTIN_PACKS

    errors: list[str] = []
    name = str(data.get("name") or "").strip()
    if not name:
        errors.append("Name is required.")

    known_packs = set(BUILTIN_PACKS)
    for p in data.get("tool_packs") or []:
        if p not in known_packs:
            errors.append(f"Unknown tool pack: {p!r}")

    known_skills = set(build_skill_registry().names())
    for s in data.get("skills") or []:
        if s not in known_skills:
            errors.append(f"Unknown skill: {s!r}")

    known_guards = set(BUILTIN_GUARDRAILS)
    for g in data.get("guardrails") or []:
        if g not in known_guards:
            errors.append(f"Unknown guardrail: {g!r}")

    # Structural validation via the model (catches type errors).
    try:
        AgentSpec(**data)
    except Exception as exc:  # noqa: BLE001 - surface pydantic messages to the GUI
        errors.append(f"Spec error: {exc}")

    return errors


def _clean_for_write(merged: dict[str, Any]) -> dict[str, Any]:
    """Drop default/empty fields and order keys for a tidy YAML file."""
    out: dict[str, Any] = {}
    keys = list(_WRITE_ORDER) + [k for k in merged if k not in _WRITE_ORDER]
    for k in keys:
        if k not in merged:
            continue
        v = merged[k]
        if k in ("name", "description"):
            out[k] = v
            continue
        if v is None or v == [] or v == {} or v == "":
            continue
        if k in _DEFAULTS and v == _DEFAULTS[k]:
            continue
        out[k] = v
    return out


def save_agent(
    rel_path: str,
    form: dict[str, Any],
    root: Path | None = None,
    *,
    overwrite: bool = False,
) -> AgentSummary:
    """Validate then write an agent.yaml, merging onto any existing file.

    Only :data:`EDITABLE_FIELDS` from ``form`` are applied, so advanced fields in an
    existing file (http_tools, mcp_servers, …) are preserved. Raises ``ValueError``
    on a bad path, ``SpecInvalid`` on validation errors, and ``AgentExists`` when the
    target already exists and ``overwrite`` is false.
    """
    root = (root or project_root()).resolve()
    target = _resolve_writable_path(rel_path, root)

    existing: dict[str, Any] = {}
    if target.is_file():
        if not overwrite:
            raise AgentExists(rel_path)
        loaded = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            existing = loaded

    merged = dict(existing)
    for k in EDITABLE_FIELDS:
        if k in form:
            merged[k] = form[k]

    errors = validate_spec(merged)
    if errors:
        raise SpecInvalid(errors)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(_clean_for_write(merged), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    summary = _summarize(target, root)
    assert summary is not None  # we just wrote a valid spec
    return summary


class SpecInvalid(ValueError):
    """Raised when a proposed spec fails validation; carries the error list."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class AgentExists(ValueError):
    """Raised when saving a new agent would overwrite an existing file."""

    def __init__(self, path: str) -> None:
        super().__init__(f"{path} already exists")
        self.path = path


__all__ = [
    "PackInfo",
    "SkillInfo",
    "AgentDetail",
    "SaveAgentRequest",
    "ValidationResult",
    "SpecInvalid",
    "AgentExists",
    "list_tool_packs",
    "list_skill_infos",
    "load_agent_detail",
    "validate_spec",
    "save_agent",
    "EDITABLE_FIELDS",
]
