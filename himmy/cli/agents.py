"""``himmy agents`` and ``himmy validate``: see and lint the specs you own.

``agents`` answers "what agents live here?" — it scans a directory (default: cwd)
the way Himmy Studio does: explicit spec names (``agent.yaml`` / ``*.agent.yaml`` /
``team.yaml`` / ``*.team.yaml`` / ``agents/*.yaml`` / ``*/agent.yaml``) always
appear — broken ones flagged with the reason, since a listing that silently
skipped your typo'd spec would hide exactly the file you care about — while other
loose top-level ``*.yaml`` files are included only when their content actually
looks like an agent or team (so docker-compose noise stays out). ``--json`` for
scripts.

``validate`` answers "will this spec work?" before a run fails mid-flight: YAML
shape, unknown top-level keys (with did-you-mean), pydantic field errors, and
unknown tool packs / skills / providers, each as a one-line finding.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any

import yaml


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


# -------------------------------------------------------------------- agents


#: Filenames that explicitly claim to be specs (mirrors Himmy Studio's scan) —
#: these are always listed, broken or not. Other loose ``*.yaml`` files are only
#: listed when their content actually looks like an agent or a team.
_EXPLICIT_PATTERNS = (
    "agent.yaml",
    "*.agent.yaml",
    "team.yaml",
    "*.team.yaml",
    "*/agent.yaml",
    "agents/*.yaml",
)


def _spec_files(directory: Path) -> tuple[list[Path], list[Path]]:
    """(explicit spec files, other loose top-level *.yaml) in stable order."""
    explicit = {
        p.resolve()
        for pattern in _EXPLICIT_PATTERNS
        for p in directory.glob(pattern)
        if p.is_file()
    }
    loose = {p.resolve() for p in directory.glob("*.yaml") if p.is_file()} - explicit
    return sorted(explicit), sorted(loose)


def _describe_file(path: Path) -> dict[str, Any]:
    """One lenient summary entry per spec file (never raises)."""
    entry: dict[str, Any] = {"file": str(path)}
    try:
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict):
            raise ValueError("not a YAML mapping")
        if "members" in raw or "entry" in raw:
            members = raw.get("members") or []
            entry.update(
                kind="team",
                name=raw.get("name") or path.stem,
                detail=f"{len(members)} member(s), entry: {raw.get('entry', '?')}",
            )
        elif "name" not in raw:
            raise ValueError("no `name` — not an agent spec")
        else:
            from himmy.config.agent_spec import AgentSpec

            spec = AgentSpec(
                **{k: v for k, v in raw.items() if k in AgentSpec.model_fields}
            )
            caps = ", ".join((*spec.tool_packs, *spec.skills)) or "no tools"
            entry.update(
                kind="agent",
                name=spec.name,
                detail=f"{spec.provider or 'auto'}/{spec.model} · {caps}",
            )
    except Exception as exc:  # noqa: BLE001 - a broken spec is a finding, not a crash
        entry.update(kind="broken", name=path.stem, detail=str(exc).split("\n")[0])
    return entry


def cmd_agents(args: argparse.Namespace) -> int:
    """List the agent/team specs in a directory (default: cwd)."""
    directory = Path(getattr(args, "directory", None) or ".").expanduser()
    if not directory.is_dir():
        _eprint(f"error: {directory} is not a directory")
        return 1
    explicit, loose = _spec_files(directory)
    entries = [_describe_file(p) for p in explicit]
    # A docker-compose.yml or CI config parsing as "broken agent" would be noise:
    # loose files only count when they genuinely look like a spec.
    entries += [
        e for p in loose if (e := _describe_file(p))["kind"] in ("agent", "team")
    ]
    entries.sort(key=lambda e: e["file"])

    if getattr(args, "json", False):
        print(json.dumps(entries, indent=2, ensure_ascii=False))
        return 0

    if not entries:
        print(f"no agent specs in {directory.resolve()}")
        _eprint('\ncreate one:  himmy init   ·   himmy new "what it should do"')
        return 0

    from himmy.cli.ui import styles

    c = styles(sys.stdout)
    badge_style = {"agent": c["green"], "team": c["gold"], "broken": c["crimson"]}
    for e in entries:
        flag = {"agent": "agent", "team": "team ", "broken": "BROKEN"}[e["kind"]]
        print(
            f"  {badge_style[e['kind']]}[{flag}]{c['reset']} "
            f"{c['bold']}{c['snow']}{e['name']:<24}{c['reset']} "
            f"{c['dim']}{e['detail']}{c['reset']}"
        )
        print(f"          {c['faint']}{e['file']}{c['reset']}")
    return 0


# ------------------------------------------------------------------ validate


def _findings_for(path: Path) -> list[str]:
    """All problems found in one spec file, as user-facing one-liners."""
    from himmy.cli.provider import PROVIDERS
    from himmy.config.agent_spec import AgentSpec
    from himmy.skills import build_skill_registry
    from himmy.toolkit import BUILTIN_PACKS

    if not path.is_file():
        return [f"file not found: {path}"]
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        return [f"not valid YAML: {str(exc).splitlines()[0]}"]
    if not isinstance(raw, dict):
        return ["spec must be a YAML mapping (key: value pairs)"]
    if "members" in raw or "entry" in raw:
        return ["this looks like a team.yaml — validate runs on single-agent specs"]

    findings: list[str] = []
    known = set(AgentSpec.model_fields)
    for key in raw:
        if key not in known:
            close = difflib.get_close_matches(key, known, n=1)
            hint = f" — did you mean {close[0]!r}?" if close else ""
            findings.append(f"unknown key {key!r}{hint}")

    try:
        spec = AgentSpec(**{k: v for k, v in raw.items() if k in known})
    except Exception as exc:  # noqa: BLE001 - pydantic message → findings
        for line in str(exc).splitlines():
            line = line.strip()
            if line and not line.startswith(("For further", "Traceback")):
                findings.append(line)
        return findings

    for pack in spec.tool_packs:
        if pack not in BUILTIN_PACKS:
            close = difflib.get_close_matches(pack, BUILTIN_PACKS, n=1)
            hint = f" — did you mean {close[0]!r}?" if close else " (see `himmy tools`)"
            findings.append(f"unknown tool pack {pack!r}{hint}")
    skill_names = set(build_skill_registry().names())
    for skill in spec.skills:
        if skill not in skill_names:
            close = difflib.get_close_matches(skill, skill_names, n=1)
            hint = (
                f" — did you mean {close[0]!r}?" if close else " (see `himmy skills`)"
            )
            findings.append(f"unknown skill {skill!r}{hint}")
    if spec.provider is not None and spec.provider not in PROVIDERS:
        close = difflib.get_close_matches(spec.provider, PROVIDERS, n=1)
        hint = f" — did you mean {close[0]!r}?" if close else ""
        findings.append(f"unknown provider {spec.provider!r}{hint}")
    return findings


def cmd_validate(args: argparse.Namespace) -> int:
    """Lint a spec file (or the discovered agent.yaml); exit 1 on findings."""
    target = getattr(args, "file", None)
    if target is None:
        from himmy.cli.commands import _discover_spec_file

        discovered = _discover_spec_file()
        if discovered is None:
            _eprint(
                "error: no agent.yaml found here (or above) — "
                "pass a file: himmy validate path/to/agent.yaml"
            )
            return 2
        target = str(discovered)

    from himmy.cli.ui import styles

    c = styles(sys.stdout)
    path = Path(target).expanduser()
    findings = _findings_for(path)
    if not findings:
        print(f"{c['green']}OK:{c['reset']} {path} is a valid agent spec")
        return 0
    print(f"{c['snow']}{path}{c['reset']}: {len(findings)} problem(s)")
    for f in findings:
        print(f"  {c['crimson']}-{c['reset']} {f}")
    return 1
