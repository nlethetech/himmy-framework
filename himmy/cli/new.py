"""``himmy new``: draft an agent.yaml from a plain-English description.

The user describes the agent ("watches RSS feeds for NEPSE news and keeps a task
list"); their own model — the best backend detected on this machine, same probe as
the init wizard — writes the spec: instructions, tool packs, skills, memory. The
draft is forced through a whitelist + :class:`~himmy.config.agent_spec.AgentSpec`
validation, so a hallucinated key or pack can never reach disk. The provider/model
in the written spec are NOT model-chosen either: they are pinned to the backend
that actually exists here.

On a TTY the rendered YAML is shown and confirmed before writing; ``--yes`` and
non-interactive callers skip the question; ``--dry-run`` prints the YAML to stdout
and writes nothing (pipe it wherever you like).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from himmy.core import HimmyError

#: Keys the drafting model may set. Everything else (provider, mcp_servers,
#: tools_module, ...) is the machine's or the user's call, not the model's.
_DRAFT_KEYS = (
    "name",
    "description",
    "role",
    "instructions",
    "tool_packs",
    "skills",
    "memory",
    "temperature",
)


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


def _capabilities_block() -> str:
    """The tool packs + skills the draft may reference, as prompt context."""
    from himmy.skills import build_skill_registry
    from himmy.toolkit import BUILTIN_PACKS

    packs = "\n".join(f"- {p.name}: {p.description}" for p in BUILTIN_PACKS.values())
    skills = ", ".join(build_skill_registry().names()) or "(none)"
    return f"Available tool_packs:\n{packs}\n\nAvailable skills: {skills}"


def _draft_system_prompt() -> str:
    """System prompt: the schema contract the drafting model must honor."""
    return (
        "You design agent specs for the himmy framework. Given a description of "
        "an agent, reply with ONE JSON object and nothing else — no markdown "
        "fences, no commentary.\n\n"
        "Allowed keys (all optional except name/description/instructions):\n"
        "  name (short kebab-case string)\n"
        "  description (one line)\n"
        "  role (two or three words, e.g. 'Research Assistant')\n"
        "  instructions (list of 2-5 imperative strings: how it should behave)\n"
        "  tool_packs (list — ONLY packs it genuinely needs, from the list below)\n"
        "  skills (list — only from the available skills, often empty)\n"
        "  memory (boolean — true only if remembering across runs helps)\n"
        "  temperature (number 0-1, omit unless creativity matters)\n\n"
        + _capabilities_block()
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model reply (tolerating fences)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise HimmyError(f"the model did not return JSON (got: {text[:120]!r})")
    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise HimmyError(f"could not parse the model's JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise HimmyError("the model returned JSON that is not an object")
    return data


def _sanitize_draft(raw: dict[str, Any], provider_key: str) -> dict[str, Any]:
    """Whitelist keys, drop unknown packs/skills, pin the provider."""
    from himmy.skills import build_skill_registry
    from himmy.toolkit import BUILTIN_PACKS

    data: dict[str, Any] = {k: raw[k] for k in _DRAFT_KEYS if k in raw}
    if isinstance(data.get("tool_packs"), list):
        data["tool_packs"] = [p for p in data["tool_packs"] if p in BUILTIN_PACKS]
    known_skills = set(build_skill_registry().names())
    if isinstance(data.get("skills"), list):
        data["skills"] = [s for s in data["skills"] if s in known_skills]
    data = {k: v for k, v in data.items() if v not in (None, [], "")}
    data.setdefault("name", "my-agent")
    data.setdefault("description", "A helpful assistant.")
    data["provider"] = provider_key
    return data


def _complete(provider: str, model: str | None, system: str, user: str) -> str:
    """One blocking completion on the chosen backend; returns the reply text."""
    import asyncio

    from himmy.cli.provider import build_inference_for
    from himmy.services.inference.models import InferenceMessage, InferenceRequest

    service = build_inference_for(provider, model)
    request = InferenceRequest(
        messages=[
            InferenceMessage(role="system", content=system),
            InferenceMessage(role="user", content=user),
        ],
        timeout_seconds=120.0,
    )
    response = asyncio.run(service.run(request))
    if response.error is not None or not response.output_text:
        detail = response.error.message if response.error else "empty reply"
        raise HimmyError(f"drafting call failed on {provider}: {detail}")
    return response.output_text


def draft_agent_dict(
    description: str, provider: str, model: str | None
) -> dict[str, Any]:
    """Description → validated spec dict (one retry when the first draft is bad)."""
    from himmy.config.agent_spec import AgentSpec

    system = _draft_system_prompt()
    user = f"Design an agent for this job:\n\n{description}"
    last_error: Exception | None = None
    for _attempt in range(2):
        reply = _complete(provider, model, system, user)
        try:
            data = _sanitize_draft(_extract_json(reply), provider)
            AgentSpec(**data)  # never write a spec `himmy run` would refuse
            return data
        except (HimmyError, ValueError) as exc:
            last_error = exc
            user = (
                f"Design an agent for this job:\n\n{description}\n\n"
                f"Your previous reply was invalid ({exc}). "
                "Reply with ONLY the corrected JSON object."
            )
    raise HimmyError(f"could not draft a valid spec: {last_error}")


def cmd_new(args: argparse.Namespace) -> int:
    """Draft, show, confirm, write. The CLI face of :func:`draft_agent_dict`."""
    from himmy.cli.wizard import _ask_yes_no, _spec_to_yaml, detect_provider_choices

    description = " ".join(args.description).strip()
    if not description:
        _eprint('error: describe the agent, e.g. himmy new "summarizes my RSS feeds"')
        return 2

    # The same probe the wizard uses: draft with the best real backend available.
    choice = detect_provider_choices()[0]
    provider = getattr(args, "provider", None) or choice.key
    model = getattr(args, "model", None) or choice.model
    if provider == "stub":
        _eprint(
            "error: drafting needs a real model (claude CLI, ollama, or an API "
            "key). Run `himmy doctor` to see what's missing."
        )
        return 1

    _eprint(f"drafting with {provider}{f' ({model})' if model else ''} …")
    try:
        data = draft_agent_dict(description, provider, model)
    except HimmyError as exc:
        _eprint(f"error: {exc}")
        return 1

    yaml_text = _spec_to_yaml(data, created_by="himmy new")
    if getattr(args, "dry_run", False):
        print(yaml_text, end="")
        return 0

    dest = Path(getattr(args, "output", None) or "agent.yaml").expanduser()
    _eprint(f"\n--- {dest} ---")
    _eprint(yaml_text)
    interactive = bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )
    if dest.exists() and not getattr(args, "yes", False):
        if not interactive:
            _eprint(f"error: {dest} already exists (use --yes to overwrite)")
            return 1
        if not _ask_yes_no(f"{dest} exists — overwrite?", input_fn=input):
            _eprint("aborted (nothing written)")
            return 1
    elif interactive and not getattr(args, "yes", False):
        if not _ask_yes_no(f"write {dest}?", default=True, input_fn=input):
            _eprint("aborted (nothing written)")
            return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml_text)
    print(f"wrote {dest}")
    _eprint(f'\nNext:\n  himmy run -f {dest} -p "hello"\n  himmy chat -f {dest}')
    return 0
