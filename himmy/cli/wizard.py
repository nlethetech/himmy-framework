"""Interactive ``himmy init`` wizard: a short conversation instead of a YAML dump.

When init runs on a real terminal (both stdin and stdout are TTYs) it asks five
quick questions — name, job, provider, tool packs, memory — and writes ONE minimal
``agent.yaml`` that is guaranteed to load (the answers round-trip through
:class:`~himmy.config.agent_spec.AgentSpec` before anything touches disk).

Providers are not listed from a hardcoded menu: the wizard probes the machine the
same way ``himmy doctor`` does (``claude``/``ollama`` on PATH, provider keys in the
environment) and offers only what will actually work, with the offline stub as the
always-available last resort.

Non-interactive callers (pipes, CI, ``--classic``, ``HIMMY_NO_WIZARD=1``) never see
the wizard; ``himmy init`` keeps writing the classic example scaffold for them.
"""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _eprint(*args: Any) -> None:
    """Print to stderr (the wizard's questions are UI, stdout stays for results)."""
    print(*args, file=sys.stderr)


def wizard_available() -> bool:
    """True when init may go interactive: both ends are TTYs and not opted out."""
    if os.environ.get("HIMMY_NO_WIZARD"):
        return False
    return bool(
        getattr(sys.stdin, "isatty", lambda: False)()
        and getattr(sys.stdout, "isatty", lambda: False)()
    )


@dataclass
class ProviderChoice:
    """One offerable inference backend: the YAML value plus how to present it."""

    key: str  # agent.yaml `provider:` value
    label: str  # menu line shown to the user
    model: str | None = None  # default model to write; None → omit (provider default)
    ask_model: bool = False  # offer editing the model name


def detect_provider_choices() -> list[ProviderChoice]:
    """Probe the machine and return providers that will work, best first.

    Mirrors the ``himmy doctor`` probes (PATH binaries + env keys) so the wizard
    never offers a backend that would fail on first run. The stub is always last.
    """
    choices: list[ProviderChoice] = []
    if shutil.which("claude"):
        choices.append(
            ProviderChoice(
                key="claude-cli",
                label="claude-cli — your local Claude CLI (included in a Claude plan)",
                model="sonnet",
                ask_model=True,
            )
        )
    if shutil.which("ollama"):
        from himmy.cli.provider import _default_ollama_model

        choices.append(
            ProviderChoice(
                key="ollama",
                label="ollama — local open models, fully offline",
                model=_default_ollama_model(),
                ask_model=True,
            )
        )
    for key_env, provider in (
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
        ("OPENROUTER_API_KEY", "openrouter"),
    ):
        if os.environ.get(key_env):
            choices.append(
                ProviderChoice(
                    key=provider,
                    label=f"{provider} — API key found in {key_env}",
                    ask_model=True,
                )
            )
    choices.append(
        ProviderChoice(
            key="stub",
            label="stub — offline deterministic stand-in (no real model; plumbing only)",
        )
    )
    return choices


# ----------------------------------------------------------------- ask helpers


def _ask(question: str, default: str = "", *, input_fn: Callable[[str], str]) -> str:
    """One free-text question; empty answer returns ``default``."""
    suffix = f" [{default}]" if default else ""
    answer = input_fn(f"{question}{suffix}: ").strip()
    return answer or default


def _ask_yes_no(
    question: str, *, default: bool = False, input_fn: Callable[[str], str]
) -> bool:
    """A y/N question (default capitalized)."""
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input_fn(f"{question} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes")


def _ask_choice(
    question: str,
    labels: list[str],
    *,
    default_index: int = 0,
    input_fn: Callable[[str], str],
) -> int:
    """A numbered menu; returns the chosen index (empty answer → default)."""
    _eprint(f"\n{question}")
    for i, label in enumerate(labels, start=1):
        marker = "*" if i - 1 == default_index else " "
        _eprint(f"  {marker} {i}) {label}")
    while True:
        answer = input_fn(f"choice [{default_index + 1}]: ").strip()
        if not answer:
            return default_index
        if answer.isdigit() and 1 <= int(answer) <= len(labels):
            return int(answer) - 1
        _eprint(f"  please answer 1-{len(labels)}")


def _ask_tool_packs(*, input_fn: Callable[[str], str]) -> list[str]:
    """Offer the built-in packs by name/number; returns validated pack names."""
    from himmy.toolkit import BUILTIN_PACKS

    packs = list(BUILTIN_PACKS.values())
    _eprint("\ntool packs (capabilities your agent can use):")
    for i, pack in enumerate(packs, start=1):
        _eprint(f"  {i:>2}) {pack.name:<13} {pack.description}")
    while True:
        answer = input_fn(
            "packs — names or numbers, comma-separated; empty for none [web, utils]: "
        ).strip()
        if not answer:
            return ["web", "utils"]
        if answer.lower() in ("none", "-"):
            return []
        names: list[str] = []
        bad: list[str] = []
        for token in answer.replace(",", " ").split():
            if token.isdigit() and 1 <= int(token) <= len(packs):
                names.append(packs[int(token) - 1].name)
            elif token in BUILTIN_PACKS:
                names.append(token)
            else:
                bad.append(token)
        if bad:
            _eprint(f"  unknown pack(s): {', '.join(bad)} — try names or numbers")
            continue
        # de-dup, preserving order
        return list(dict.fromkeys(names))


# ------------------------------------------------------------------ the wizard


def _spec_to_yaml(data: dict[str, Any], *, created_by: str = "himmy init") -> str:
    """Render the answers as a clean agent.yaml with a short orienting header."""
    import yaml

    body = yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88)
    return (
        f"# Agent spec created by `{created_by}`.\n"
        "# Reference: run `himmy tools` / `himmy skills` for capabilities, or see\n"
        "# the full annotated example via `himmy init --classic` in an empty dir.\n"
        + body
    )


def run_wizard(
    target: Path,
    *,
    force: bool = False,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Ask, validate, write ``agent.yaml`` into ``target``. Returns an exit code."""
    dest = target / "agent.yaml"
    try:
        _eprint("himmy init — let's set up your agent (Ctrl-C to abort)\n")

        if dest.exists() and not force:
            if not _ask_yes_no(
                f"{dest} already exists — overwrite it?", input_fn=input_fn
            ):
                _eprint("aborted (nothing written)")
                return 1

        default_name = target.resolve().name or "my-agent"
        name = _ask("agent name", default_name, input_fn=input_fn)
        description = _ask(
            "what should it do? (one line)",
            "A helpful assistant.",
            input_fn=input_fn,
        )

        choices = detect_provider_choices()
        idx = _ask_choice(
            "which model should it run on? (detected on this machine)",
            [c.label for c in choices],
            input_fn=input_fn,
        )
        chosen = choices[idx]
        model = chosen.model
        if chosen.ask_model:
            model = _ask("model", chosen.model or "default", input_fn=input_fn) or None

        packs = _ask_tool_packs(input_fn=input_fn)
        memory = _ask_yes_no(
            "\nremember durable facts across runs (long-term memory)?",
            input_fn=input_fn,
        )
    except (KeyboardInterrupt, EOFError):
        _eprint("\naborted (nothing written)")
        return 130

    data: dict[str, Any] = {
        "name": name,
        "description": description,
        "instructions": [
            "Be concise and accurate.",
            "Use your tools rather than guessing.",
        ],
        "provider": chosen.key,
    }
    if model and model != "default":
        data["model"] = model
    if packs:
        data["tool_packs"] = packs
    if memory:
        data["memory"] = True

    # Round-trip through the real spec model so the wizard can never write a file
    # that `himmy run` would then refuse to load.
    from himmy.config.agent_spec import AgentSpec

    AgentSpec(**data)

    target.mkdir(parents=True, exist_ok=True)
    dest.write_text(_spec_to_yaml(data))
    print(f"wrote {dest}")

    _eprint(f'\nNext:\n  himmy run -f {dest} -p "hello"\n  himmy chat -f {dest}')
    if chosen.key == "stub":
        _eprint(
            "\n(the stub answers deterministically — when you're ready for a real "
            "model, run `himmy doctor` for options)"
        )
    return 0
