"""Run a named skill as a focused sub-agent from the CLI / REPL.

``himmy skills`` *lists* the catalog; this module *runs* one. ``himmy skill <name>
<prompt>`` resolves the named capability via the real skills registry, expands it into a
fresh :class:`AgentSpec` (its tool packs + injected know-how, through the same
:func:`~himmy.config.agent_spec.apply_skills` path that ``_spec_from_args`` uses), wires a
runtime on the CLI-selected provider, runs the prompt, and prints the answer. The REPL
``/skill <name> <task>`` slash does the same within a live session.

The skill is scoped: the sub-agent is bound to exactly that skill's tools and primed with
its know-how — nothing inherited from the parent agent. That makes it a focused,
auditable unit ("answer this with the ``data_analysis`` skill") rather than a free-form
delegation, and it never carries a ``dispatch_skill`` tool of its own, so a dispatched
skill cannot fan out again.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import sys
from typing import Any

from himmy.cli.commands import _answer, _build_runtime_for, _eprint, _model_label
from himmy.config.agent_spec import AgentSpec, apply_skills
from himmy.runtime import from_spec


def _skill_registry() -> Any:
    """The built-in catalog with project-local skills overlaid (the real loader)."""
    from himmy.skills import build_skill_registry

    return build_skill_registry()


def _unknown_skill_msg(name: str, registry: Any) -> str:
    """A helpful 'unknown skill' line with a did-you-mean and the known names."""
    known = registry.names()
    close = difflib.get_close_matches(name, known, n=1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    listing = ", ".join(known) if known else "(none)"
    return f"error: unknown skill {name!r}{hint}\n  known skills: {listing}"


def build_skill_spec(name: str, args: argparse.Namespace, registry: Any) -> AgentSpec:
    """A focused :class:`AgentSpec` that *is* the named skill, ready to wire.

    Starts from a bare spec carrying only ``skills=[name]``, folds in project defaults
    and the CLI ``--provider``/``--model`` overrides, then expands the skill through the
    real :func:`apply_skills` (the same path ``_spec_from_args`` uses) so the skill's
    tool packs + know-how + guardrails land in the spec exactly as a declared agent
    would get them. Raises nothing for an unknown name — the caller checks membership
    first so it can print a friendly hint.
    """
    spec = AgentSpec(
        name=f"skill:{name}",
        description="",
        skills=[name],
        # A skill's whole point is to wield its tools; route only when the skill binds
        # a large toolset (apply_skills leaves tool_router at its adaptive default).
    )
    spec = from_spec.apply_project_defaults(spec)
    provider = getattr(args, "provider", None)
    model = getattr(args, "model", None)
    if provider:
        spec.provider = provider
    if model and model != "default":
        spec.model = model
    return apply_skills(spec, registry)


async def _run_skill(
    spec: AgentSpec, args: argparse.Namespace, prompt: str, *, on_event: Any = None
) -> Any:
    """Wire a runtime for the skill spec and produce a final answer (RunResult-shaped).

    A skill with tool packs runs the bounded agent loop (act -> observe -> answer); a
    pure know-how skill (no tools) is a single inference call. Returns ``run_agent_loop``
    final turn or the ``run_task_detailed`` result — both expose ``output_text``/``status``.

    RBAC is enforced consistently with ``run``/``chat``: a restricted ``--role`` re-gates
    any gated tool it may not run (fail closed) before the loop starts.
    """
    from himmy.cli.rbac_cmd import enforce_role_on_registry

    runtime, registry = _build_runtime_for(spec, args, on_event=on_event)
    enforce_role_on_registry(
        registry, role=getattr(args, "role", None), on_line=_eprint
    )
    return await _answer(
        runtime,
        spec.to_persona(),
        spec.make_task(prompt),
        llm_config=spec.to_llm_config(),
        has_tools=registry is not None,
        route_tools=spec.tool_router,
    )


def cmd_skill(args: argparse.Namespace) -> int:
    """``himmy skill <name> <prompt...>`` — run one skill as a focused sub-agent.

    The prompt is the positional words after the skill name (joined), or ``-p/--prompt``.
    Prints the skill's answer to stdout; diagnostics go to stderr. Non-TTY/CI safe:
    no spinner overlay, plain markdown, no prompts.
    """
    name = (getattr(args, "name", None) or "").strip()
    if not name:
        _eprint("error: give a skill name — `himmy skill <name> <prompt>`")
        return 2

    prompt = (
        getattr(args, "prompt", None) or " ".join(getattr(args, "words", None) or [])
    ).strip()
    if not prompt:
        _eprint(
            f"error: give a prompt for the {name!r} skill — "
            f"positional words or -p/--prompt"
        )
        return 2

    registry = _skill_registry()
    if name not in registry:
        _eprint(_unknown_skill_msg(name, registry))
        return 1

    spec = build_skill_spec(name, args, registry)
    applied = spec.metadata.get("skills", [name])
    binds = spec.tool_packs or ["(no tools)"]
    _eprint(f"skill {name} → applied {', '.join(applied)} · binds {', '.join(binds)}")

    from himmy.cli.ui import LiveRunUI, render_markdown_lite

    # Live overlay only on a real terminal; piped/CI runs stay plain (no hang, no ANSI).
    live = (
        LiveRunUI(model_label=_model_label(spec, args)) if sys.stderr.isatty() else None
    )
    on_event = live.handle if (live is not None and live.enabled) else None

    try:
        result = asyncio.run(_run_skill(spec, args, prompt, on_event=on_event))
    except Exception as exc:  # noqa: BLE001 - surface a clean error, no traceback to a user
        if live is not None:
            live.finish()
        _eprint(f"error: skill {name!r} failed: {exc}")
        return 1
    if live is not None:
        live.finish()

    answer = (getattr(result, "output_text", None) or "").strip()
    if not answer:
        _eprint(f"note: the {name!r} skill produced no answer")
    print(render_markdown_lite(answer, stream=sys.stdout))
    if live is not None:
        live.footer()
    return 0


# --------------------------------------------------------------------- REPL slash


def slash_skill(repl: Any, rest: list[str]) -> Any:
    """``/skill <name> <task>`` — run a named skill as a sub-agent in the session.

    ``repl`` is the live :class:`~himmy.cli.repl.ChatRepl` (we read ``repl.args`` for the
    selected provider/model and ``repl._c``/``repl._eprint`` for the look). Runs OUTSIDE
    the chat thread — a focused, isolated capability call whose answer is printed but not
    folded into the conversation, mirroring the one-level dispatch_skill contract. Returns
    nothing meaningful; callers ignore the return.
    """
    c = repl._c
    if not rest:
        repl._eprint(f"{c['dim']}usage: /skill <name> <task>{c['reset']}")
        return None
    name = rest[0].strip()
    task = " ".join(rest[1:]).strip()
    if not task:
        repl._eprint(f"{c['dim']}usage: /skill <name> <task>{c['reset']}")
        return None

    registry = _skill_registry()
    if name not in registry:
        repl._eprint(f"{c['crimson']}{_unknown_skill_msg(name, registry)}{c['reset']}")
        return None

    spec = build_skill_spec(name, repl.args, registry)
    binds = ", ".join(spec.tool_packs) or "(no tools)"
    repl._eprint(
        f"{c['gold']}⚙ skill {name}{c['reset']}{c['faint']} · {binds}{c['reset']}"
    )

    try:
        result = asyncio.run(_run_skill(spec, repl.args, task))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the REPL
        repl._eprint(f"{c['crimson']}skill {name!r} failed: {exc}{c['reset']}")
        return None

    from himmy.cli.ui import render_markdown_lite

    answer = (getattr(result, "output_text", None) or "").strip()
    if not answer:
        repl._eprint(f"{c['faint']}(the {name!r} skill produced no answer){c['reset']}")
        return None
    print(render_markdown_lite(answer, stream=sys.stdout))
    return None


__all__ = ["cmd_skill", "slash_skill", "build_skill_spec"]
