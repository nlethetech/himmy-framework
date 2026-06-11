"""Tests for ``himmy skill <name> <prompt>`` and the REPL ``/skill`` slash.

Exercises the real skills path: a named built-in skill is resolved through the live
:class:`~himmy.skills.registry.SkillRegistry` + :func:`apply_skills`, its tool packs are
bound into a runtime, and the prompt runs on the offline stub. No mocking of the framework
primitives — only the provider is the (real) deterministic stub.
"""

from __future__ import annotations

import argparse
from typing import Any

import pytest

from himmy.cli.skill_dispatch import (
    build_skill_spec,
    cmd_skill,
    slash_skill,
)


def _args(
    name: str, words: list[str] | None = None, **extra: Any
) -> argparse.Namespace:
    base: dict[str, Any] = {
        "name": name,
        "words": words or [],
        "prompt": None,
        "provider": "stub",
        "model": None,
    }
    base.update(extra)
    return argparse.Namespace(**base)


# --------------------------------------------------------------------- spec build


def test_build_skill_spec_binds_tool_pack() -> None:
    """A tool-backed skill expands into the spec's packs + know-how (real apply_skills)."""
    from himmy.skills import build_skill_registry

    reg = build_skill_registry()
    spec = build_skill_spec("data_analysis", _args("data_analysis"), reg)
    # The skill's pack flowed into the spec...
    assert "data" in spec.tool_packs
    # ...its know-how landed in the description...
    assert "sql_schema" in spec.description
    # ...and the contributing skill is recorded where the prompt renderer reads it.
    assert spec.metadata.get("skills") == ["data_analysis"]


def test_build_skill_spec_pure_knowhow_has_no_tools() -> None:
    """A know-how-only skill (summarize) binds no packs but still injects guidance."""
    from himmy.skills import build_skill_registry

    reg = build_skill_registry()
    spec = build_skill_spec("summarize", _args("summarize"), reg)
    assert spec.tool_packs == []
    assert "summarize" in spec.metadata.get("skills", [])


def test_dispatched_tool_skill_actually_binds_tools_in_runtime() -> None:
    """The runtime wired for a tool skill carries exactly that skill's tools."""
    from himmy.cli.commands import _build_runtime_for
    from himmy.skills import build_skill_registry

    reg = build_skill_registry()
    args = _args("data_analysis")
    spec = build_skill_spec("data_analysis", args, reg)
    _runtime, registry = _build_runtime_for(spec, args)
    assert registry is not None
    bound = {d.name for d in registry.list()}
    assert {"sql_query", "sql_schema"} <= bound


# --------------------------------------------------------------------- cmd_skill


def test_cmd_skill_runs_skill_and_prints_answer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`himmy skill summarize <prompt>` runs on the stub and prints an answer."""
    code = cmd_skill(_args("summarize", ["summarize", "the", "report"]))
    out = capsys.readouterr()
    assert code == 0
    # The stub echoes the task, proving the prompt reached the skill sub-agent.
    assert "the report" in out.out
    # And the diagnostic line names the applied skill on stderr (not mixed into stdout).
    assert "skill summarize" in out.err


def test_cmd_skill_prompt_via_flag(capsys: pytest.CaptureFixture[str]) -> None:
    """`-p/--prompt` supplies the prompt when there are no positional words."""
    code = cmd_skill(_args("summarize", [], prompt="condense this memo"))
    out = capsys.readouterr()
    assert code == 0
    assert "condense this memo" in out.out


def test_cmd_skill_enforces_rbac_role(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Finding 2: a skill run enforces the --role consistently with run/chat so a
    viewer can't slip a gated tool by switching entry point to `himmy skill`."""
    seen: dict[str, Any] = {}
    import himmy.cli.rbac_cmd as rbac_cmd

    real = rbac_cmd.enforce_role_on_registry

    def _spy(registry: Any, *, role: Any = None, **kw: Any) -> Any:
        seen["role"] = role
        seen["called"] = True
        return real(registry, role=role, **kw)

    monkeypatch.setattr(rbac_cmd, "enforce_role_on_registry", _spy)
    code = cmd_skill(_args("data_analysis", ["analyze", "x"], role="viewer"))
    assert code == 0
    assert seen.get("called") is True
    assert seen["role"] == "viewer"


# --------------------------------------------------------------------- edge cases


def test_cmd_skill_unknown_name_errors_with_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown skill exits 1 with a did-you-mean + the catalog, on stderr."""
    code = cmd_skill(_args("summarise", ["x"]))  # British spelling typo
    err = capsys.readouterr().err
    assert code == 1
    assert "unknown skill 'summarise'" in err
    assert "did you mean 'summarize'" in err


def test_cmd_skill_missing_prompt_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    """No prompt anywhere is a usage error (exit 2), not a run."""
    code = cmd_skill(_args("summarize", []))
    err = capsys.readouterr().err
    assert code == 2
    assert "prompt" in err


def test_cmd_skill_missing_name_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty skill name is a usage error."""
    code = cmd_skill(_args("", ["hello"]))
    assert code == 2
    assert "skill name" in capsys.readouterr().err


# --------------------------------------------------------------------- /skill slash


class _FakeRepl:
    """Minimal stand-in for ChatRepl: the attributes slash_skill actually reads."""

    def __init__(self) -> None:
        from himmy.cli.ui import styles

        self.args = argparse.Namespace(provider="stub", model=None)
        # Force a non-styling palette so assertions match plain text.
        self._c = styles(None)
        self.errs: list[str] = []

    def _eprint(self, *parts: Any) -> None:
        self.errs.append(" ".join(str(p) for p in parts))


def test_slash_skill_runs_in_session(capsys: pytest.CaptureFixture[str]) -> None:
    """`/skill summarize <task>` runs the skill and prints its answer."""
    repl = _FakeRepl()
    slash_skill(repl, ["summarize", "brief", "me", "on", "X"])
    out = capsys.readouterr().out
    assert "brief me on X" in out


def test_slash_skill_usage_when_no_task(capsys: pytest.CaptureFixture[str]) -> None:
    """`/skill summarize` with no task prints usage and does not run."""
    repl = _FakeRepl()
    slash_skill(repl, ["summarize"])
    assert any("usage: /skill" in e for e in repl.errs)
    assert capsys.readouterr().out == ""


def test_slash_skill_unknown_name(capsys: pytest.CaptureFixture[str]) -> None:
    """`/skill nope <task>` reports an unknown skill and runs nothing."""
    repl = _FakeRepl()
    slash_skill(repl, ["nope", "do", "a", "thing"])
    assert any("unknown skill 'nope'" in e for e in repl.errs)
    assert capsys.readouterr().out == ""
