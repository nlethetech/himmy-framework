"""deploy-dx docs-and-hints (P1 #7 + #8 + #10) — the newcomer-facing signposts.

The machinery (worker, scheduler flock, inbound webhook, extras) already exists; these
tests pin the THIN front-door signposts over it:

* ``himmy routines add`` warns when nothing will fire the routine (no scheduler here);
* ``himmy doctor`` flags enabled-routines-without-a-scheduler in RED;
* the stub hint names the EXACT detected re-run command (never "pull a model you have");
* ``himmy doctor`` prints the exact ``pip install 'himmy[...]'`` for a missing extra;
* the docs carry the agent-over-HTTP (signed inbound webhook) recipe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from himmy.cli import commands
from himmy.cli.__main__ import main
from himmy.cli.wizard import ProviderChoice


def _stub_detect(*choices: ProviderChoice):
    """A ``detect_provider_choices`` stand-in returning ``choices`` (stub appended if absent)."""
    result = list(choices)
    if not result or result[-1].key != "stub":
        result.append(ProviderChoice(key="stub", label="stub"))
    return lambda: result


@pytest.fixture
def local_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean project root: tmp cwd + tmp routines/store paths, no scheduler."""
    from himmy.api import routines as svc

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_ROUTINES_PATH", str(tmp_path / ".himmy" / "routines.db"))
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / ".himmy" / "storage.db"))
    monkeypatch.setenv("HIMMY_SPINE_PATH", str(tmp_path / ".himmy" / "spine.db"))
    monkeypatch.setenv("HIMMY_ROUTINES_SCHEDULER", "off")
    (tmp_path / ".himmy").mkdir(exist_ok=True)
    (tmp_path / "agent.yaml").write_text("name: helper\ndescription: A helper.\n")
    svc.reset_routines_store()
    svc.reset_scheduler()
    yield tmp_path
    svc.reset_routines_store()
    svc.reset_scheduler()


# --------------------------------------------------------------- #8 routines worker hint


def test_routines_add_warns_without_a_worker(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`routines add` on an enabled routine warns when no scheduler will fire it."""
    rc = main(
        [
            "routines", "add",
            "--name", "brief",
            "-f", "agent.yaml",
            "-p", "summarize",
            "--daily", "09:00",
        ]
    )
    assert rc == 0
    err = capsys.readouterr().err
    assert "no scheduler is running" in err
    assert "himmy worker" in err


def test_routines_add_disabled_does_not_warn(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A routine created ``--disabled`` won't fire yet, so the worker hint is silent."""
    rc = main(
        [
            "routines", "add",
            "--name", "brief",
            "-f", "agent.yaml",
            "-p", "summarize",
            "--daily", "09:00",
            "--disabled",
        ]
    )
    assert rc == 0
    assert "no scheduler is running" not in capsys.readouterr().err


def test_routines_add_quiet_when_scheduler_running(
    local_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When a scheduler IS detected on this host, no worker hint is printed."""
    # The hint helper imports the probe by name from the leader module; patch it there.
    import himmy.api.scheduler_leader as leader

    monkeypatch.setattr(leader, "scheduler_running_on_host", lambda: True)
    rc = main(
        [
            "routines", "add",
            "--name", "brief",
            "-f", "agent.yaml",
            "-p", "summarize",
            "--daily", "09:00",
        ]
    )
    assert rc == 0
    assert "no scheduler is running" not in capsys.readouterr().err


# ------------------------------------------------------------ #8 doctor runtime section


def test_doctor_runtime_flags_routines_without_scheduler(
    local_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`himmy doctor` RED-flags an enabled routine with no scheduler to fire it."""
    import himmy.api.scheduler_leader as leader

    monkeypatch.setattr(leader, "scheduler_running_on_host", lambda: False)
    main(
        [
            "routines", "add",
            "--name", "brief",
            "-f", "agent.yaml",
            "-p", "summarize",
            "--daily", "09:00",
        ]
    )
    capsys.readouterr()  # drop the add output
    code = main(["doctor"])
    assert code == 0
    out = capsys.readouterr().out
    assert "runtime (unattended runs)" in out
    assert "RED:" in out
    assert "will NEVER fire" in out


def test_doctor_runtime_healthy_when_scheduler_running(
    local_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No RED line when a scheduler is running for the enabled routine."""
    import himmy.api.scheduler_leader as leader

    monkeypatch.setattr(leader, "scheduler_running_on_host", lambda: True)
    main(
        [
            "routines", "add",
            "--name", "brief",
            "-f", "agent.yaml",
            "-p", "summarize",
            "--daily", "09:00",
        ]
    )
    capsys.readouterr()
    main(["doctor"])
    out = capsys.readouterr().out
    assert "runtime (unattended runs)" in out
    assert "RED:" not in out


def test_doctor_runtime_no_routines_no_red(
    local_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With zero routines defined there is nothing to fire, so no RED line."""
    import himmy.api.scheduler_leader as leader

    monkeypatch.setattr(leader, "scheduler_running_on_host", lambda: False)
    main(["doctor"])
    out = capsys.readouterr().out
    assert "runtime (unattended runs)" in out
    assert "RED:" not in out


# ------------------------------------------------------------------ #10 stub re-run hint


def _run_args(spec_file: Path) -> argparse.Namespace:
    return argparse.Namespace(
        file=str(spec_file), name=None, instruction=None, provider=None, model=None
    )


def test_stub_hint_prints_detected_rerun_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stub hint names the EXACT `--provider/--model` for the detected backend."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(
        wizard,
        "detect_provider_choices",
        _stub_detect(
            ProviderChoice(key="ollama", label="ollama — local", model="qwen2.5:3b")
        ),
    )
    # The spec names no provider (so the hint isn't suppressed as an explicit choice) and
    # resolves to the stub — pin that so the test doesn't depend on the host's real backends.
    from himmy.cli import provider as provider_mod

    monkeypatch.setattr(provider_mod, "resolves_to_stub", lambda *a, **k: True)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.delenv("HIMMY_NO_HINTS", raising=False)
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec(name="a")
    args = _run_args(tmp_path / "agent.yaml")
    commands._maybe_hint_stub(spec, args)
    err = capsys.readouterr().err
    assert "--provider ollama --model qwen2.5:3b" in err
    # Never tell a user to pull a model they already have.
    assert "ollama pull" not in err


def test_stub_hint_install_lines_when_nothing_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no real backend detected the hint falls back to install lines."""
    import himmy.cli.wizard as wizard

    monkeypatch.setattr(wizard, "detect_provider_choices", _stub_detect())
    from himmy.cli import provider as provider_mod

    monkeypatch.setattr(provider_mod, "resolves_to_stub", lambda *a, **k: True)
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.delenv("HIMMY_NO_HINTS", raising=False)
    from himmy.config.agent_spec import AgentSpec

    spec = AgentSpec(name="a")
    commands._maybe_hint_stub(spec, _run_args(tmp_path / "agent.yaml"))
    err = capsys.readouterr().err
    assert "no real backend detected" in err
    assert "ollama pull llama3.2" in err


# ------------------------------------------------------------ #10 doctor pip-install hint


def test_doctor_prints_pip_install_for_missing_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A missing optional extra shows the exact `pip install 'himmy[<extra>]'`."""
    from himmy.runtime import diagnostics

    # Force one extra to read as missing so the install hint must appear.
    real_can_import = diagnostics._can_import
    monkeypatch.setattr(
        diagnostics,
        "_can_import",
        lambda mod: False if mod == "asyncpg" else real_can_import(mod),
    )
    monkeypatch.chdir(tmp_path)
    main(["doctor"])
    out = capsys.readouterr().out
    assert "pip install 'himmy[postgres]'" in out


# ------------------------------------------------------------------ #7 docs recipe present


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_deployment_docs_contain_inbound_recipe() -> None:
    """The deployment runbook documents the signed inbound-webhook agent-over-HTTP path."""
    text = (_repo_root() / "docs" / "enterprise" / "deployment.md").read_text()
    assert "HIMMY_INBOUND_AGENT_PATH" in text
    assert "/v1/connectors/webhook" in text
    assert "X-Himmy-Signature" in text
    assert "HIMMY_WEBHOOK_SIGNING_SECRET" in text
    # a working signed curl (compute a signature, never echo the secret)
    assert "openssl dgst -sha256 -hmac" in text
    assert "curl" in text


def test_recipes_contain_inbound_recipe() -> None:
    """RECIPES.md carries the same agent-over-HTTP recipe for discoverability."""
    text = (_repo_root() / "RECIPES.md").read_text()
    assert "/v1/connectors/webhook" in text
    assert "HIMMY_INBOUND_AGENT_PATH" in text
    assert "X-Himmy-Signature" in text
