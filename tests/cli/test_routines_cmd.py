"""T3c — ``himmy routines`` CLI over the shared routines store + canonical run store.

The CLI surface is single-user-local: routines live in the ``__local__`` workspace and
bind their agent by a filesystem ``agent.yaml`` path. The headline coherence assertion:
a ``himmy routines run-now`` run lands in the ONE canonical ``.himmy/storage.db`` that
``himmy runs`` and ``GET /v1/runs`` read.

Isolation: each test chdir's into a fresh tmp dir and pins the routines/store/spine paths
to tmp files, so nothing leaks into the repo's ``.himmy/``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from himmy.api import routines as svc
from himmy.cli.__main__ import _command_names, build_parser, main


@pytest.fixture
def local_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean project root: tmp cwd + tmp routines/store/spine paths, no scheduler."""
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


def _canonical_runs(local_project: Path) -> list:
    """Read the local-workspace runs straight from the canonical store (the /v1 view)."""
    from himmy.services.storage.models import LOCAL_WORKSPACE
    from himmy.services.storage.sqlite import SqliteStorageService

    async def _read() -> list:
        store = SqliteStorageService(str(local_project / ".himmy" / "storage.db"))
        try:
            return await store.list_runs(workspace_id=LOCAL_WORKSPACE)
        finally:
            await store.close()

    return asyncio.run(_read())


def _add_routine(name: str = "brief") -> str:
    """Add a routine via the CLI and return its id (printed to stdout)."""
    rc = main(
        [
            "routines",
            "add",
            "--name",
            name,
            "-f",
            "agent.yaml",
            "-p",
            "write a brief",
            "--every",
            "6",
            "--provider",
            "stub",
        ]
    )
    assert rc == 0
    # The id is the last stored routine (list is newest-first).
    return svc.get_routines_store().list(workspace_id=svc.LOCAL_WORKSPACE)[0].id


# --------------------------------------------------------------- parser wiring


def test_routines_subcommand_registered() -> None:
    assert "routines" in _command_names(build_parser())


def test_add_requires_exactly_one_schedule() -> None:
    parser = build_parser()
    # argparse mutually-exclusive group is required → SystemExit when neither given.
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["routines", "add", "--name", "n", "-f", "a.yaml", "-p", "x"]
        )


# ------------------------------------------------------------------ CRUD cycle


def test_add_list_show_toggle_remove(local_project: Path, capsys: pytest.CaptureFixture) -> None:
    rid = _add_routine()
    capsys.readouterr()

    assert main(["routines", "list"]) == 0
    out = capsys.readouterr().out
    assert rid in out

    assert main(["routines", "show", rid]) == 0
    shown = capsys.readouterr().out
    assert rid in shown
    assert "every" in shown

    assert main(["routines", "disable", rid]) == 0
    assert svc.get_routines_store().get(rid).enabled is False
    assert main(["routines", "enable", rid]) == 0
    assert svc.get_routines_store().get(rid).enabled is True

    assert main(["routines", "remove", rid]) == 0
    assert svc.get_routines_store().get(rid) is None


def test_add_rejects_missing_agent(local_project: Path) -> None:
    rc = main(
        [
            "routines",
            "add",
            "--name",
            "n",
            "-f",
            "nope.yaml",
            "-p",
            "x",
            "--every",
            "1",
        ]
    )
    assert rc == 1


# ----------------------------------------------- run-now → canonical run store


def test_run_now_lands_in_canonical_store(
    local_project: Path, capsys: pytest.CaptureFixture
) -> None:
    """A CLI run-now run is visible in the canonical store (himmy runs / /v1 / Studio)."""
    rid = _add_routine()
    capsys.readouterr()

    assert main(["routines", "run-now", rid]) == 0
    # The routine recorded a successful run…
    routine = svc.get_routines_store().get(rid, workspace_id=svc.LOCAL_WORKSPACE)
    assert routine.last_status == "ok"
    assert routine.last_run_at is not None

    # …and that run landed in the ONE canonical store under __local__.
    runs = _canonical_runs(local_project)
    assert len(runs) == 1
    assert runs[0].status.value == "SUCCEEDED"


def test_run_now_unknown_routine(local_project: Path) -> None:
    assert main(["routines", "run-now", "does-not-exist"]) == 1
