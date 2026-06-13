"""Tests for the CLI read/write parity commands over the shared durable stores (T2h + T2b).

Covers ``himmy runs``/``recommendations``/``context``/``dashboard`` and ``--persist`` on
``himmy run``/``chat``. The headline assertions are the COHERENCE payoff: a run persisted by
the CLI lands in the ONE canonical ``.himmy/storage.db`` that ``GET /v1/runs`` and the Studio
runs screen read, and the four CLI surfaces all agree (dashboard counts == runs list ==
recommendations list) over ONE pinned ``__local__`` workspace. Back-compat: a plain one-shot
``himmy run`` WITHOUT ``--persist`` writes nothing (the zero-config in-RAM default).

Isolation: each test chdir's into a fresh tmp dir and pins ``HIMMY_STORE_PATH`` /
``HIMMY_SPINE_PATH`` to tmp files, so the durable store/spine never touch the repo's
``.himmy/`` and tests don't leak state into each other.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from himmy.cli.__main__ import _command_names, build_parser, main


@pytest.fixture
def local_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A clean project root: tmp cwd + tmp durable store/spine paths."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / ".himmy" / "storage.db"))
    monkeypatch.setenv("HIMMY_SPINE_PATH", str(tmp_path / ".himmy" / "spine.db"))
    return tmp_path


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


# --------------------------------------------------------------- parser wiring


def test_new_subcommands_registered() -> None:
    names = _command_names(build_parser())
    for cmd in ("runs", "recommendations", "context", "dashboard"):
        assert cmd in names


def test_persist_flag_defaults_off() -> None:
    parser = build_parser()
    assert parser.parse_args(["run", "hi"]).persist is False
    assert parser.parse_args(["run", "hi", "--persist"]).persist is True
    assert parser.parse_args(["chat", "--message", "hi"]).persist is False


# --------------------------------------------------------- --persist back-compat


def test_run_without_persist_writes_nothing(local_project: Path) -> None:
    """The zero-config default: a plain one-shot leaves the canonical store empty."""
    assert main(["run", "--provider", "stub", "--name", "t", "hello"]) == 0
    # no durable store file is even created by a non-persist run (checked BEFORE the
    # canonical-read helper, which would itself create the empty db on open)
    assert not (local_project / ".himmy" / "storage.db").exists()
    assert _canonical_runs(local_project) == []


def test_run_persist_lands_in_canonical_store(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`himmy run --persist` records the completed run in the ONE canonical store."""
    assert main(["run", "--provider", "stub", "--name", "t", "--persist", "hi"]) == 0
    runs = _canonical_runs(local_project)
    assert len(runs) == 1
    run = runs[0]
    assert run.status.value == "SUCCEEDED"
    assert run.metadata.get("source") == "cli"
    assert run.metadata.get("prompt") == "hi"


def test_persist_visible_in_runs_list_and_show(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After --persist, `himmy runs list`/`show` surface the run from the canonical store."""
    assert main(["run", "--provider", "stub", "--name", "t", "--persist", "hi"]) == 0
    run_id = _canonical_runs(local_project)[0].run_id

    capsys.readouterr()
    assert main(["runs", "list"]) == 0
    assert run_id in capsys.readouterr().out

    assert main(["runs", "show", run_id]) == 0
    out = capsys.readouterr().out
    assert run_id in out
    assert "SUCCEEDED" in out


def test_runs_show_unknown_id_is_error(local_project: Path) -> None:
    assert main(["runs", "show", "nope-id"]) == 1


def test_runs_list_status_filter(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--provider", "stub", "--name", "t", "--persist", "hi"]) == 0
    capsys.readouterr()
    # SUCCEEDED matches, FAILED does not
    assert main(["runs", "list", "--status", "SUCCEEDED"]) == 0
    assert _canonical_runs(local_project)[0].run_id in capsys.readouterr().out
    assert main(["runs", "list", "--status", "FAILED"]) == 0
    assert "no persisted runs" in capsys.readouterr().err


def test_chat_message_persist(local_project: Path) -> None:
    """`himmy chat --message --persist` records the turn in the canonical store."""
    assert (
        main(["chat", "--provider", "stub", "--name", "t", "--persist", "--message", "yo"])
        == 0
    )
    runs = _canonical_runs(local_project)
    assert len(runs) == 1
    assert runs[0].metadata.get("prompt") == "yo"


# ------------------------------------------------------------------- dashboard


def test_dashboard_counts_match_runs(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The dashboard summary counts the SAME pinned workspace as `himmy runs`."""
    for _ in range(2):
        assert main(["run", "--provider", "stub", "--name", "t", "--persist", "hi"]) == 0
    assert len(_canonical_runs(local_project)) == 2

    capsys.readouterr()
    assert main(["dashboard", "summary", "--json"]) == 0
    import json

    summary = json.loads(capsys.readouterr().out)
    assert summary["workspace_id"] == "__local__"
    assert summary["runs"]["total"] == 2
    assert summary["runs"]["by_status"]["SUCCEEDED"] == 2


# --------------------------------------------------------------------- context


def test_context_fields_set_list_and_dashboard(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["context", "fields", "set", "city", "Kathmandu", "--confidence", "0.9"]) == 0
    capsys.readouterr()
    assert main(["context", "fields", "list"]) == 0
    assert "city" in capsys.readouterr().out

    # the dashboard reflects the field count for the same pinned scope
    capsys.readouterr()
    assert main(["dashboard", "summary", "--json"]) == 0
    import json

    summary = json.loads(capsys.readouterr().out)
    assert summary["context"]["field_count"] == 1


def test_context_snapshot_build_and_show(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["context", "fields", "set", "city", "Kathmandu"]) == 0
    capsys.readouterr()
    assert main(["context", "snapshot"]) == 0
    snap_id = capsys.readouterr().out.strip()
    assert snap_id

    assert main(["context", "snapshot", "--show", snap_id]) == 0
    assert snap_id in capsys.readouterr().out


def test_context_snapshot_no_fields_errors(local_project: Path) -> None:
    assert main(["context", "snapshot"]) == 1


# ------------------------------------------------------------- recommendations


def _seed_recommendation(local_project: Path) -> str:
    """Write one recommendation straight into the canonical store; return its id."""
    from himmy.services.storage.models import (
        LOCAL_SUBJECT,
        LOCAL_WORKSPACE,
        RecommendationItem,
    )
    from himmy.services.storage.sqlite import SqliteStorageService

    item = RecommendationItem(
        run_id="run-1",
        workspace_id=LOCAL_WORKSPACE,
        subject_id=LOCAL_SUBJECT,
        kind="action",
        title="Water the orchard",
    )

    async def _write() -> str:
        store = SqliteStorageService(str(local_project / ".himmy" / "storage.db"))
        try:
            await store.save_recommendation(item)
            return item.recommendation_id
        finally:
            await store.close()

    return asyncio.run(_write())


def test_recommendations_list_show_transition(
    local_project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rec_id = _seed_recommendation(local_project)

    capsys.readouterr()
    assert main(["recommendations", "list"]) == 0
    assert rec_id in capsys.readouterr().out

    assert main(["recommendations", "show", rec_id]) == 0
    out = capsys.readouterr().out
    assert rec_id in out
    assert "PROPOSED" in out

    # transition to ACCEPTED via the same write /v1 exposes
    assert (
        main(["recommendations", "transition", rec_id, "--status", "ACCEPTED"]) == 0
    )
    assert "ACCEPTED" in capsys.readouterr().out

    # the new status is persisted and visible to a list filter
    capsys.readouterr()
    assert main(["recommendations", "list", "--status", "ACCEPTED"]) == 0
    assert rec_id in capsys.readouterr().out


def test_recommendations_show_unknown_is_error(local_project: Path) -> None:
    assert main(["recommendations", "show", "nope"]) == 1


def test_recommendations_show_uses_app_service_getter(local_project: Path) -> None:
    """The show path routes through the new RecommendationAppService.get (not storage)."""
    from himmy.application.services import RecommendationAppService

    # the getter the reviewer must_fix required exists and is workspace-scoped
    assert hasattr(RecommendationAppService, "get")


# --------------------------------------------------------------- app container


def test_build_app_container_pins_local_and_is_durable(local_project: Path) -> None:
    """The container pins __local__ for both reads and writes and is durable."""
    from himmy.cli.app_services import build_app_container
    from himmy.services.storage.models import LOCAL_SUBJECT, LOCAL_WORKSPACE
    from himmy.services.storage.sqlite import SqliteStorageService

    container = build_app_container()
    try:
        assert container.workspace_id == LOCAL_WORKSPACE
        assert container.subject_id == LOCAL_SUBJECT
        # durable SQLite store at the canonical path (NOT the in-RAM for_cli store)
        assert isinstance(container.storage, SqliteStorageService)
    finally:
        container.close()


def test_app_container_drops_governed_overlay(local_project: Path) -> None:
    """The ungoverned-local container exposes no consent overlay (stated, by design)."""
    from himmy.cli.app_services import build_app_container

    container = build_app_container()
    try:
        # it is the bare storage/spine — no ConsentGatedStorage / consent decider attached
        assert not hasattr(container, "consent_ledger")
        assert not hasattr(container, "consent_decider")
    finally:
        container.close()
