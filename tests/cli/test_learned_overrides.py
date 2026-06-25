"""Tests for the Sprint-2 ``himmy learned`` operator-override verbs (L2-cli).

Covers: the bare ``himmy learned`` report staying byte-identical to Sprint 1 (the
feature-off / no-override invariant), the new pin / mute(ignore) / reset / clear /
trust-feedback verbs writing through :class:`OverrideStore`, the override-aware read
view (STATUS column + ``--json`` per-tool ``override`` + top-level ``trust_feedback``),
and the verb-dispatch / bare-command coexistence under argparse.
"""

from __future__ import annotations

import argparse
import json

import pytest

from himmy.cli import learned_cmd
from himmy.core.events import EventType, RunEvent
from himmy.entities.sqlite_registry import SqliteEntityRegistry
from himmy.services.learning.overrides import OverrideStore
from himmy.services.storage.service import StorageService
from tests.conftest import run_async


def _emit(
    store: StorageService, tool: str, ok: bool, *, workspace: str | None = None
) -> None:
    run_async(
        store.append_event(
            RunEvent(
                event_type=EventType.TOOL_COMPLETED if ok else EventType.TOOL_FAILED,
                workspace_id=workspace,
                payload={"tool_name": tool},
            )
        )
    )


@pytest.fixture()
def wired(monkeypatch):
    """Wire ``himmy learned`` to an in-memory event store + an injected override spine.

    Returns ``(store, registry)`` so a test can seed tool events and inspect the override
    version chain. The same in-memory ``SqliteEntityRegistry`` backs every ``for_cli()``
    call, so a write from one verb is visible to a subsequent read (parity with the real
    shared ``.himmy/spine.db``).
    """
    store = StorageService()
    registry = SqliteEntityRegistry()

    monkeypatch.setattr(
        learned_cmd.StoreFactory, "for_cli_durable", classmethod(lambda cls: store)
    )
    monkeypatch.setattr(
        OverrideStore, "for_cli", classmethod(lambda cls: OverrideStore(registry))
    )
    return store, registry


def _ns(**kw) -> argparse.Namespace:
    base = {"workspace": None, "outcome_weight": 0.0, "json": False, "all": False}
    base.update(kw)
    return argparse.Namespace(**base)


def _seed_flaky(store: StorageService) -> None:
    """A genuinely-flaky tool (10% success) the auto-learner would demote."""
    _emit(store, "flaky", True)
    for _ in range(9):
        _emit(store, "flaky", False)


# --------------------------------------------------------------------------- (1)
def test_bare_learned_report_unchanged_vs_sprint1(wired, capsys):
    """Bare ``himmy learned`` with no overrides renders the Sprint-1 report.

    The new ``trust_feedback`` summary line is additive; the per-tool STATUS and the
    flaky verdict are unchanged when no override is set.
    """
    store, _ = wired
    _seed_flaky(store)

    assert learned_cmd.cmd_learned(_ns()) == 0
    out = capsys.readouterr().out

    # The flaky tool is still flagged exactly as Sprint 1 — no override changed it.
    assert "flaky — demoted" in out
    # No override labels appear with nothing set.
    assert "pinned" not in out
    assert "ignored" not in out
    # trust_feedback defaults to OFF.
    assert "trust_feedback: off" in out


def test_bare_learned_json_has_override_and_trust_keys(wired, capsys):
    """``--json`` carries the new ``override`` (None) + ``trust_feedback`` (False) keys."""
    store, _ = wired
    _seed_flaky(store)

    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)

    assert data["trust_feedback"] is False
    row = {t["tool_name"]: t for t in data["tools"]}["flaky"]
    assert row["override"] is None
    assert row["flaky"] is True


# --------------------------------------------------------------------------- (2)
def test_pin_then_json_shows_pinned_and_not_flaky(wired, capsys):
    """``himmy learned pin`` persists, and the read view flips the tool to pinned."""
    store, registry = wired
    _seed_flaky(store)

    rc = learned_cmd.cmd_learned_pin(_ns(tool_name="flaky"))
    assert rc == 0
    confirm = capsys.readouterr().out
    assert "pinned flaky" in confirm
    # The verb re-renders the override-aware report inline.
    assert "pinned (trusted)" in confirm

    # The override is durable on the injected spine.
    assert registry.get_latest(OverrideStore._override_stable_id(None, "flaky")) is not None

    # A fresh read sees the pin: override==pinned, no longer flaky.
    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    row = {t["tool_name"]: t for t in data["tools"]}["flaky"]
    assert row["override"] == "pinned"
    assert row["flaky"] is False
    assert data["flaky_count"] == 0


def test_mute_alias_ignore_shows_ignored(wired, capsys):
    """``mute`` records an ignore override surfaced as ``ignored`` in the read view."""
    store, _ = wired
    _seed_flaky(store)

    assert learned_cmd.cmd_learned_mute(_ns(tool_name="flaky")) == 0
    capsys.readouterr()

    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    row = {t["tool_name"]: t for t in data["tools"]}["flaky"]
    assert row["override"] == "ignored"
    assert row["flaky"] is False


def test_clear_removes_override(wired, capsys):
    """``clear`` returns a tool to pure auto-learning (override back to None)."""
    store, _ = wired
    _seed_flaky(store)
    learned_cmd.cmd_learned_pin(_ns(tool_name="flaky"))
    capsys.readouterr()

    assert learned_cmd.cmd_learned_clear(_ns(tool_name="flaky")) == 0
    capsys.readouterr()

    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    row = {t["tool_name"]: t for t in data["tools"]}["flaky"]
    assert row["override"] is None
    # Auto-learning is back in charge — the tool is flaky again.
    assert row["flaky"] is True


def test_reset_shows_rebuilding(wired, capsys):
    """``reset`` stamps a cutoff so the tool forgets its history and rebuilds neutral."""
    store, _ = wired
    _seed_flaky(store)

    assert learned_cmd.cmd_learned_reset(_ns(tool_name="flaky")) == 0
    capsys.readouterr()

    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    row = {t["tool_name"]: t for t in data["tools"]}["flaky"]
    assert row["override"] == "reset"
    # Pre-cutoff events dropped -> below min_samples -> neutral, not punished.
    assert row["has_min_samples"] is False
    assert row["flaky"] is False


# --------------------------------------------------------------------------- (3)
def test_trust_feedback_on_persists_and_shows_in_json(wired, capsys):
    """``trust-feedback on`` persists and lifts the displayed outcome_weight + flag."""
    store, _ = wired
    _emit(store, "t", True)

    assert learned_cmd.cmd_learned_trust_feedback(_ns(state="on")) == 0
    confirm = capsys.readouterr().out
    assert "trust_feedback on" in confirm

    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["trust_feedback"] is True
    # The gate lifts the effective weight from 0 to the sensible default.
    assert data["outcome_weight"] == pytest.approx(0.3)


def test_trust_feedback_off_is_default_and_toggles_back(wired, capsys):
    store, _ = wired
    _emit(store, "t", True)

    learned_cmd.cmd_learned_trust_feedback(_ns(state="on"))
    capsys.readouterr()
    assert learned_cmd.cmd_learned_trust_feedback(_ns(state="off")) == 0
    capsys.readouterr()

    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["trust_feedback"] is False
    assert data["outcome_weight"] == 0.0


# --------------------------------------------------------------------------- (4)
def test_verb_dispatch_and_bare_command_coexist(wired, capsys):
    """The argparse wiring routes verbs to handlers AND the bare command to the report."""
    store, registry = wired
    _seed_flaky(store)

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    learned_cmd.add_learned_parser(sub)

    # Bare command -> cmd_learned (renders the report).
    bare = parser.parse_args(["learned"])
    assert bare.func is learned_cmd.cmd_learned
    assert bare.func(bare) == 0
    bare_out = capsys.readouterr().out
    assert "flaky — demoted" in bare_out

    # Verb -> its handler.
    pinned = parser.parse_args(["learned", "pin", "flaky"])
    assert pinned.func is learned_cmd.cmd_learned_pin
    assert pinned.tool_name == "flaky"
    assert pinned.func(pinned) == 0
    capsys.readouterr()

    # ``ignore`` alias routes to the mute handler.
    ignored = parser.parse_args(["learned", "ignore", "other"])
    assert ignored.func is learned_cmd.cmd_learned_mute

    # trust-feedback verb parses its on/off state.
    tf = parser.parse_args(["learned", "trust-feedback", "on"])
    assert tf.func is learned_cmd.cmd_learned_trust_feedback
    assert tf.state == "on"

    # The pin we ran above is durable on the shared injected spine.
    assert registry.get_latest(OverrideStore._override_stable_id(None, "flaky")) is not None


def test_verb_workspace_scopes_override(wired, capsys):
    """A workspace-scoped pin is invisible to the default subject (tenant isolation)."""
    store, _ = wired
    # The flaky tool has events in BOTH the default subject and workspace A.
    _seed_flaky(store)
    _emit(store, "flaky", True, workspace="A")
    for _ in range(9):
        _emit(store, "flaky", False, workspace="A")

    assert learned_cmd.cmd_learned_pin(_ns(tool_name="flaky", workspace="A")) == 0
    capsys.readouterr()

    # Default-subject read does not see workspace A's override.
    assert learned_cmd.cmd_learned(_ns(json=True)) == 0
    data = json.loads(capsys.readouterr().out)
    row = {t["tool_name"]: t for t in data["tools"]}["flaky"]
    assert row["override"] is None

    # Reading workspace A surfaces it.
    assert learned_cmd.cmd_learned(_ns(json=True, workspace="A")) == 0
    data_a = json.loads(capsys.readouterr().out)
    row_a = {t["tool_name"]: t for t in data_a["tools"]}["flaky"]
    assert row_a["override"] == "pinned"
