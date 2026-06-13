"""``himmy seclog`` — the SECURITY audit-log CLI (distinct from the privacy ``audit``).

Asserts, over an in-memory :class:`EntityRegistry` + :class:`SecurityAuditLog`:

* recorded events render as a who/what/when table (newest first), with allow/deny outcomes;
* an empty spine is success (exit 0) + a "no security events" note, not a failure;
* ``--json`` emits the full events and never the ANSI table;
* ``--type`` / ``--limit`` filter the rows;
* :func:`cmd_seclog` over a fresh durable spine exits 0 with the empty note;
* the durable ``.himmy/spine.db`` persists events across calls and ``record_security_event``
  fails open on a logging error.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import himmy.cli.security_audit_cmd as seclog_cmd
from himmy.cli.security_audit_cmd import cmd_seclog, render_seclog
from himmy.entities.registry import EntityRegistry
from himmy.services.audit.log import SecurityAuditLog
from himmy.services.audit.models import SecurityEvent


def _audit_with_events() -> SecurityAuditLog:
    """A log over a fresh in-memory spine holding a deny + an allow event."""
    audit = SecurityAuditLog(EntityRegistry())
    audit.record(
        SecurityEvent(
            event_type="authz_denied",
            outcome="deny",
            actor={"subject": "alice"},
            action="tool_call",
            resource="shell_exec",
            created_at="2026-06-11T10:00:00Z",
            detail="policy blocked",
        )
    )
    audit.record(
        SecurityEvent(
            event_type="access",
            outcome="allow",
            actor={"subject": "bob"},
            action="tool_call",
            resource="kb_search",
            created_at="2026-06-11T11:00:00Z",
        )
    )
    return audit


# ----------------------------------------------------------------- happy path
def test_render_table_lists_recorded_events(capsys: Any) -> None:
    """Both events render with their actor / action / resource / outcome."""
    rc = render_seclog(_audit_with_events())
    assert rc == 0
    out = capsys.readouterr().out
    assert "alice" in out and "bob" in out
    assert "shell_exec" in out and "kb_search" in out
    assert "deny" in out and "allow" in out
    # The header columns are present.
    assert "actor" in out and "outcome" in out


def test_newest_event_first(capsys: Any) -> None:
    """``recent`` returns newest-first, so bob (11:00) precedes alice (10:00)."""
    render_seclog(_audit_with_events())
    out = capsys.readouterr().out
    assert out.index("bob") < out.index("alice")


# ------------------------------------------------------------------- empty spine
def test_empty_spine_is_success_with_note(capsys: Any) -> None:
    """An empty trail exits 0 (clean posture) and prints the empty-spine note."""
    rc = render_seclog(SecurityAuditLog(EntityRegistry()))
    assert rc == 0
    out = capsys.readouterr().out
    assert "no security events" in out


# ------------------------------------------------------------------------- json
def test_json_emits_full_events_no_table(capsys: Any) -> None:
    """``--json`` is machine-readable: a list of full events, no header row."""
    rc = render_seclog(_audit_with_events(), as_json=True)
    assert rc == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert isinstance(payload, list) and len(payload) == 2
    subjects = {row["actor"]["subject"] for row in payload}
    assert subjects == {"alice", "bob"}
    # No styled table leaked into the JSON path.
    assert "outcome\x1b" not in out
    assert "\n" not in out.rstrip("\n")  # single JSON line


def test_json_empty_spine_is_empty_list(capsys: Any) -> None:
    """An empty spine in ``--json`` mode emits ``[]`` and exits 0."""
    rc = render_seclog(SecurityAuditLog(EntityRegistry()), as_json=True)
    assert rc == 0
    assert json.loads(capsys.readouterr().out) == []


# ----------------------------------------------------------------------- filters
def test_type_filter(capsys: Any) -> None:
    """``event_type`` filters to just the matching events."""
    render_seclog(_audit_with_events(), event_type="authz_denied")
    out = capsys.readouterr().out
    assert "alice" in out
    assert "bob" not in out


def test_limit_bounds_rows(capsys: Any) -> None:
    """``limit=1`` shows only the newest event."""
    render_seclog(_audit_with_events(), limit=1)
    out = capsys.readouterr().out
    assert "bob" in out
    assert "alice" not in out


def test_limit_zero_shows_empty_note(capsys: Any) -> None:
    """``limit=0`` is treated as no rows ⇒ the empty-spine note (still exit 0)."""
    rc = render_seclog(_audit_with_events(), limit=0)
    assert rc == 0
    assert "no security events" in capsys.readouterr().out


# ------------------------------------------------------------- cmd_seclog wiring
def test_cmd_seclog_default_spine_is_empty(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """``cmd_seclog`` over a fresh durable spine exits 0 with the empty note."""
    monkeypatch.chdir(tmp_path)
    args = argparse.Namespace(limit=20, type=None, workspace=None, json=False)
    rc = cmd_seclog(args)
    assert rc == 0
    assert "no security events" in capsys.readouterr().out


# -------------------------------------------------- durable spine (Finding 3)
def test_cli_security_log_is_durable_across_calls(
    monkeypatch: Any, tmp_path: Any
) -> None:
    """A SecurityEvent recorded via one cli_security_log() is visible to a fresh one
    (the spine is the on-disk .himmy/spine.db, not a per-call in-memory registry)."""
    # Exercise the cwd-relative default path explicitly (the global test fixture pins
    # HIMMY_SPINE_PATH for isolation; this test verifies the unset/default resolution).
    monkeypatch.delenv("HIMMY_SPINE_PATH", raising=False)
    monkeypatch.chdir(tmp_path)
    seclog_cmd.record_security_event(
        event_type="authz_denied",
        outcome="deny",
        actor={"subject": "viewer"},
        action="tool_call",
        resource="wire_money",
        detail="rbac re-gated",
    )
    assert (tmp_path / ".himmy" / "spine.db").is_file()
    # A *fresh* log over the same path sees the persisted event.
    events = seclog_cmd.cli_security_log().recent(limit=10)
    assert [e.resource for e in events] == ["wire_money"]
    assert events[0].event_type == "authz_denied"
    assert events[0].outcome == "deny"


def test_cmd_seclog_reads_durable_spine(
    monkeypatch: Any, tmp_path: Any, capsys: Any
) -> None:
    """`himmy seclog` reads the SAME durable spine the deny sites write to."""
    monkeypatch.chdir(tmp_path)
    seclog_cmd.record_security_event(
        event_type="authz_denied",
        outcome="deny",
        actor={"subject": "viewer"},
        action="tool_call",
        resource="run_python",
    )
    args = argparse.Namespace(limit=20, type=None, workspace=None, json=True)
    rc = cmd_seclog(args)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["resource"] == "run_python"
    assert payload[0]["actor"]["subject"] == "viewer"


def test_record_security_event_fails_open(monkeypatch: Any, tmp_path: Any) -> None:
    """A recording error never propagates — logging is not the gate (fail open)."""
    monkeypatch.chdir(tmp_path)

    def _boom() -> None:
        raise RuntimeError("db is wedged")

    monkeypatch.setattr(seclog_cmd, "cli_security_log", _boom)
    # Must NOT raise despite the broken log.
    seclog_cmd.record_security_event(
        event_type="authz_denied", outcome="deny", resource="x"
    )
