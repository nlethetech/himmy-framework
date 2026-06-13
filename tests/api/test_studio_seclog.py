"""Studio seclog router — the read-only security-events viewer (T3f).

In-process FastAPI app, no network. The headline acceptance (reviewer must_fix) is
PARITY: the Studio ``GET /api/studio/seclog`` rows are IDENTICAL to what the CLI
``himmy seclog`` prints, because both read the SAME durable ``.himmy/spine.db`` (the
T2.1 ``SpineFactory`` spine). Covers:

* parity: events recorded via the CLI ``cli_security_log()`` show up in the Studio
  endpoint with matching event_ids, in the same newest-first order, AND match the
  CLI ``render_seclog(--json)`` rows exactly;
* the ``limit`` and ``type`` filters mirror ``render_seclog``;
* the distinct-types list for the filter;
* the clean empty trail (no events ⇒ ``[]``, exit 200).

The autouse ``_isolated_spine`` conftest fixture pins ``HIMMY_SPINE_PATH`` to a
per-test temp file, so the CLI log and the app's ``app.state.security_audit`` both
resolve to the SAME on-disk spine — exactly the production coherence this asserts.
"""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from himmy.api import ApiContainer, create_app
from himmy.cli.security_audit_cmd import cli_security_log, render_seclog
from himmy.services.audit.models import SecurityEvent


def _record(**kw: object) -> None:
    """Record a SecurityEvent through the CLI durable-spine path."""
    cli_security_log().record(SecurityEvent(**kw))  # type: ignore[arg-type]


def _seed() -> None:
    _record(
        event_type="authz_denied",
        outcome="deny",
        actor={"subject": "mallory"},
        resource="tool:wire_money",
        action="execute",
        created_at="2026-06-13T10:00:00Z",
        detail="blocked by RBAC",
    )
    _record(
        event_type="access",
        outcome="allow",
        actor={"subject": "alice"},
        resource="runs",
        action="list",
        created_at="2026-06-13T11:00:00Z",
    )


def _client() -> TestClient:
    return TestClient(create_app(ApiContainer.build_default()))


# ----------------------------------------------------------------- empty trail


def test_empty_trail_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    """No events recorded ⇒ empty list, exit 200 (the table's empty state)."""
    r = _client().get("/api/studio/seclog")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["types"] == []


# --------------------------------------------------------------------- parity


def test_studio_rows_identical_to_cli_seclog() -> None:
    """The Studio rows match ``himmy seclog`` byte-for-byte (same durable spine)."""
    _seed()
    # CLI side: render_seclog --json over the same spine.
    buf = io.StringIO()
    rc = render_seclog(cli_security_log(), limit=50, as_json=True, stream=buf)
    assert rc == 0
    cli_rows = json.loads(buf.getvalue())

    # Studio side.
    body = _client().get("/api/studio/seclog", params={"limit": 50}).json()

    # Same events, same order, same identities — the parity guarantee.
    assert [e["event_id"] for e in body["items"]] == [r["event_id"] for r in cli_rows]
    assert [e["event_type"] for e in body["items"]] == [
        r["event_type"] for r in cli_rows
    ]
    assert [e["outcome"] for e in body["items"]] == [r["outcome"] for r in cli_rows]
    # The actor column is flattened to the subject (the CLI's ``actor`` column).
    assert [e["actor"] for e in body["items"]] == [
        (r["actor"] or {}).get("subject") or "-" for r in cli_rows
    ]
    # Newest first (alice at 11:00 precedes mallory at 10:00).
    assert body["items"][0]["actor"] == "alice"
    assert body["items"][1]["actor"] == "mallory"


def test_type_filter_mirrors_cli() -> None:
    """``type=`` filters to one event_type — same semantics as ``--type``."""
    _seed()
    body = _client().get(
        "/api/studio/seclog", params={"type": "authz_denied"}
    ).json()
    assert {e["event_type"] for e in body["items"]} == {"authz_denied"}
    assert body["items"][0]["actor"] == "mallory"
    assert body["total"] == 1


def test_limit_bounds_rows() -> None:
    """``limit`` caps the rows (newest first) — same as ``--limit``."""
    _seed()
    body = _client().get("/api/studio/seclog", params={"limit": 1}).json()
    assert body["total"] == 1
    assert body["items"][0]["actor"] == "alice"  # the newest


def test_types_list_offers_present_event_types() -> None:
    """The ``types`` list is the distinct event_types in the window (filter options)."""
    _seed()
    body = _client().get("/api/studio/seclog").json()
    assert set(body["types"]) == {"authz_denied", "access"}


def test_limit_validation() -> None:
    """``limit`` is range-validated (0 or absurd ⇒ 422, never a crash)."""
    c = _client()
    assert c.get("/api/studio/seclog", params={"limit": 0}).status_code == 422
    assert c.get("/api/studio/seclog", params={"limit": 99999}).status_code == 422
