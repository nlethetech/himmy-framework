"""The auditable "all features everywhere" coverage guard (Tier-3 capstone).

This test ENCODES the *Feature × Interface Coverage Matrix* from the program plan
(``himmy-interface-coherence-parity.md``) **as data** and asserts, for every cell
the matrix marks as present (``✓`` target-this-program or ``=`` already-connected),
that the corresponding surface actually EXISTS on that interface:

* **/v1 (REST)** — the route is registered in the FastAPI app built by
  :func:`himmy.api.app.create_app` (introspected via ``app.routes``).
* **CLI** — the subcommand is registered in the ``himmy`` argparse parser
  (introspected via :func:`himmy.cli.__main__._command_names`).
* **Studio (GUI)** — the BFF route is registered in the same app **and**
  (best-effort) a frontend artifact exists under ``studio/src`` (a *non-empty*
  feature-named route in ``main.tsx`` or a screen/component/lib module that renders
  the surface). The always-present "/" home/index route is deliberately NOT a
  satisfying artifact: a feature-specific cell can never "pass" on the home screen,
  so a genuinely-absent Studio surface (e.g. Recommendations, Context — which exist
  on /v1 + CLI but were never built into Studio) is encoded as a documented
  principled-omit, never as a hollow ``route=""`` cell.

A ``✓``/``=`` cell whose surface is missing FAILS the test. A principled-omit
(``⊘``) MUST carry a documented footnote reason (encoded inline on the cell), so
the matrix can never silently drop a feature from an interface: adding a feature
to one interface without its siblings makes this build go red.

The encoding is deliberately exhaustive — every row of the plan's matrix is here,
updated to reflect everything Tier 0-3 actually shipped (knowledge/teams/workflows/
routines/models on /v1; routines/models on CLI; guardrails/seclog/telegram on
Studio). The footnote text mirrors the plan's FOOTNOTES block.

The "a deliberately-removed route makes it FAIL" guarantee is itself tested by
:func:`test_a_removed_v1_route_makes_the_guard_fail` and its CLI/Studio siblings,
which prove the assertion helpers go red when a required surface is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from himmy.api.app import create_app
from himmy.api.route_introspection import collect_route_paths
from himmy.cli.__main__ import _command_names, build_parser

# --------------------------------------------------------------------------- #
# Surface inventories — introspected ONCE from the live app / parser / source.
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STUDIO_SRC = _REPO_ROOT / "studio" / "src"
_STUDIO_MAIN = _STUDIO_SRC / "main.tsx"


def _registered_paths() -> frozenset[str]:
    """Every route path registered in the FastAPI app (templated form, e.g. ``{id}``)."""
    app = create_app()
    return frozenset(p for p in collect_route_paths(app) if p)


def _cli_command_names() -> frozenset[str]:
    """Every top-level ``himmy`` subcommand name (introspected from the parser)."""
    return frozenset(_command_names(build_parser()))


def _studio_route_paths() -> frozenset[str]:
    """Client-side route paths declared in ``studio/src/main.tsx`` (e.g. ``activity``).

    Parsed from the ``path: "..."`` entries in the ``createBrowserRouter`` table.

    The index route (``index: true`` → the always-present "/" home screen) is
    deliberately NOT added: it is not a dedicated surface for any feature, and
    :func:`assert_studio` rejects ``route=""`` so a feature-specific cell can never
    "pass" on the home screen. Encoding the index here would re-open that loophole.
    Only non-empty, feature-named client-side routes are returned.
    """
    if not _STUDIO_MAIN.is_file():  # pragma: no cover - studio source always present
        return frozenset()
    text = _STUDIO_MAIN.read_text(encoding="utf-8")
    paths = {p for p in re.findall(r"""path:\s*["']([^"']*)["']""", text) if p}
    return frozenset(paths)


def _studio_source_blob() -> str:
    """Concatenated text of every ``studio/src`` source file (best-effort artifact probe).

    Used to confirm a Studio surface that renders *inside* an existing screen (a tab
    or embedded card — e.g. the Guardrails viewer in Tools, the seclog viewer in
    Privacy, the Telegram listener card in Connections) actually has a frontend
    artifact, even when it has no dedicated top-level route.
    """
    if not _STUDIO_SRC.is_dir():  # pragma: no cover - studio source always present
        return ""
    parts: list[str] = []
    for path in sorted(_STUDIO_SRC.rglob("*")):
        if path.suffix in (".ts", ".tsx", ".js", ".jsx") and path.is_file():
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


# Introspected inventories (module-level so the parametrized matrix can reference them).
_PATHS = _registered_paths()
_CLI = _cli_command_names()
_STUDIO_ROUTES = _studio_route_paths()
_STUDIO_BLOB = _studio_source_blob()


# --------------------------------------------------------------------------- #
# Matrix encoding.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CliTarget:
    """A CLI cell: the subcommand that must be registered (or a principled omit)."""

    command: str | None = None
    omit_reason: str | None = None


@dataclass(frozen=True)
class V1Target:
    """A /v1 cell: a route path that must be registered (or a principled omit)."""

    path: str | None = None
    omit_reason: str | None = None


@dataclass(frozen=True)
class StudioTarget:
    """A Studio cell.

    ``bff_path`` — a BFF route under ``/api/studio`` that must be registered.
    ``route`` — a client-side route in ``main.tsx`` that must exist.
    ``artifact`` — a substring that must appear in ``studio/src`` (for a surface
    embedded in another screen, with no dedicated route). At least one of
    ``bff_path``/``route``/``artifact`` is required for a present cell.
    """

    bff_path: str | None = None
    route: str | None = None
    artifact: str | None = None
    omit_reason: str | None = None


@dataclass(frozen=True)
class MatrixRow:
    """One feature row of the coverage matrix, with a target per interface."""

    feature: str
    cli: CliTarget
    studio: StudioTarget
    v1: V1Target
    footnotes: tuple[str, ...] = field(default_factory=tuple)


# Footnotes (verbatim intent from the plan's FOOTNOTES block) keyed by tag, so a
# principled-omit cell can cite the documented reason and the encoding stays auditable.
FOOTNOTES: dict[str, str] = {
    "1": (
        "CLI agents are file-based (agent.yaml) by design — the CLI's native unit is "
        "the on-disk spec, which `himmy run -f`/`himmy mcp`/`himmy agents` operate on."
    ),
    "2": (
        "Plan mode is interactive in CLI/Studio (the human sees the plan and "
        "continues); on /v1 it is plan=true → the run pauses at PLAN-READY and is "
        "resumable via the same approve endpoint machinery."
    ),
    "3": (
        "CLI knowledge is reached through skill/kb tooling on the on-disk spec "
        "(the spec's knowledge: field auto-ingests); full standalone CLI KB CRUD is "
        "a thin follow-on, not a universal gap."
    ),
    "4": (
        "MCP on /v1 is closed as a per-agent-spec field WITH the tenant-spec "
        "sanitizer: tenant-submitted mcp_servers/tools_module/http_tools are REJECTED "
        "unless operator-provisioned. Standalone tenant-driven /v1/mcp live process "
        "management is the one principled omit — spawning arbitrary stdio subprocesses "
        "from a tenant request is an operator-owned boundary."
    ),
    "5": (
        "Missions/steering is a Studio-native real-time control surface (live token "
        "steering). /v1 omit is principled — programmatic mid-run steering over REST "
        "is a future event-driven concern explicitly out of scope (no event bus this "
        "program). The CLI offers only a thin status seam."
    ),
    "6": (
        "Guardrails on /v1: the catalog is readable via GET /v1/diagnostics and "
        "firings are visible per-run in GET /v1/runs/{id}/events (GUARDRAIL_APPLIED "
        "frames); attaching guardrails is via the agent spec's guardrails: field. No "
        "separate /v1/guardrails control plane — guardrails are not a tenant-stateful "
        "resource."
    ),
    "7": (
        "Telegram inbound on /v1 is omitted by design: Telegram uses getUpdates "
        "long-poll, an in-process always-on listener tied to a single bot token — a "
        "single-user-local operator concern (CLI/Studio), not a per-tenant REST "
        "resource. /v1's inbound equivalent is the signed-HTTP connector webhook."
    ),
    "8": (
        "/v1 connectors = read-only catalog only (coherence plan scope); /v1 "
        "connector CONFIG is explicitly out of scope."
    ),
    "9": (
        "CLI change feed omit is principled — a terminal subcommand is a one-shot "
        "process; it reads the latest durable state at invocation. Live SSE is a "
        "GUI/long-lived-client concern."
    ),
    "single-user-local": (
        "Multi-tenant RBAC / workspace admin and idempotency are /v1's nature: the "
        "CLI and Studio are single-user-local (one operator, one process), so a "
        "tenant-isolation control plane and retry-idempotency keys do not apply."
    ),
    "10": (
        "Recommendations are exposed on /v1 (GET /v1/recommendations) and the CLI "
        "(`himmy recommendations`); Studio has NO dedicated recommendations surface "
        "today. The plan's matrix marked the Studio cell '= recs screen' "
        "aspirationally, but the screen was never built — there is no recommendations "
        "BFF route, no client-side route, and no recommendations artifact anywhere in "
        "studio/src. Encoded as a principled Studio omit rather than certifying a "
        "surface that does not exist; a recs screen is a thin follow-on, not a "
        "shipped surface."
    ),
    "11": (
        "Context fields/snapshots are exposed on /v1 (GET /v1/context/fields) and the "
        "CLI (`himmy context`); Studio has NO dedicated context-fields/snapshot "
        "surface today. The plan's matrix marked the Studio cell '= context UI' "
        "aspirationally, but no such UI exists in studio/src (the only `context` "
        "references are React `createContext`/model-context, unrelated). Encoded as a "
        "principled Studio omit rather than certifying a surface that does not exist."
    ),
}


# The matrix, encoded row-by-row. ``=`` (already-present) and ``✓`` (target this
# program) cells both require the surface to EXIST; ``⊘`` cells set an omit_reason.
MATRIX: tuple[MatrixRow, ...] = (
    MatrixRow(
        feature="Single agent run (one-shot)",
        cli=CliTarget(command="run"),
        studio=StudioTarget(bff_path="/api/studio/run", route="chat"),
        v1=V1Target(path="/v1/runs"),
    ),
    MatrixRow(
        feature="Run history browse (canonical, ONE store)",
        cli=CliTarget(command="runs"),
        studio=StudioTarget(bff_path="/api/studio/runs", route="activity"),
        v1=V1Target(path="/v1/runs"),
    ),
    MatrixRow(
        feature="Interactive chat / thread continuation",
        cli=CliTarget(command="chat"),
        studio=StudioTarget(bff_path="/api/studio/chats", route="chats"),
        v1=V1Target(path="/v1/threads/{thread_id}/messages"),
    ),
    MatrixRow(
        feature="Stored agent defs (build/store/list)",
        cli=CliTarget(command="agents"),
        studio=StudioTarget(bff_path="/api/studio/agents", route="agents"),
        v1=V1Target(path="/v1/agents"),
        footnotes=("1",),
    ),
    MatrixRow(
        feature="Run-by-stored-agent (tools bound)",
        cli=CliTarget(command="agents"),  # via agents + --persist
        studio=StudioTarget(bff_path="/api/studio/run", route="agents"),
        v1=V1Target(path="/v1/runs"),  # POST /v1/runs with agent_id
    ),
    MatrixRow(
        feature="Plan mode / plan-first",
        cli=CliTarget(command="run"),  # run --plan
        studio=StudioTarget(bff_path="/api/studio/run", route="chat"),
        v1=V1Target(path="/v1/runs"),  # plan=true
        footnotes=("2",),
    ),
    MatrixRow(
        feature="Approvals / HITL resume",
        cli=CliTarget(command="run"),  # run pauses interactively
        studio=StudioTarget(bff_path="/api/studio/approvals", route="approvals"),
        v1=V1Target(path="/v1/runs/{run_id}/approve"),
    ),
    MatrixRow(
        feature="Knowledge upload + search",
        cli=CliTarget(command="skill"),  # skill/kb tooling †3
        studio=StudioTarget(
            bff_path="/api/studio/knowledge", route="advanced/knowledge"
        ),
        v1=V1Target(path="/v1/knowledge/{collection}/search"),
        footnotes=("3",),
    ),
    MatrixRow(
        feature="MCP server config",
        cli=CliTarget(command="mcp"),
        studio=StudioTarget(bff_path="/api/studio/mcp/servers", route="mcp"),
        v1=V1Target(path="/v1/agents"),  # per-agent-spec field (sanitized)
        footnotes=("4",),
    ),
    MatrixRow(
        feature="Teams / multi-agent orchestration",
        cli=CliTarget(command="team"),
        studio=StudioTarget(bff_path="/api/studio/teams", route="advanced/teams"),
        v1=V1Target(path="/v1/teams/{team_id}/run"),
    ),
    MatrixRow(
        feature="Workflows execution",
        cli=CliTarget(command="workflow"),
        studio=StudioTarget(
            bff_path="/api/studio/workflows", route="advanced/workflows"
        ),
        v1=V1Target(path="/v1/workflows/{workflow_id}/run"),
    ),
    MatrixRow(
        feature="Routines / scheduler",
        cli=CliTarget(command="routines"),
        studio=StudioTarget(bff_path="/api/studio/routines", route="routines"),
        v1=V1Target(path="/v1/routines"),
    ),
    MatrixRow(
        feature="Missions (steering)",
        cli=CliTarget(command=None, omit_reason=FOOTNOTES["5"]),  # CLI-thin †5
        studio=StudioTarget(bff_path="/api/studio/missions", route="missions"),
        v1=V1Target(omit_reason=FOOTNOTES["5"]),
        footnotes=("5",),
    ),
    MatrixRow(
        feature="Recommendations",
        cli=CliTarget(command="recommendations"),
        # No Studio recommendations surface exists today (the plan's "recs screen" was
        # never built) — documented principled omit, NOT an always-true index route. †10
        studio=StudioTarget(omit_reason=FOOTNOTES["10"]),
        v1=V1Target(path="/v1/recommendations"),
        footnotes=("10",),
    ),
    MatrixRow(
        feature="Context fields/snapshots",
        cli=CliTarget(command="context"),
        # No Studio context-fields/snapshot surface exists today (the plan's "context
        # UI" was never built) — documented principled omit. †11
        studio=StudioTarget(omit_reason=FOOTNOTES["11"]),
        v1=V1Target(path="/v1/context/fields"),
        footnotes=("11",),
    ),
    MatrixRow(
        feature="Dashboard / operational summary",
        cli=CliTarget(command="dashboard"),
        # The operational summary is the runs-analytics BFF, consumed by the Home
        # screen; the registered BFF route IS the verifiable surface (no hollow
        # route="" — the index route certifies nothing on its own).
        studio=StudioTarget(bff_path="/api/studio/runs/analytics"),
        v1=V1Target(path="/v1/dashboard/summary"),
    ),
    MatrixRow(
        feature="Evaluation / bench / compare-grader",
        cli=CliTarget(command="eval"),
        studio=StudioTarget(bff_path="/api/studio/eval/run", route="advanced/eval"),
        v1=V1Target(path="/v1/evaluation/runs"),
    ),
    MatrixRow(
        feature="Model catalog + quick compare",
        cli=CliTarget(command="models"),
        studio=StudioTarget(bff_path="/api/studio/models", route="models"),
        v1=V1Target(path="/v1/models"),
    ),
    MatrixRow(
        feature="Guardrails catalog + firings",
        cli=CliTarget(command="guardrails"),
        # T3e Guardrails viewer renders inside the Tools screen (guardrailsApi client).
        studio=StudioTarget(
            bff_path="/api/studio/guardrails", route="tools", artifact="guardrailsApi"
        ),
        v1=V1Target(path="/v1/diagnostics"),  # †6 catalog via diagnostics
        footnotes=("6",),
    ),
    MatrixRow(
        feature="Security audit (seclog)",
        cli=CliTarget(command="seclog"),
        # T3f seclog viewer renders inside the Privacy screen (getSeclog client).
        studio=StudioTarget(
            bff_path="/api/studio/seclog", route="privacy", artifact="getSeclog"
        ),
        v1=V1Target(path="/v1/audit/events"),
        footnotes=("6",),
    ),
    MatrixRow(
        feature="Lineage / provenance",
        cli=CliTarget(command="lineage"),
        studio=StudioTarget(
            bff_path="/api/studio/lineage/graph", route="advanced/lineage"
        ),
        v1=V1Target(path="/v1/runs/{run_id}/lineage"),
    ),
    MatrixRow(
        feature="Consent / privacy / retention",
        cli=CliTarget(command="consent"),
        studio=StudioTarget(bff_path="/api/studio/privacy/consents", route="privacy"),
        v1=V1Target(path="/v1/consent/grant"),
    ),
    MatrixRow(
        feature="Doctor / diagnostics",
        cli=CliTarget(command="doctor"),
        studio=StudioTarget(bff_path="/api/studio/doctor", route="advanced/doctor"),
        v1=V1Target(path="/v1/diagnostics"),
    ),
    MatrixRow(
        feature="Telegram inbound listener",
        cli=CliTarget(command="telegram"),
        # T3g listener manager renders as a card in the Connections screen.
        studio=StudioTarget(
            bff_path="/api/studio/telegram/start",
            route="connections",
            artifact="TelegramListenerCard",
        ),
        v1=V1Target(omit_reason=FOOTNOTES["7"]),
        footnotes=("7",),
    ),
    MatrixRow(
        feature="Connectors (catalog/configure/inbound)",
        cli=CliTarget(command="connectors"),
        studio=StudioTarget(bff_path="/api/studio/connections", route="connections"),
        v1=V1Target(path="/v1/connectors"),  # read-only catalog †8
        footnotes=("8",),
    ),
    MatrixRow(
        feature="Cross-process live updates",
        cli=CliTarget(command=None, omit_reason=FOOTNOTES["9"]),  # terminal one-shot †9
        # SSE change feed surfaces through the notify BFF (the Studio long-lived client).
        studio=StudioTarget(bff_path="/api/studio/notify"),
        v1=V1Target(path="/v1/runs"),  # data_version poll over the runs surface
        footnotes=("9",),
    ),
    MatrixRow(
        feature="Multi-tenant RBAC / workspace admin",
        cli=CliTarget(command=None, omit_reason=FOOTNOTES["single-user-local"]),
        studio=StudioTarget(omit_reason=FOOTNOTES["single-user-local"]),
        v1=V1Target(path="/v1/audit/events"),  # RBAC is /v1's nature
    ),
    MatrixRow(
        feature="Idempotency on mutating writes",
        cli=CliTarget(command=None, omit_reason=FOOTNOTES["single-user-local"]),
        studio=StudioTarget(omit_reason=FOOTNOTES["single-user-local"]),
        v1=V1Target(path="/v1/runs"),  # /v1 idempotency_key on the mutating surface
    ),
)


# --------------------------------------------------------------------------- #
# Per-interface assertion helpers (also used by the negative-control tests).
# --------------------------------------------------------------------------- #


def assert_cli(target: CliTarget, *, registered: frozenset[str]) -> None:
    """A present CLI cell's subcommand must be registered; an omit must be documented."""
    if target.command is None:
        assert target.omit_reason, "a CLI omit must carry a documented reason"
        return
    assert target.command in registered, (
        f"CLI subcommand {target.command!r} is required by the parity matrix but is "
        f"not registered in `himmy` — add a feature to one interface without its "
        f"siblings and this guard goes red."
    )


def assert_v1(target: V1Target, *, registered: frozenset[str]) -> None:
    """A present /v1 cell's route must be registered; an omit must be documented."""
    if target.path is None:
        assert target.omit_reason, "a /v1 omit must carry a documented reason"
        return
    assert target.path in registered, (
        f"/v1 route {target.path!r} is required by the parity matrix but is not "
        f"registered in the FastAPI app."
    )


def assert_studio(
    target: StudioTarget,
    *,
    registered: frozenset[str],
    routes: frozenset[str],
    blob: str,
) -> None:
    """A present Studio cell needs its BFF route (if any) AND a frontend artifact.

    A principled-omit must be documented. Otherwise the BFF route (when the cell
    declares one) must be registered, and at least one frontend artifact — a
    declared client-side route OR a referenced source artifact — must be present.
    """
    if (
        target.bff_path is None
        and target.route is None
        and target.artifact is None
    ):
        assert target.omit_reason, "a Studio omit must carry a documented reason"
        return
    if target.bff_path is not None:
        assert target.bff_path in registered, (
            f"Studio BFF route {target.bff_path!r} is required by the parity matrix "
            f"but is not registered in the FastAPI app."
        )
    # Best-effort frontend artifact: a declared NON-EMPTY client-side route or a
    # referenced source token. The empty-string index route (``""``) is the always-
    # present "/" home screen — it is NOT a dedicated surface for any feature, so it
    # must NEVER satisfy a frontend-artifact assertion (otherwise a feature-specific
    # Studio cell could "pass" on the home screen and the guard could never go red for
    # it). A cell that wants frontend coverage must carry a real route or artifact.
    declares_route = target.route is not None and target.route != ""
    has_route = declares_route and target.route in routes
    has_artifact = target.artifact is not None and target.artifact in blob
    if declares_route or target.artifact is not None:
        assert has_route or has_artifact, (
            f"Studio frontend artifact for this cell is missing under studio/src "
            f"(route={target.route!r}, artifact={target.artifact!r})."
        )
    elif target.route == "" and target.bff_path is None:
        # ``route=""`` with no BFF route and no artifact is the defective pattern the
        # capstone reviewer flagged: it asserts nothing (the index always exists). A
        # cell that genuinely has no dedicated Studio surface must be a documented
        # principled-omit, not a hollow ``route=""`` cell.
        raise AssertionError(
            "a Studio cell may not certify coverage via the empty index route "
            '(route="") alone — declare a real BFF route / client-side route / '
            "artifact, or encode the cell as a documented principled-omit."
        )


# --------------------------------------------------------------------------- #
# The guard tests — one parametrized case per (feature × interface) cell.
# --------------------------------------------------------------------------- #

_ROW_IDS = [row.feature for row in MATRIX]


def test_matrix_is_complete() -> None:
    """Sanity: the matrix covers every feature row exactly once (no dup/drop)."""
    features = [row.feature for row in MATRIX]
    assert len(features) == len(set(features)), "duplicate feature row in the matrix"
    # The plan's matrix has 28 feature rows; encode that as a drift tripwire so a row
    # silently deleted from this file (dropping a feature from the guard) is caught.
    assert len(MATRIX) == 28, (
        f"expected 28 feature rows (the plan's matrix); found {len(MATRIX)} — a row "
        f"was added/removed without updating this tripwire."
    )


@pytest.mark.parametrize("row", MATRIX, ids=_ROW_IDS)
def test_cli_cell_exists(row: MatrixRow) -> None:
    """Every present CLI cell's subcommand is registered (or it is a documented omit)."""
    assert_cli(row.cli, registered=_CLI)


@pytest.mark.parametrize("row", MATRIX, ids=_ROW_IDS)
def test_v1_cell_exists(row: MatrixRow) -> None:
    """Every present /v1 cell's route is registered (or it is a documented omit)."""
    assert_v1(row.v1, registered=_PATHS)


@pytest.mark.parametrize("row", MATRIX, ids=_ROW_IDS)
def test_studio_cell_exists(row: MatrixRow) -> None:
    """Every present Studio cell has its BFF route + a frontend artifact (or omit)."""
    assert_studio(
        row.studio, registered=_PATHS, routes=_STUDIO_ROUTES, blob=_STUDIO_BLOB
    )


def test_every_cell_is_present_or_documented_omit() -> None:
    """No cell is silently blank: each is either a present surface or a documented omit.

    This is the auditable guarantee — a cell with neither a target surface nor an
    omit reason is a coverage hole, which fails here.
    """
    for row in MATRIX:
        for label, ok in (
            ("CLI", row.cli.command is not None or bool(row.cli.omit_reason)),
            ("/v1", row.v1.path is not None or bool(row.v1.omit_reason)),
            (
                "Studio",
                # A present Studio surface needs a BFF route, a NON-EMPTY client-side
                # route, or an artifact token — the empty index route (route="") is
                # hollow and does NOT count (it would re-open the capstone loophole).
                # Absent any of those, the cell MUST be a documented principled-omit.
                row.studio.bff_path is not None
                or (row.studio.route is not None and row.studio.route != "")
                or row.studio.artifact is not None
                or bool(row.studio.omit_reason),
            ),
        ):
            assert ok, (
                f"matrix cell ({row.feature!r}, {label}) is neither a present surface "
                f"nor a documented principled-omit — that is an unexplained gap."
            )


def test_footnotes_referenced_by_rows_exist() -> None:
    """Every footnote tag a row cites resolves to documented text."""
    for row in MATRIX:
        for tag in row.footnotes:
            assert tag in FOOTNOTES, (
                f"row {row.feature!r} cites footnote {tag!r} with no documented text"
            )


# --------------------------------------------------------------------------- #
# Negative controls — PROVE the guard goes red when a required surface is gone.
# These demonstrate the test design's central claim: "add a feature to one
# interface without its siblings -> build fails".
# --------------------------------------------------------------------------- #


def test_a_removed_v1_route_makes_the_guard_fail() -> None:
    """Drop ``/v1/runs`` from the registered set → a required /v1 cell fails."""
    runs_row = next(r for r in MATRIX if r.feature.startswith("Single agent run"))
    crippled = frozenset(p for p in _PATHS if p != "/v1/runs")
    with pytest.raises(AssertionError):
        assert_v1(runs_row.v1, registered=crippled)


def test_a_removed_cli_command_makes_the_guard_fail() -> None:
    """Drop ``routines`` from the CLI set → the routines CLI cell fails."""
    routines_row = next(r for r in MATRIX if r.feature.startswith("Routines"))
    crippled = frozenset(c for c in _CLI if c != "routines")
    with pytest.raises(AssertionError):
        assert_cli(routines_row.cli, registered=crippled)


def test_a_removed_studio_bff_route_makes_the_guard_fail() -> None:
    """Drop ``/api/studio/guardrails`` → the Guardrails Studio cell fails."""
    guard_row = next(r for r in MATRIX if r.feature.startswith("Guardrails"))
    crippled = frozenset(p for p in _PATHS if p != "/api/studio/guardrails")
    with pytest.raises(AssertionError):
        assert_studio(
            guard_row.studio,
            registered=crippled,
            routes=_STUDIO_ROUTES,
            blob=_STUDIO_BLOB,
        )


def test_a_removed_studio_frontend_artifact_makes_the_guard_fail() -> None:
    """Strip the Telegram listener card from the source blob → the Studio cell fails."""
    tg_row = next(r for r in MATRIX if r.feature.startswith("Telegram"))
    crippled_blob = _STUDIO_BLOB.replace("TelegramListenerCard", "REMOVED")
    crippled_routes = frozenset(r for r in _STUDIO_ROUTES if r != "connections")
    with pytest.raises(AssertionError):
        assert_studio(
            tg_row.studio,
            registered=_PATHS,
            routes=crippled_routes,
            blob=crippled_blob,
        )


def test_an_artifact_only_studio_cell_goes_red_when_its_token_is_stripped() -> None:
    """A token-backed Studio cell (no route) must fail once its artifact is gone.

    The Guardrails cell carries a real ``artifact='guardrailsApi'`` and no client-side
    route of its own (it renders inside the Tools screen). Removing the token from the
    source blob and the ``tools`` route from the route set must make the cell go red —
    proving artifact-backed coverage is genuinely verified, not hollow.
    """
    guard_row = next(r for r in MATRIX if r.feature.startswith("Guardrails"))
    assert guard_row.studio.artifact == "guardrailsApi"  # guards the encoding
    crippled_blob = _STUDIO_BLOB.replace("guardrailsApi", "REMOVED")
    crippled_routes = frozenset(r for r in _STUDIO_ROUTES if r != "tools")
    # BFF route stays registered → the failure must come from the missing artifact.
    with pytest.raises(AssertionError):
        assert_studio(
            guard_row.studio,
            registered=_PATHS,
            routes=crippled_routes,
            blob=crippled_blob,
        )


def test_the_empty_index_route_cannot_satisfy_a_studio_cell() -> None:
    """The capstone loophole the reviewer flagged is closed: ``route=""`` is hollow.

    A Studio cell that declares ONLY the empty index route (the defective pattern the
    Recommendations/Context cells used) asserts nothing — the "/" home screen always
    exists. This must now FAIL so such a cell can never silently certify coverage; a
    genuinely-absent Studio surface must instead be a documented principled-omit.
    """
    hollow = StudioTarget(route="")  # no bff_path, no artifact — index only
    with pytest.raises(AssertionError):
        assert_studio(hollow, registered=_PATHS, routes=_STUDIO_ROUTES, blob=_STUDIO_BLOB)
    # Even if the parser were to (re-)introduce "" into the route set, the empty route
    # must STILL not satisfy the cell.
    with pytest.raises(AssertionError):
        assert_studio(
            hollow,
            registered=_PATHS,
            routes=_STUDIO_ROUTES | frozenset({""}),
            blob=_STUDIO_BLOB,
        )
