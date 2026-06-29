"""centralize-tool-gate STRUCTURAL gate: every runtime-building surface reaches the gate.

The deliverable that kills the recurrence class (the spiritual sibling of
``tests/api/test_rbac_route_coverage.py``, which guards that every ROUTE carries a
permission). Tool-capability enforcement lives at ONE chokepoint —
:meth:`himmy.services.tools.service.ToolService._execute_invocation`, which resolves the
EFFECTIVE authorizer from the explicit constructor arg AND the ambient contextvar
(:func:`himmy.services.tools.ambient.resolve_effective_authorizer`). For that chokepoint
to actually gate a given execution path, the path must put an authorizer in scope: either
EXPLICITLY (``build_runtime_for_spec(tool_authorizer=...)`` / a ``ToolService`` built with
one) or AMBIENTLY (wrapping the drive in ``use_tool_authorizer`` / setting the contextvar
/ ``ToolCapabilityAuthorizer.from_request``).

This test walks the SOURCE TREE for every site that builds a runtime (``build_runtime`` /
``build_runtime_for_spec``) and asserts, BY CONSTRUCTION, that each one is covered:

* an EXPLICIT site passes ``tool_authorizer=`` at the call; or
* an AMBIENT site lives in a module that references the ambient gate API (so the drive is
  wrapped in a principal scope); or
* an OFFLINE site is on a small, REASONED allow-list (CLI / benchmark / eval-harness /
  framework factories that never run under an authenticator — a deliberate, documented
  no-op).

A NEW server execution surface that builds a runtime but threads NEITHER an explicit nor an
ambient authorizer — and is not on the offline allow-list — FAILS this gate, so the
"a path forgot the gate" bug cannot ship unnoticed. The analyzer is self-tested against a
synthetic guarded/unguarded pair so it genuinely WOULD catch a real omission.

Belt-and-braces, the chokepoint ALSO fails closed when auth is configured and no authorizer
is in scope (see ``tests/tools/test_ambient_tool_authz.py``), so even a missed site denies
rather than runs un-gated — this gate makes that omission VISIBLE at build time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import himmy

#: The two runtime-factory call names that put a ToolService into an execution path.
_BUILD_CALLS = frozenset({"build_runtime", "build_runtime_for_spec"})

#: Names that, referenced anywhere in a module, prove it threads the ambient/explicit gate.
#: Covers BOTH the ambient API (``use_tool_authorizer`` / ``_active_authorizer`` /
#: ``from_request``) AND the explicit threading vocabulary — a module that names
#: ``tool_authorizer`` (e.g. builds a ``ToolService(..., tool_authorizer=...)`` for a
#: sub-agent, or forwards it through ``**overrides``) is threading the gate even if the
#: keyword does not appear literally on the ``build_runtime`` call.
_GATE_MARKERS = frozenset(
    {
        "use_tool_authorizer",
        "_active_authorizer",
        "resolve_effective_authorizer",
        "from_request",  # ToolCapabilityAuthorizer.from_request(...)
        "tool_authorizer",  # explicit threading (ToolService(...)/overrides forwarding)
    }
)

#: Modules whose runtime build is a DELIBERATE offline no-op: no AccessPolicy is ever wired
#: in their scope, so an authorizer would be ``None`` and the gate inert by design. Each is
#: dev/CLI/test tooling, NOT a multi-tenant server execution surface. A NEW entry here is a
#: reviewed decision that the path never runs under a configured authenticator.
_OFFLINE_ALLOW: frozenset[str] = frozenset(
    {
        # The framework runtime factories themselves (they RECEIVE the authorizer via
        # overrides / args; they do not own a principal scope).
        "himmy/runtime/builder.py",
        "himmy/runtime/from_spec.py",
        # CLI: no AccessPolicy is ever wired on the CLI — byte-identical offline no-op.
        "himmy/cli/commands.py",
        "himmy/cli/orchestrate.py",
        # Dev/bench/eval-harness tooling: no RBAC policy in scope, inert by design.
        "himmy/benchmark/runner.py",
        # Background off-request service surfaces whose control is the chokepoint's
        # process-level FAIL-CLOSED default (they run with no live principal; under a
        # configured authenticator a tool reaching the chokepoint with no authorizer in
        # scope is DENIED, not run un-gated). They are listed (not silently skipped) so a
        # future change that should bind a real least-privilege principal here is a visible,
        # reasoned decision rather than an accidental gap.
        "himmy/api/missions.py",
        "himmy/api/studio_telegram.py",
    }
)

#: Builder modules whose runtime is wrapped in the ambient gate by their REQUEST-BOUNDARY
#: CALLER (the router), not in the builder itself — so the gate marker lives one frame up.
#: ``stream_agent_run`` / ``stream_team_run`` (studio_service) are entered inside
#: ``use_tool_authorizer(...)`` by the studio router (``run`` / ``run_team`` / ``research``
#: / the routine drain); ``studio_approvals.resolve`` is entered inside the approve/reject
#: router's ``use_tool_authorizer``. :func:`test_ambient_by_caller_callers_are_wrapped`
#: asserts those caller modules DO reference the gate, so this indirection stays honest —
#: and the chokepoint's fail-closed default denies anyway if a caller ever drops the wrap.
_AMBIENT_BY_CALLER: frozenset[str] = frozenset(
    {
        "himmy/api/studio_service.py",
        "himmy/api/studio_approvals.py",
    }
)

#: The request-boundary caller modules that MUST wrap the ``_AMBIENT_BY_CALLER`` builders.
_AMBIENT_CALLER_MODULES: frozenset[str] = frozenset(
    {
        "himmy/api/routers/studio.py",
        "himmy/api/routines.py",
    }
)


def _himmy_root() -> Path:
    return Path(himmy.__file__).resolve().parent


def _module_key(path: Path, root: Path) -> str:
    return "himmy/" + str(path.relative_to(root)).replace("\\", "/")


def _has_tool_authorizer_kw(call: ast.Call) -> bool:
    """Whether this build_* call passes a ``tool_authorizer=`` keyword (explicit gate)."""
    return any(kw.arg == "tool_authorizer" for kw in call.keywords)


def _module_references_gate(tree: ast.AST) -> bool:
    """Whether the module names any ambient/explicit gate marker anywhere (ambient cover)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _GATE_MARKERS:
            return True
        if isinstance(node, ast.Attribute) and node.attr in _GATE_MARKERS:
            return True
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in _GATE_MARKERS:
                    return True
    return False


def _build_call_sites(tree: ast.AST) -> list[ast.Call]:
    """Every ``build_runtime(...)`` / ``build_runtime_for_spec(...)`` call in the module."""
    sites: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.attr
            if isinstance(func, ast.Attribute)
            else func.id
            if isinstance(func, ast.Name)
            else None
        )
        if name in _BUILD_CALLS:
            sites.append(node)
    return sites


def _iter_offending_modules() -> list[tuple[str, list[int]]]:
    """Return ``(module_key, [lines])`` for modules with an UNCOVERED build site."""
    root = _himmy_root()
    offenders: list[tuple[str, list[int]]] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        sites = _build_call_sites(tree)
        if not sites:
            continue
        key = _module_key(path, root)
        if key in _OFFLINE_ALLOW or key in _AMBIENT_BY_CALLER:
            continue
        if _module_references_gate(tree):
            continue  # ambient/explicit gate is threaded in this module
        # Otherwise: only OK if EVERY build site explicitly passes tool_authorizer=.
        uncovered = [s.lineno for s in sites if not _has_tool_authorizer_kw(s)]
        if uncovered:
            offenders.append((key, uncovered))
    return offenders


def test_every_runtime_build_surface_reaches_the_tool_gate() -> None:
    """No runtime-building surface ships without putting an authorizer in scope.

    Each ``build_runtime`` / ``build_runtime_for_spec`` site must be EXPLICITLY threaded
    (``tool_authorizer=``), AMBIENTLY covered (module references the ambient gate API), or
    on the reasoned offline allow-list. A new server path that builds a runtime but threads
    neither would surface here.
    """
    # Sanity: the walk actually found the known build surfaces (not vacuously passing).
    # (``services.py`` passes ``build_runtime_for_spec`` as a bare arg to ``to_thread``
    # rather than calling it directly, so it is intentionally not asserted here — its
    # coverage is the ambient/explicit threading the gate-marker walk verifies.)
    root = _himmy_root()
    seen_modules = {
        _module_key(p, root)
        for p in root.rglob("*.py")
        if _build_call_sites(ast.parse(p.read_text(encoding="utf-8")))
    }
    assert "himmy/api/studio_service.py" in seen_modules
    assert "himmy/api/routers/studio_teams.py" in seen_modules
    assert "himmy/api/studio_workflows.py" in seen_modules

    offenders = _iter_offending_modules()
    assert not offenders, (
        "these modules build an agent runtime but thread NEITHER an explicit "
        "tool_authorizer= NOR an ambient gate (use_tool_authorizer / from_request) — a "
        "tool dispatched there runs un-gated under a configured authenticator. Thread the "
        "gate, or add the module to _OFFLINE_ALLOW with a reason if it never runs under "
        f"auth: {offenders}"
    )


def test_known_server_surfaces_are_recognised_as_covered() -> None:
    """Regression: the formerly-ungated Studio surfaces are now recognised as gated.

    These three (Studio chat/team via studio.py, workflow router, and the stream service)
    were the confirmed confused-deputy holes the work package closed; they must read as
    covered so a re-introduction of the gap (dropping the ambient binding) re-fails the
    gate above.
    """
    root = _himmy_root()
    for module in (
        "himmy/api/routers/studio.py",
        "himmy/api/routers/studio_teams.py",
        "himmy/api/studio_workflows.py",
        "himmy/api/routines.py",
    ):
        tree = ast.parse((root / Path(module).relative_to("himmy")).read_text("utf-8"))
        assert _module_references_gate(tree), f"{module} lost its ambient gate threading"


def test_ambient_by_caller_callers_are_wrapped() -> None:
    """Each builder deferred to its caller IS wrapped — the request-boundary caller modules
    reference the ambient gate, so the ``_AMBIENT_BY_CALLER`` indirection is genuine."""
    root = _himmy_root()
    for module in _AMBIENT_CALLER_MODULES:
        tree = ast.parse((root / Path(module).relative_to("himmy")).read_text("utf-8"))
        assert _module_references_gate(tree), (
            f"{module} is supposed to wrap an _AMBIENT_BY_CALLER builder in "
            "use_tool_authorizer but no longer references the gate"
        )


def test_analyzer_flags_a_synthetic_ungated_build() -> None:
    """The analyzer genuinely FAILS a module that builds a runtime with no gate.

    Proves the gate is not vacuously passing: a synthetic module that calls
    ``build_runtime_for_spec`` with no ``tool_authorizer=`` and no ambient marker is
    detected as uncovered, while a guarded one is not.
    """
    ungated = ast.parse("def f(spec):\n    return build_runtime_for_spec(spec)\n")
    explicit = ast.parse(
        "def f(spec, az):\n    return build_runtime_for_spec(spec, tool_authorizer=az)\n"
    )
    ambient = ast.parse(
        "from himmy.services.tools.ambient import use_tool_authorizer\n"
        "def f(spec, az):\n"
        "    with use_tool_authorizer(az):\n"
        "        return build_runtime_for_spec(spec)\n"
    )

    # ungated: a build site with no tool_authorizer= and no gate marker.
    assert _build_call_sites(ungated)
    assert not _module_references_gate(ungated)
    assert any(not _has_tool_authorizer_kw(s) for s in _build_call_sites(ungated))

    # explicit: the build site passes tool_authorizer=.
    assert all(_has_tool_authorizer_kw(s) for s in _build_call_sites(explicit))

    # ambient: the module references the gate API.
    assert _module_references_gate(ambient)


@pytest.mark.parametrize(
    "module",
    [
        "himmy/application/services.py",
        "himmy/application/orchestration_runner.py",
        "himmy/api/connector_inbound.py",
        "himmy/skills/dispatch.py",
        "himmy/toolkit/spawn.py",
    ],
)
def test_core_paths_thread_an_authorizer(module: str) -> None:
    """The core /v1 + spawn paths each thread an authorizer (explicit or ambient)."""
    root = _himmy_root()
    tree = ast.parse((root / Path(module).relative_to("himmy")).read_text("utf-8"))
    explicit = any(
        _has_tool_authorizer_kw(s) for s in _build_call_sites(tree)
    )
    assert explicit or _module_references_gate(tree), module
