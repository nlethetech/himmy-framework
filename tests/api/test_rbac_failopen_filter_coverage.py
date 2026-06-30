"""failopen-lint — the structural gate against the fail-OPEN ``in (None, scope)`` filter shape.

The recurring tenancy root cause across this hardening effort was a *lenient* tenant
filter that fails OPEN on an UNSTAMPED record:

    rows = [r for r in rows if r.workspace_id in (None, workspace_id)]

For a tenant-bound caller this admits every ``workspace_id IS NULL`` row (an offline /
admin / other-tenant unstamped record) into the result — a cross-tenant leak that sits
BELOW the route's ``resolve_workspace`` scoping gate, so the route-coverage gate
(``tests/api/test_rbac_route_coverage.py``) and the /v1 scoping AST guard
(``tests/api/test_v1_workspace_scoping_guard.py``) both pass while the read still leaks.
The fixes replaced these with STRICT ``== workspace_id`` for the tenant-bound branch,
keeping the lenient form ONLY behind an ``all_tenants`` (offline / admin) guard.

SCOPE OF THIS GATE — be precise: this lint covers the ``<scope-expr> in (None, ...)``
*membership-LITERAL* shape ONLY. It is **not** full coverage of the broader "admit a
NULL-tenant row" class. The two SIBLING shapes of that same class — a record-side
``<scope> is None or <scope> == workspace_id`` BoolOp (e.g.
``himmy/api/studio_approvals.py`` ``list_pending``,
``himmy/api/studio_tenant_scope.py:row_in_scope``) and the SQL
``... OR <column> IS NULL`` fragment (``himmy/api/studio_tenant_scope.py:scope_clause``)
— are deliberately NOT matched here. Those are the audited, intentional legacy-NULL-visibility
design: a ``NULL`` stamp predates tenant binding and belongs to no tenant, so a bound reader
is allowed to see its own rows AND those legacy rows (mirroring
:func:`himmy.api.auth.context.authorize_studio_object`). ``studio_tenant_scope.row_in_scope``
/ ``scope_clause`` are the AUDITED HOME of the ``IS NULL`` variant; review NULL-visibility
changes there, not here. Keeping this gate scoped to the literal shape keeps it a precise,
low-false-positive recurrence guard rather than a blanket NULL-visibility ban.

Within its shape, this test is the recurrence guard — the spiritual sibling of the
route-coverage and tool-gate structural gates. It walks the reader/store/service SOURCE
TREE and FAILS if a module contains a record-side ``<tenant-field> in (None, ...)``
membership filter UNLESS that site is one of:

* **``all_tenants``-gated** — the lenient filter is lexically inside the BODY of an ``if``
  branch whose test references ``all_tenants`` (the established offline/admin carve-out, so a
  tenant-bound caller takes the strict ``else`` arm and never reaches it); or
* **an annotated genuinely-global opt-out** — the site carries a ``# rbac-allow-global``
  comment AND its ``(module, anchor)`` is enumerated in :data:`_GLOBAL_ROW_OPT_OUTS` with a
  written reason (e.g. a None-stamped global ops security event that has no tenant
  dimension at all).

A NEW sibling reader that re-introduces a bare ``in (None, scope)`` fail-open filter —
without an ``all_tenants`` guard and without a reasoned, annotated opt-out — FAILS here, so
this SHAPE cannot reappear unnoticed. The analyzer is self-tested against a synthetic
fail-open / strict / guarded triple (and the swapped-arm ``else``-body case) so it genuinely
WOULD catch a real omission.
"""

from __future__ import annotations

import ast
from pathlib import Path

import himmy

#: Tenant/object-scope field names whose record-side None-membership is the fail-open class.
#: A membership test ``<expr> in (None, ...)`` is only suspect when ``<expr>`` reads one of
#: these scope dimensions off a RECORD (``r.workspace_id`` / ``r.metadata.get("tenant_id")``
#: / a local named ``workspace_id``), not e.g. an unrelated ``status in (None, "X")``.
_SCOPE_TOKENS: frozenset[str] = frozenset(
    {"workspace_id", "tenant_id", "tenant", "subject_id"}
)

#: The variable whose presence in an enclosing ``if`` test marks the lenient filter as the
#: documented offline/admin carve-out (a tenant-bound caller takes the strict ``else`` arm).
_OFFLINE_GUARD = "all_tenants"

#: The in-source annotation a genuinely-global opt-out site MUST carry on/above the filter.
_OPT_OUT_MARKER = "rbac-allow-global"

#: Source subtrees that own tenant-scoped reads (stores, the application service layer, and
#: the BFF routers/readers). These are where a fail-open filter would actually leak.
_SCANNED_SUBDIRS: tuple[str, ...] = (
    "services/storage",
    "services/audit",
    "services/governance",
    "services/evaluation",
    "services/knowledge",
    "services/learning",
    "application",
    "api",
)

#: Genuinely-global rows whose None stamp means "no tenant dimension at all" and is benign to
#: surface to any tenant-bound caller — the ONLY sanctioned exceptions to the strict gate.
#: Keyed by ``(module_key, anchor)`` where ``anchor`` is a stable token on the filter line;
#: each value is the written reason. A site here MUST also carry the ``# rbac-allow-global``
#: annotation in-source (asserted by :func:`test_global_opt_outs_are_annotated_in_source`), so
#: the carve-out is visible at the call site, not only in this registry. A NEW entry is a
#: deliberate, reviewed decision that the row truly has no tenant to leak.
_GLOBAL_ROW_OPT_OUTS: dict[tuple[str, str], str] = {
    (
        "himmy/api/routers/audit.py",
        'r.metadata.get("workspace_id") in (None, workspace_id)',
    ): (
        "a None-stamped SECURITY_EVENT is a global/ops security event with no tenant "
        "dimension and no cross-tenant subject_refs; benign to surface to any auditor "
        "(the PRIVACY_AUDIT_REPORT branch stays STRICT == workspace_id)"
    ),
}


def _himmy_root() -> Path:
    return Path(himmy.__file__).resolve().parent


def _module_key(path: Path, root: Path) -> str:
    return "himmy/" + str(path.relative_to(root)).replace("\\", "/")


def _scanned_files(root: Path) -> list[Path]:
    """Every reader/store/service ``.py`` under the scoped-read subtrees."""
    files: list[Path] = []
    for sub in _SCANNED_SUBDIRS:
        files.extend(sorted((root / sub).rglob("*.py")))
    return files


def _tuple_has_none(node: ast.expr) -> bool:
    """Whether ``node`` is a ``(None, ...)`` / ``[None, ...]`` literal containing ``None``."""
    if not isinstance(node, (ast.Tuple, ast.List)):
        return False
    return any(
        isinstance(elt, ast.Constant) and elt.value is None for elt in node.elts
    )


def _names_in(node: ast.expr) -> set[str]:
    """All identifier-ish tokens appearing in ``node`` (attr/name/subscript-constant)."""
    tokens: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            tokens.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            tokens.add(sub.attr)
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            # ``r.metadata.get("workspace_id")`` — the scope name is a string constant.
            tokens.add(sub.value)
    return tokens


def _lhs_reads_scope(lhs: ast.expr) -> bool:
    """Whether the membership LHS reads a record-side tenant/object-scope dimension."""
    return bool(_names_in(lhs) & _SCOPE_TOKENS)


def _failopen_compares(tree: ast.AST) -> list[ast.Compare]:
    """Every ``<scope-expr> in (None, ...)`` membership test in the module.

    A fail-open candidate: an ``ast.Compare`` with a single ``in`` operator whose
    right-hand side is a ``(None, ...)`` literal and whose left-hand side reads a
    record-side scope field (so ``status in (None, 'X')`` and the like are not flagged).
    """
    out: list[ast.Compare] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        if len(node.ops) != 1 or not isinstance(node.ops[0], ast.In):
            continue
        rhs = node.comparators[0]
        if not _tuple_has_none(rhs):
            continue
        if _lhs_reads_scope(node.left):
            out.append(node)
    return out


class _OfflineGuardLocator(ast.NodeVisitor):
    """Records line spans of every ``if``/``elif`` branch gated on ``all_tenants``.

    A lenient filter is the sanctioned offline/admin carve-out when it lives inside an
    ``if`` whose test references ``all_tenants`` — the tenant-bound caller takes the strict
    ``else`` arm. We collect ``(start, end)`` line spans of the BODY of every such branch so
    a filter's line can be tested for containment, robust to nesting and to the filter being
    a list-comprehension spanning multiple lines.

    Crucially the span covers the ``if all_tenants:`` BODY ONLY — never its ``else``/``elif``
    arms, which are the path a TENANT-BOUND caller takes. A lenient filter swapped into the
    ``else`` arm (the strict arm's home) must therefore be flagged as an offender, not
    sanctioned by proximity to the guard. We still recurse into ``orelse`` via
    :meth:`generic_visit` so a nested guarded ``if`` there is handled normally.
    """

    def __init__(self) -> None:
        self.guarded_spans: list[tuple[int, int]] = []

    def visit_If(self, node: ast.If) -> None:  # noqa: N802 - ast visitor name
        if _OFFLINE_GUARD in _names_in(node.test):
            body_nodes = node.body
            if body_nodes:
                start = body_nodes[0].lineno
                # The guarded span is the ``if all_tenants:`` BODY only. Walking each body
                # statement (NOT ``node`` itself) excludes the ``else``/``elif`` arms, where a
                # tenant-bound caller lands — a lenient filter there is a real offender.
                end = max(
                    getattr(n, "end_lineno", n.lineno) or n.lineno
                    for stmt in body_nodes
                    for n in ast.walk(stmt)
                    if hasattr(n, "lineno")
                )
                self.guarded_spans.append((start, end))
        self.generic_visit(node)


def _guarded_spans(tree: ast.AST) -> list[tuple[int, int]]:
    loc = _OfflineGuardLocator()
    loc.visit(tree)
    return loc.guarded_spans


def _line_in_spans(line: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= line <= end for start, end in spans)


def _source_anchor(text: str, node: ast.Compare) -> str:
    """The exact source slice of the compare node (its stable opt-out anchor)."""
    return ast.get_source_segment(text, node) or ""


def _line_window_has_marker(lines: list[str], node: ast.Compare) -> bool:
    """Whether the ``# rbac-allow-global`` marker is on/just-above the filter site.

    Scans the filter's own line span plus a few lines above it (where a multi-line
    annotating comment naturally sits), so an annotated opt-out is recognised regardless of
    whether the marker is inline or in a preceding comment block.
    """
    start = node.lineno
    end = getattr(node, "end_lineno", node.lineno) or node.lineno
    # ``lines`` is 0-indexed; node line numbers are 1-based. Scan a few lines above the
    # filter (where a multi-line annotating comment sits) through its last line.
    lo = max(0, start - 8 - 1)
    window = lines[lo:end]
    return any(_OPT_OUT_MARKER in line for line in window)


def _iter_offenders() -> list[str]:
    """``module:line`` for every fail-open None-stamp filter that is not sanctioned."""
    root = _himmy_root()
    offenders: list[str] = []
    for path in _scanned_files(root):
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        compares = _failopen_compares(tree)
        if not compares:
            continue
        key_mod = _module_key(path, root)
        spans = _guarded_spans(tree)
        lines = text.splitlines()
        for node in compares:
            if _line_in_spans(node.lineno, spans):
                continue  # offline/admin carve-out: tenant-bound caller takes strict arm
            anchor = _source_anchor(text, node)
            registry_key = (key_mod, anchor)
            if registry_key in _GLOBAL_ROW_OPT_OUTS and _line_window_has_marker(
                lines, node
            ):
                continue  # annotated, reasoned genuinely-global opt-out
            offenders.append(f"{key_mod}:{node.lineno}  ::  {anchor}")
    return offenders


# --------------------------------------------------------------------------- the gate


def test_no_unsanctioned_failopen_in_none_literal_filter() -> None:
    """No tenant-scoped reader admits unstamped rows via a bare ``in (None, scope)`` filter.

    Gates the ``<scope-expr> in (None, ...)`` membership-LITERAL shape only (see the module
    docstring: the ``is None or ==`` BoolOp and the SQL ``OR ... IS NULL`` variants are the
    audited, intentional legacy-NULL-visibility design homed in
    ``himmy.api.studio_tenant_scope`` and are out of scope here). Every record-side
    ``<scope> in (None, ...)`` filter in the reader/store/service tree must be either gated by
    ``all_tenants`` (offline/admin carve-out, so a tenant-bound caller takes the strict
    ``== workspace_id`` arm) or an annotated, reasoned genuinely-global opt-out in
    :data:`_GLOBAL_ROW_OPT_OUTS`. A new sibling reader that re-introduces this fail-open
    LITERAL pattern surfaces here.
    """
    offenders = _iter_offenders()
    assert not offenders, (
        "these tenant-scoped readers admit UNSTAMPED rows to a tenant-bound caller via a "
        "fail-OPEN `<scope> in (None, ...)` filter (a cross-tenant leak below the route's "
        "scoping gate). Make the tenant-bound branch STRICT `== workspace_id` and keep the "
        "lenient form only behind an `if all_tenants:` guard — or, if the None-stamped row "
        "is genuinely global (no tenant dimension), annotate the site `# rbac-allow-global` "
        f"and register it in _GLOBAL_ROW_OPT_OUTS with a reason: {offenders}"
    )


def test_gate_actually_inspects_the_known_filter_sites() -> None:
    """Sanity: the walk really reaches the modules that carry None-stamp filters.

    Guards against the gate passing vacuously (e.g. a refactor moving these readers out of
    the scanned subtree). The dashboard/eval service and the audit router are the known
    homes of sanctioned ``in (None, ...)`` filters.
    """
    root = _himmy_root()
    modules_with_filters = {
        _module_key(p, root)
        for p in _scanned_files(root)
        if _failopen_compares(ast.parse(p.read_text(encoding="utf-8")))
    }
    assert "himmy/application/services.py" in modules_with_filters
    assert "himmy/api/routers/audit.py" in modules_with_filters


def test_services_failopen_filters_are_all_tenants_gated() -> None:
    """Regression: the dashboard/eval ``in (None, ...)`` filters stay behind all_tenants.

    The scope-r3/r4 fixes made the tenant-bound branch strict and kept the lenient filter
    ONLY inside ``if all_tenants:``. This asserts every None-stamp filter in the service
    module is inside an ``all_tenants``-guarded span — so dropping the guard (regressing to a
    bare lenient filter for tenant-bound callers) re-fails the gate above.
    """
    root = _himmy_root()
    path = root / "application" / "services.py"
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    spans = _guarded_spans(tree)
    compares = _failopen_compares(tree)
    assert compares, "expected the dashboard/eval None-stamp filters to be present"
    for node in compares:
        assert _line_in_spans(node.lineno, spans), (
            f"services.py None-stamp filter at line {node.lineno} is no longer inside an "
            "`if all_tenants:` guard — a tenant-bound caller would now fail OPEN"
        )


def test_audit_global_opt_out_is_annotated() -> None:
    """The audit security-event opt-out carries its in-source ``rbac-allow-global`` marker."""
    root = _himmy_root()
    path = root / "api" / "routers" / "audit.py"
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tree = ast.parse(text)
    annotated = [
        node
        for node in _failopen_compares(tree)
        if _line_window_has_marker(lines, node)
    ]
    assert annotated, "the audit SECURITY_EVENT global opt-out lost its rbac-allow-global note"


def test_global_opt_outs_are_annotated_in_source() -> None:
    """Every registered global opt-out actually exists AND carries its in-source marker.

    Keeps :data:`_GLOBAL_ROW_OPT_OUTS` honest: a registry entry that no longer matches a
    real, annotated filter site is a stale exemption that would silently widen the gate.
    """
    root = _himmy_root()
    matched: set[tuple[str, str]] = set()
    for path in _scanned_files(root):
        key_mod = _module_key(path, root)
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        tree = ast.parse(text)
        for node in _failopen_compares(tree):
            anchor = _source_anchor(text, node)
            if (key_mod, anchor) in _GLOBAL_ROW_OPT_OUTS:
                assert _line_window_has_marker(lines, node), (
                    f"{key_mod}:{node.lineno} is registered as a global opt-out but is not "
                    f"annotated `# {_OPT_OUT_MARKER}` at the site"
                )
                matched.add((key_mod, anchor))
    stale = set(_GLOBAL_ROW_OPT_OUTS) - matched
    assert not stale, f"stale _GLOBAL_ROW_OPT_OUTS entries (no matching source site): {stale}"


# ------------------------------------------------------------------ analyzer self-tests


def test_analyzer_flags_a_synthetic_failopen_filter() -> None:
    """The analyzer FAILS a bare record-side ``in (None, scope)`` filter (not vacuous)."""
    failopen = ast.parse(
        "def read(rows, workspace_id):\n"
        "    return [r for r in rows if r.workspace_id in (None, workspace_id)]\n"
    )
    compares = _failopen_compares(failopen)
    assert compares, "synthetic fail-open filter should be detected"
    # ...and it is NOT inside an all_tenants guard, so it would be reported.
    assert not _line_in_spans(compares[0].lineno, _guarded_spans(failopen))


def test_analyzer_passes_a_strict_filter() -> None:
    """A strict ``== workspace_id`` filter is NOT flagged as fail-open."""
    strict = ast.parse(
        "def read(rows, workspace_id):\n"
        "    return [r for r in rows if r.workspace_id == workspace_id]\n"
    )
    assert not _failopen_compares(strict)


def test_analyzer_passes_an_all_tenants_guarded_filter() -> None:
    """A lenient filter inside ``if all_tenants:`` is recognised as the sanctioned arm."""
    guarded = ast.parse(
        "def read(rows, workspace_id, all_tenants):\n"
        "    if all_tenants:\n"
        "        rows = [r for r in rows if r.workspace_id in (None, workspace_id)]\n"
        "    else:\n"
        "        rows = [r for r in rows if r.workspace_id == workspace_id]\n"
        "    return rows\n"
    )
    compares = _failopen_compares(guarded)
    assert compares, "the lenient branch should still be detected as a candidate"
    spans = _guarded_spans(guarded)
    # The single lenient (None-tuple) candidate must fall inside the all_tenants span.
    assert _line_in_spans(compares[0].lineno, spans)


def test_analyzer_flags_lenient_filter_in_else_arm_of_all_tenants() -> None:
    """A lenient filter in the ``else`` arm of ``if all_tenants:`` is an OFFENDER, not safe.

    The ``else``/``elif`` arm of ``if all_tenants:`` is exactly the path a TENANT-BOUND caller
    takes — a lenient ``in (None, scope)`` filter there fails OPEN for that caller. This is the
    swapped-arm regression (lenient and strict arms transposed in ``services.py``): the guard
    span must NOT swallow the ``else`` body, so the filter is reported.
    """
    swapped = ast.parse(
        "def read(rows, workspace_id, all_tenants):\n"
        "    if all_tenants:\n"
        "        rows = [r for r in rows if r.workspace_id == workspace_id]\n"
        "    else:\n"
        "        rows = [r for r in rows if r.workspace_id in (None, workspace_id)]\n"
        "    return rows\n"
    )
    compares = _failopen_compares(swapped)
    assert compares, "the lenient else-arm branch should be detected as a candidate"
    spans = _guarded_spans(swapped)
    # The lenient filter is in the `else` arm (tenant-bound path) — it must NOT be inside the
    # `if all_tenants:` guarded span, so it would be reported as an offender.
    assert not _line_in_spans(compares[0].lineno, spans), (
        "a lenient filter in the `else` arm of `if all_tenants:` was wrongly treated as "
        "guarded — the swapped-arm fail-open regression would slip through"
    )


def test_analyzer_handles_nested_guard_in_else_arm() -> None:
    """A genuinely guarded ``if all_tenants:`` NESTED inside an ``else`` arm is still honored.

    Narrowing the span to the ``if`` body must not stop recursion: a nested ``all_tenants``
    guard living in some outer ``else`` arm still sanctions its own body.
    """
    nested = ast.parse(
        "def read(rows, workspace_id, mode, all_tenants):\n"
        "    if mode:\n"
        "        rows = [r for r in rows if r.workspace_id == workspace_id]\n"
        "    else:\n"
        "        if all_tenants:\n"
        "            rows = [r for r in rows if r.workspace_id in (None, workspace_id)]\n"
        "        else:\n"
        "            rows = [r for r in rows if r.workspace_id == workspace_id]\n"
        "    return rows\n"
    )
    compares = _failopen_compares(nested)
    assert compares, "the nested lenient branch should be detected as a candidate"
    spans = _guarded_spans(nested)
    assert _line_in_spans(compares[0].lineno, spans), (
        "a lenient filter inside a nested `if all_tenants:` (itself in an else arm) should "
        "still be recognised as the sanctioned offline/admin carve-out"
    )


def test_analyzer_ignores_non_scope_membership() -> None:
    """A non-scope ``status in (None, 'X')`` membership is not a fail-open candidate."""
    unrelated = ast.parse(
        "def read(rows):\n"
        "    return [r for r in rows if r.status in (None, 'DONE')]\n"
    )
    assert not _failopen_compares(unrelated)
