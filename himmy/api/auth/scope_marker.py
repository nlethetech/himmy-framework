"""The data-scoping marker: a stamp that declares a read path enforces tenant scope.

**Why this exists (the by-construction failure mode it kills).** Tenant(+subject)
data scoping is enforced router-by-router today — a read handler is *trusted* to
remember to call :func:`~himmy.api.auth.context.resolve_workspace` /
:func:`~himmy.api.auth.context.authorize_object` /
:func:`~himmy.api.auth.context.narrow_subject` (and to thread the resolved scope
into its store read) before returning objects. A NEW read route that forgets is a
silent cross-tenant / cross-subject (IDOR/BOLA) leak. The existing static AST guard
(``tests/api/test_v1_workspace_scoping_guard.py``) catches this ONLY on ``/v1``-prefixed
routers — the entire ``/api/studio/*`` family and the process-local missions/notify
registries are never inspected, so each hardening round historically leaked one more
Studio surface.

This module is the recurrence guard's load-bearing half. :func:`scoped_read` is a
dependency that does nothing at runtime but carries the ``_scoped_read`` marker (mirroring
the ``_rbac_permission`` marker :func:`himmy.api.auth.rbac.require_permission` stamps for
the RBAC coverage gate). A read route declares ``Depends(scoped_read)`` — or sits under a
router whose ``dependencies=[...]`` includes it — to ASSERT "this path derives its scope
from the verified principal and refuses out-of-scope rows". The whole-app tenant-scope
coverage gate (``tests/api/test_tenant_scope_coverage.py``) then walks the BUILT app's
runtime route tree (Studio + missions + notify included, closing the AST guard's blind
spot) and requires every GET/list/aggregate leaf to EITHER carry this marker OR be on one
of three documented, reviewed allow-lists. A new unscoped read fails the build.

**Offline invariant — byte-unchanged.** :func:`scoped_read` is a pure no-op dependency
(it returns ``None`` and touches nothing); adding it to a route changes no behaviour on
any path, offline or online. It is purely an introspection stamp — the *actual* scoping
is still performed by the handler's ``resolve_workspace`` / object-axis calls, which are
themselves no-ops for an ANONYMOUS / ``all_tenants`` principal. So the single-box
zero-config path is byte-unchanged: the marker is inert, the scoping it advertises is a
no-op, and the coverage gate only asserts the invisibility property when tenant keys are
configured.
"""

from __future__ import annotations


async def scoped_read() -> None:
    """A no-op dependency that STAMPS a read route as enforcing tenant(+subject) scope.

    Declared on a read route (``Depends(scoped_read)``) or its router
    (``dependencies=[Depends(scoped_read)]``) to assert that the route derives its
    effective scope from the verified principal (never from client-supplied
    ``workspace_id``) and refuses out-of-scope rows — the property the whole-app
    tenant-scope coverage gate enforces by construction. Does nothing at runtime; its
    only effect is the ``_scoped_read`` marker the gate detects in the route's resolved
    dependency graph (exactly like ``require_permission``'s ``_rbac_permission`` marker).

    Parameterless on purpose: it must add NO query/path parameter to the route it stamps
    (FastAPI would otherwise infer one from a typed argument), so the OpenAPI shape and
    request contract of every stamped route is byte-unchanged. A strict NO-OP on every
    path — offline / ``all_tenants`` / tenant-bound alike. The real scoping is the
    handler's ``resolve_workspace``/``authorize_object``/``narrow_subject`` calls
    (themselves no-ops for an unrestricted principal), not this stamp.
    """
    return None


# The marker the coverage gate looks for in a route's dependency tree (mirrors the
# ``_rbac_permission`` attribute on the require_permission/studio_permission closures).
scoped_read._scoped_read = True  # type: ignore[attr-defined]


__all__ = ["scoped_read"]
