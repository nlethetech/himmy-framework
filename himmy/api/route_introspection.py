"""Version-robust route-path introspection for the FastAPI/Starlette app.

FastAPI 0.137 changed ``include_router`` so ``app.routes`` no longer contains
flattened ``APIRoute`` objects: an included router now appears as an
``_IncludedRouter`` wrapper that has NO ``.path`` of its own but exposes its
children via ``.routes``. Starlette 1.x made an analogous change to its route
model. Older FastAPI (0.116–0.136) flattens everything into top-level
``APIRoute`` objects with a ``.path``.

Any code that walks ``app.routes`` looking for ``.path`` therefore silently
drops every nested route on the newer versions. This module provides a single
recursive collector that works across BOTH layouts (and across Starlette
``Mount`` sub-applications): it descends into every object that exposes a
``.routes`` sub-tree, composes ``Mount`` prefixes, and returns every LEAF path.
It is intentionally defensive (``hasattr``/``getattr``) so an unfamiliar route
type from a future version cannot raise — at worst its own path is recorded and
its children, if any, are still descended into.
"""

from __future__ import annotations

from typing import Any

__all__ = ["collect_route_paths"]


def _own_path(route: Any) -> str | None:
    """Return this route's own path string, if it exposes one.

    Across versions a concrete (leaf) route carries the templated path on
    ``path`` (FastAPI ``APIRoute``/Starlette ``Route``) or ``path_format``.
    Wrapper objects (``_IncludedRouter``) and routers expose neither.
    """
    for attr in ("path", "path_format"):
        value = getattr(route, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _mount_prefix(route: Any) -> str:
    """The URL prefix a ``Mount`` contributes to its children (``""`` if none).

    A Starlette ``Mount`` stores the prefix under which its sub-app is mounted on
    ``.path`` (e.g. ``/sub``); its children's paths are relative to it. We
    normalize away a lone trailing slash so ``"/sub" + "/x"`` does not become
    ``"/sub//x"``.
    """
    prefix = getattr(route, "path", None)
    if not isinstance(prefix, str) or not prefix:
        return ""
    return prefix[:-1] if prefix.endswith("/") and len(prefix) > 1 else prefix


def _as_route_list(sub: Any) -> list[Any]:
    if sub is None:
        return []
    try:
        return list(sub)
    except TypeError:  # pragma: no cover - non-iterable .routes is not expected
        return []


def _sub_routes(route: Any) -> list[Any]:
    """The child routes of a router/mount/wrapper, or ``[]`` for a leaf.

    Covers three shapes:

    * a direct ``.routes`` iterable — Starlette ``Mount`` / ``Router`` and the
      flattened FastAPI ``app.routes`` itself;
    * FastAPI 0.137+ ``_IncludedRouter``, whose children are NOT on ``.routes``
      but on the wrapped router reachable via ``include_context.included_router``
      (equivalently ``original_router``). Its children's ``.path`` values are
      already fully composed (FastAPI bakes the include prefix into each child
      path at include time), so we descend into them with no extra prefix.

    A concrete leaf route exposes none of these and yields ``[]``.
    """
    direct = _as_route_list(getattr(route, "routes", None))
    if direct:
        return direct
    # FastAPI 0.137+ _IncludedRouter: reach the wrapped router's routes.
    ctx = getattr(route, "include_context", None)
    included = getattr(ctx, "included_router", None) or getattr(
        route, "original_router", None
    )
    if included is not None:
        return _as_route_list(getattr(included, "routes", None))
    return []


def _collect(route: Any, prefix: str, out: set[str]) -> None:
    children = _sub_routes(route)
    if children:
        # A container: compose its mount prefix (if any) and descend. Some
        # wrappers (e.g. _IncludedRouter) contribute no prefix; _mount_prefix
        # returns "" for them, so composition is a no-op.
        child_prefix = prefix + _mount_prefix(route)
        for child in children:
            _collect(child, child_prefix, out)
        return
    # A leaf: record its (prefixed) path, if it has one.
    own = _own_path(route)
    if own is not None:
        out.add(prefix + own)


def collect_route_paths(app: Any) -> set[str]:
    """Every LEAF route path registered on ``app`` (templated form, e.g. ``{id}``).

    Recursively walks ``app.routes`` so it returns the SAME flattened set of
    concrete paths regardless of whether the framework flattens included routers
    (FastAPI 0.116–0.136) or wraps them in ``_IncludedRouter``/``Mount`` objects
    (FastAPI 0.137+, Starlette 1.x). ``Mount`` prefixes are composed into the
    child paths.
    """
    out: set[str] = set()
    for route in getattr(app, "routes", []):
        _collect(route, "", out)
    return out
