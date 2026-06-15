"""Version-robust route-path collection (``collect_route_paths``).

The collector must return the SAME flattened set of concrete leaf paths whether
the framework flattens included routers into top-level ``APIRoute`` objects
(FastAPI 0.116–0.136) or wraps each included router in an ``_IncludedRouter``
object with no ``.path`` (FastAPI 0.137+) / a Starlette 1.x route tree.

Two layers of coverage:

* ``test_collector_finds_real_mounted_subroutes`` builds the REAL app and
  asserts deep, mounted subroutes are present (a ``/v1/runs/{run_id}/...`` path
  AND an ``/api/studio/...`` path). These routes are reachable on the new
  versions ONLY by descending into the ``_IncludedRouter`` wrappers; a
  top-level-only walk would silently drop them there.

* ``test_collector_descends_into_included_router_wrappers`` builds a SYNTHETIC
  app whose routes are deliberately wrapped (mirroring the 0.137 ``_IncludedRouter``
  shape: a container object with ``.routes`` but NO ``.path``). It proves the
  collector descends — and that a naive top-level ``{r.path for r in routes}``
  walk would FAIL — independent of the installed FastAPI version, so the
  silent-drop failure mode is guarded even when CI runs the capped (flattened)
  FastAPI.
"""

from __future__ import annotations

from himmy.api.app import create_app
from himmy.api.route_introspection import collect_route_paths


def test_collector_finds_real_mounted_subroutes() -> None:
    app = create_app()
    paths = collect_route_paths(app)

    # A /v1 subroute with a path param — registered via include_router, so on
    # FastAPI 0.137+ it lives INSIDE an _IncludedRouter wrapper and is reachable
    # only by descending. (On 0.116–0.136 it is flattened; the collector returns
    # the identical set either way.)
    assert "/v1/runs/{run_id}/events" in paths
    # A Studio BFF subroute, likewise nested under an included router.
    assert "/api/studio/runs/{run_id}" in paths


def test_collector_descends_into_included_router_wrappers() -> None:
    """A SYNTHETIC nested tree mirroring 0.137 ``_IncludedRouter`` wrappers.

    This is the regression guard: it must hold on EVERY FastAPI version (even the
    capped, flattened one CI runs today) so the silent-drop failure mode cannot
    return unnoticed. The leaf paths sit two levels deep under container objects
    that expose ``.routes`` but no ``.path`` of their own — exactly the shape a
    naive ``{r.path for r in app.routes}`` walk drops.
    """

    class _Leaf:
        def __init__(self, path: str) -> None:
            self.path = path

    class _IncludedRouterLike:
        """A wrapper with sub-routes but NO ``.path`` (the 0.137 shape)."""

        def __init__(self, routes: list[object]) -> None:
            self.routes = routes

    class _Mount:
        """A Starlette ``Mount``: a prefix on ``.path`` plus nested ``.routes``."""

        def __init__(self, path: str, routes: list[object]) -> None:
            self.path = path
            self.routes = routes

    class _App:
        def __init__(self, routes: list[object]) -> None:
            self.routes = routes

    app = _App(
        [
            _Leaf("/health"),  # top-level leaf
            _IncludedRouterLike(
                [_Leaf("/v1/runs"), _Leaf("/v1/runs/{run_id}/events")]
            ),
            _Mount("/api", [_IncludedRouterLike([_Leaf("/studio/runs/{run_id}")])]),
        ]
    )

    collected = collect_route_paths(app)
    assert collected == {
        "/health",
        "/v1/runs",
        "/v1/runs/{run_id}/events",
        "/api/studio/runs/{run_id}",
    }

    # Prove the failure mode: a top-level-only walk drops every nested leaf, so
    # the collector is doing real work the naive approach cannot.
    naive_top_level = {getattr(r, "path", "") for r in app.routes}
    assert "/v1/runs/{run_id}/events" not in naive_top_level
    assert "/api/studio/runs/{run_id}" not in naive_top_level
