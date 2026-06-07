"""Inference kernel: a fallback/routing client manager (the LiteLLM-gap closer).

``InferenceService`` wraps exactly one :class:`ClientManager`, and its retry loop
only re-hits the SAME model — so a hard ``QUOTA``/``AUTH``/``PROVIDER_UNAVAILABLE``
failure (precisely when you want a *different* provider) goes straight to FAILED.
:class:`RoutingClientManager` closes that gap: it holds an ordered list of routes
and, on a failover-eligible failure, transparently re-dispatches to the next one —
stamping ``metadata['fallback_chain']`` with exactly what was tried and which route
served. Because it *is* a :class:`ClientManager`, it composes uniformly over the
stub, pydantic-ai, and gateway managers and slots straight into ``InferenceService``.

Streaming: this manager intentionally does NOT implement ``generate_stream``, so
``InferenceService.run_stream`` transparently falls back to buffering via ``run()``
(which routes + fails over) and then chunking — i.e. streaming keeps working and
keeps the failover guarantee, at the cost of token-by-token delivery through the
router. Wrap a single streaming provider directly when real deltas matter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from himmy.core.errors import HimmyError
from himmy.core.metadata import RouteMetadata
from himmy.services.inference.client_manager import ClientManager
from himmy.services.inference.models import (
    InferenceError,
    InferenceErrorCode,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)

#: Failure codes that justify trying a *different* provider. ``INVALID_REQUEST``
#: (the request itself is bad) and ``UNKNOWN`` (unclassified) are excluded — a
#: second provider is unlikely to help and would just mask a real bug.
DEFAULT_FAILOVER_CODES: frozenset[InferenceErrorCode] = frozenset(
    {
        InferenceErrorCode.PROVIDER_UNAVAILABLE,
        InferenceErrorCode.RATE_LIMITED,
        InferenceErrorCode.TIMEOUT,
        InferenceErrorCode.QUOTA,
        InferenceErrorCode.AUTH,
    }
)


@dataclass(frozen=True)
class Route:
    """One entry in a routing chain: a manager plus an optional model override.

    ``model_key`` overrides the request's model key for this route (so a chain can
    map a single logical request onto different concrete models per provider).
    ``label`` is a human-readable id stamped into the fallback chain metadata.
    """

    manager: ClientManager
    model_key: str | None = None
    label: str | None = None
    #: A relative cost hint (e.g. $/1k tokens) used by :meth:`cost_ordered` to try
    #: cheaper routes first; 0.0 for free/local providers.
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class RoutingClientManager:
    """A :class:`ClientManager` that fails over across an ordered list of routes."""

    def __init__(
        self,
        routes: Iterable[Route],
        *,
        failover_codes: Iterable[InferenceErrorCode] = DEFAULT_FAILOVER_CODES,
        provider_name: str = "routing",
    ) -> None:
        """Wire the ordered routes and the set of failover-eligible error codes."""
        self._routes: list[Route] = list(routes)
        if not self._routes:
            raise HimmyError("RoutingClientManager requires at least one route.")
        self._failover_codes = frozenset(failover_codes)
        self.provider_name = provider_name

    @classmethod
    def from_managers(
        cls,
        *managers: ClientManager,
        failover_codes: Iterable[InferenceErrorCode] = DEFAULT_FAILOVER_CODES,
    ) -> RoutingClientManager:
        """Build a router from bare managers (no per-route model override)."""
        return cls([Route(manager=m) for m in managers], failover_codes=failover_codes)

    @classmethod
    def cost_ordered(
        cls,
        routes: Iterable[Route],
        *,
        failover_codes: Iterable[InferenceErrorCode] = DEFAULT_FAILOVER_CODES,
    ) -> RoutingClientManager:
        """Build a cost-aware router: try cheapest routes first (local/free, then cloud).

        Routes are ordered by ascending :attr:`Route.cost` (stable on ties), so a
        free local model is tried before a paid cloud one and the cloud is reached
        only when the cheaper route fails over. ``$0`` local managers therefore
        carry the load, with cloud as the reliability backstop.
        """
        return cls(sorted(routes, key=lambda r: r.cost), failover_codes=failover_codes)

    def resolve(self, model_key: str) -> str:
        """Resolve via the primary route (the model that would serve by default)."""
        primary = self._routes[0]
        return primary.manager.resolve(primary.model_key or model_key)

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Dispatch through the routes, failing over on eligible failures.

        Returns the first SUCCESS (annotated with the chain), or — if every route
        is exhausted or a non-failover-eligible failure is hit — the last failure,
        also annotated. Never raises for provider errors (a raised exception from a
        route is treated as a ``PROVIDER_UNAVAILABLE`` failover trigger); the one
        sanctioned exception, ``NotImplementedError``, propagates unchanged.
        """
        attempts: list[dict[str, Any]] = []
        last: InferenceResponse | None = None
        last_index = 0

        for index, route in enumerate(self._routes):
            req = (
                request
                if route.model_key is None
                else request.model_copy(update={"model_key": route.model_key})
            )
            try:
                response = await route.manager.generate(req)
            except NotImplementedError:
                raise  # genuinely unsupported path — do not mask via failover
            except Exception as exc:  # noqa: BLE001 - normalize to a failover trigger
                response = self._exception_response(req, exc)

            attempts.append(self._attempt(index, route, response))
            last, last_index = response, index

            if response.status == InferenceStatus.SUCCESS:
                return self._annotate(response, attempts, index, route)

            code = response.error.code if response.error else InferenceErrorCode.UNKNOWN
            exhausted = index == len(self._routes) - 1
            if code not in self._failover_codes or exhausted:
                return self._annotate(response, attempts, index, route)
            # otherwise: fall through to the next route

        # Unreachable (the loop always returns), but keeps types total.
        assert last is not None
        return self._annotate(last, attempts, last_index, self._routes[last_index])

    @staticmethod
    def _exception_response(
        request: InferenceRequest, exc: Exception
    ) -> InferenceResponse:
        """Wrap a raised provider error as a failover-eligible FAILED response."""
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
                message=f"{type(exc).__name__}: {exc}",
                retryable=True,
            ),
        )

    @staticmethod
    def _attempt(
        index: int, route: Route, response: InferenceResponse
    ) -> dict[str, Any]:
        """A compact record of one route attempt for the fallback chain."""
        return {
            "index": index,
            "label": route.label,
            "model_key": route.model_key,
            "status": response.status.value,
            "error_code": response.error.code.value if response.error else None,
        }

    @staticmethod
    def _annotate(
        response: InferenceResponse,
        attempts: list[dict[str, Any]],
        index: int,
        route: Route,
    ) -> InferenceResponse:
        """Stamp the fallback chain + serving-route info onto a response."""
        route_meta: RouteMetadata = {
            "fallback_chain": attempts,
            "route_index": index,
            "route_label": route.label,
        }
        response.metadata = {**response.metadata, **route_meta}
        return response


__all__ = ["RoutingClientManager", "Route", "DEFAULT_FAILOVER_CODES"]
