"""Tests for RoutingClientManager: cross-provider fallback on eligible failures."""

from __future__ import annotations

import pytest

from himmy.core.errors import HimmyError
from himmy.services.inference import (
    InferenceError,
    InferenceErrorCode,
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceService,
    InferenceStatus,
    Route,
    RoutingClientManager,
)
from tests.conftest import run_async


class _Manager:
    """A scripted client manager: SUCCESS, or FAILED with a given error code."""

    def __init__(
        self,
        *,
        provider: str,
        status: InferenceStatus,
        code: InferenceErrorCode | None = None,
    ) -> None:
        self.provider = provider
        self.status = status
        self.code = code
        self.seen_model_keys: list[str] = []
        self.calls = 0

    def resolve(self, model_key: str) -> str:
        return f"{self.provider}:{model_key}"

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        self.calls += 1
        self.seen_model_keys.append(request.model_key)
        if self.status == InferenceStatus.SUCCESS:
            return InferenceResponse(
                request_id=request.request_id,
                status=InferenceStatus.SUCCESS,
                output_text=f"served by {self.provider}",
                provider_name=self.provider,
            )
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.FAILED,
            error=InferenceError(
                code=self.code or InferenceErrorCode.UNKNOWN, message="x"
            ),
            provider_name=self.provider,
        )


class _Raises:
    provider = "boom"

    def resolve(self, model_key: str) -> str:
        return model_key

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        raise RuntimeError("connection reset")


def _req() -> InferenceRequest:
    return InferenceRequest(messages=[InferenceMessage(role="user", content="hi")])


def test_primary_success_short_circuits() -> None:
    """A healthy primary serves; the secondary is never called."""
    primary = _Manager(provider="a", status=InferenceStatus.SUCCESS)
    secondary = _Manager(provider="b", status=InferenceStatus.SUCCESS)
    router = RoutingClientManager.from_managers(primary, secondary)
    resp = run_async(router.generate(_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.provider_name == "a"
    assert secondary.calls == 0
    assert resp.metadata["route_index"] == 0
    assert len(resp.metadata["fallback_chain"]) == 1


def test_fails_over_to_secondary_on_eligible_code() -> None:
    """A QUOTA failure on the primary fails over to a healthy secondary."""
    primary = _Manager(
        provider="a", status=InferenceStatus.FAILED, code=InferenceErrorCode.QUOTA
    )
    secondary = _Manager(provider="b", status=InferenceStatus.SUCCESS)
    router = RoutingClientManager.from_managers(primary, secondary)
    resp = run_async(router.generate(_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.provider_name == "b"
    assert resp.metadata["route_index"] == 1
    chain = resp.metadata["fallback_chain"]
    assert [a["status"] for a in chain] == ["FAILED", "SUCCESS"]
    assert chain[0]["error_code"] == "QUOTA"


def test_non_failover_code_does_not_route() -> None:
    """INVALID_REQUEST is not a provider problem — it is returned, not failed over."""
    primary = _Manager(
        provider="a",
        status=InferenceStatus.FAILED,
        code=InferenceErrorCode.INVALID_REQUEST,
    )
    secondary = _Manager(provider="b", status=InferenceStatus.SUCCESS)
    router = RoutingClientManager.from_managers(primary, secondary)
    resp = run_async(router.generate(_req()))
    assert resp.status == InferenceStatus.FAILED
    assert secondary.calls == 0
    assert resp.metadata["route_index"] == 0


def test_raised_exception_is_treated_as_failover() -> None:
    """A route that raises is normalized to a failover trigger, not propagated."""
    secondary = _Manager(provider="b", status=InferenceStatus.SUCCESS)
    router = RoutingClientManager.from_managers(_Raises(), secondary)
    resp = run_async(router.generate(_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.provider_name == "b"
    assert resp.metadata["fallback_chain"][0]["error_code"] == "PROVIDER_UNAVAILABLE"


def test_all_routes_exhausted_returns_last_failure() -> None:
    """When every route fails over, the last failure is returned with the chain."""
    primary = _Manager(
        provider="a",
        status=InferenceStatus.FAILED,
        code=InferenceErrorCode.PROVIDER_UNAVAILABLE,
    )
    secondary = _Manager(
        provider="b", status=InferenceStatus.FAILED, code=InferenceErrorCode.TIMEOUT
    )
    router = RoutingClientManager.from_managers(primary, secondary)
    resp = run_async(router.generate(_req()))
    assert resp.status == InferenceStatus.FAILED
    assert resp.provider_name == "b"
    assert len(resp.metadata["fallback_chain"]) == 2


def test_route_model_key_override_is_applied() -> None:
    """A route's model_key overrides the request's key for that provider."""
    primary = _Manager(
        provider="a",
        status=InferenceStatus.FAILED,
        code=InferenceErrorCode.RATE_LIMITED,
    )
    secondary = _Manager(provider="b", status=InferenceStatus.SUCCESS)
    router = RoutingClientManager(
        [
            Route(manager=primary, model_key="fast", label="fast"),
            Route(manager=secondary, model_key="strong", label="strong"),
        ]
    )
    run_async(router.generate(_req()))
    assert primary.seen_model_keys == ["fast"]
    assert secondary.seen_model_keys == ["strong"]


def test_empty_routes_rejected() -> None:
    """A router with no routes is a construction error."""
    with pytest.raises(HimmyError):
        RoutingClientManager([])


def test_routes_through_inference_service_end_to_end() -> None:
    """InferenceService.run over a router fails over and returns the SUCCESS."""
    primary = _Manager(
        provider="a", status=InferenceStatus.FAILED, code=InferenceErrorCode.AUTH
    )
    secondary = _Manager(provider="b", status=InferenceStatus.SUCCESS)
    service = InferenceService(RoutingClientManager.from_managers(primary, secondary))
    resp = run_async(service.run(_req()))
    assert resp.status == InferenceStatus.SUCCESS
    assert resp.provider_name == "b"
    assert resp.metadata["route_index"] == 1
