"""Tests for MultiProviderClientManager + mixed-provider teams."""

from __future__ import annotations

from himmy.config.team_spec import (
    TeamMemberSpec,
    TeamSpec,
    build_team,
    build_team_inference,
)
from himmy.services.inference.models import (
    InferenceMessage,
    InferenceRequest,
    InferenceResponse,
    InferenceStatus,
)
from himmy.services.inference.multi_provider import MultiProviderClientManager
from tests.conftest import run_async


class _Tagged:
    """A fake manager that echoes its tag (to prove dispatch)."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def resolve(self, model_key: str) -> str:
        return self.tag

    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        return InferenceResponse(
            request_id=request.request_id,
            status=InferenceStatus.SUCCESS,
            output_text=self.tag,
            model_path=self.tag,
        )


def _req(key: str) -> InferenceRequest:
    return InferenceRequest(
        model_key=key, messages=[InferenceMessage(role="user", content="hi")]
    )


def test_dispatch_by_model_key() -> None:
    mgr = MultiProviderClientManager(
        {"brain": _Tagged("opus"), "local": _Tagged("ollama")}, default_key="local"
    )
    assert run_async(mgr.generate(_req("brain"))).output_text == "opus"
    assert run_async(mgr.generate(_req("local"))).output_text == "ollama"
    # unknown key → default backend
    assert run_async(mgr.generate(_req("nope"))).output_text == "ollama"


def test_resets_model_key_for_sub_manager() -> None:
    """The sub-manager is called with a neutral model_key (uses its own model)."""
    seen: dict = {}

    class _Spy:
        def resolve(self, k: str) -> str:
            return "x"

        async def generate(self, request: InferenceRequest) -> InferenceResponse:
            seen["model_key"] = request.model_key
            return InferenceResponse(
                request_id=request.request_id, status=InferenceStatus.SUCCESS
            )

    mgr = MultiProviderClientManager({"brain": _Spy()})
    run_async(mgr.generate(_req("brain")))
    assert seen["model_key"] == "default"


def test_build_team_inference_single_when_no_providers() -> None:
    """No per-member providers → a single InferenceService (not the multiplexer)."""
    spec = TeamSpec(members=[TeamMemberSpec(name="a")], entry="a")
    inference = build_team_inference(spec, default_provider="stub")
    assert not isinstance(inference._client_manager, MultiProviderClientManager)


def test_build_team_inference_multiplexes_with_providers() -> None:
    """Per-member providers → a MultiProviderClientManager keyed by provider:model."""
    spec = TeamSpec(
        members=[
            TeamMemberSpec(name="brain", provider="stub", model="default"),
            TeamMemberSpec(name="worker", provider="stub", model="default"),
        ],
        entry="brain",
    )
    inference = build_team_inference(spec, default_provider="stub")
    assert isinstance(inference._client_manager, MultiProviderClientManager)
    # build_team assigns each member the matching dispatch key.
    team, _registry = build_team(spec)
    assert team.require("brain").model_key == "stub:default"
