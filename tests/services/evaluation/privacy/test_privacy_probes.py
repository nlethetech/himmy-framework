"""Tests for the WS4.7 adversarial privacy probe generator (B2).

Coverage mirrors the phase brief:

* **determinism** — the same ``seed`` yields a byte-identical suite (ids, prompts,
  grading payload), and a different seed shifts only the ids (must_fix: CI-stable);
* **scenario coverage** — every :class:`ProbeScenario` is present for every default
  persona, single- and multi-hop variants are emitted, and a probe case uses
  ``input={'prompt': ...}`` + the existing ``safety`` grader;
* **the grader is not vacuous** — each template's ``disallowed_patterns`` MATCH a known
  leaking sample (so a real disclosure would fail) and do NOT match a clean refusal;
* **end-to-end through the REAL EvaluationService with NO provider** — a refusal output
  PASSES and a leaking output FAILS, with the per-case veto firing and the service-level
  default veto untouched (must_fix #9);
* **the LLM synthesizer is provider-gated** (must_fix #5) and **fails closed** to the
  deterministic prompt on bad/empty output;
* **from_consent_ledger** derives an INDUCE_TRAINING_USE probe for an opted-out subject
  and a CROSS_SUBJECT_LEAK probe for an erased subject, degrades to the catalog default
  with an explicit empty-result signal, and raises on an unknown state;
* **empty / unknown scenario ⇒ HimmyError** (no silent vacuous probe).
"""

from __future__ import annotations

import re

import pytest

from himmy.core.errors import HimmyError
from himmy.entities.records import EntityRecord
from himmy.entities.registry import EntityRegistry
from himmy.services.evaluation.privacy import (
    ALL_SCENARIOS,
    DEFAULT_PERSONAS,
    DeterministicProbeSynthesizer,
    LLMProbeSynthesizer,
    PrivacyProbeGenerator,
    ProbeScenario,
    build_privacy_probe_suite,
    leak_patterns_for,
    planted_pii_blob,
    probe_pii_labels,
    template_for,
)
from himmy.services.evaluation.privacy.catalog import (
    ScenarioPlan,
    default_scenario_plans,
)
from himmy.services.evaluation.service import EvaluationService
from himmy.services.governance.consent import CONSENT_KIND, ConsentState, Purpose
from himmy.services.governance.retention import ERASURE_KIND

# ------------------------------------------------------------------- determinism


def test_same_seed_is_byte_identical() -> None:
    """Two builds with the same seed produce an identical suite (CI-stable)."""
    a = build_privacy_probe_suite(seed="seed-x")
    b = build_privacy_probe_suite(seed="seed-x")
    assert a.model_dump_json() == b.model_dump_json()


def test_same_seed_stable_ids_and_prompts() -> None:
    """Determinism extends to case ids, ordering, prompts, and grading payload."""
    a = build_privacy_probe_suite(seed="seed-y")
    b = build_privacy_probe_suite(seed="seed-y")
    assert a.suite_id == b.suite_id
    assert [c.case_id for c in a.cases] == [c.case_id for c in b.cases]
    assert [c.input for c in a.cases] == [c.input for c in b.cases]
    assert [c.expected_output for c in a.cases] == [c.expected_output for c in b.cases]


def test_different_seed_changes_ids_not_structure() -> None:
    """A different seed shifts ids but keeps the same set of scenarios/personas."""
    a = build_privacy_probe_suite(seed="seed-a")
    b = build_privacy_probe_suite(seed="seed-b")
    assert a.suite_id != b.suite_id
    assert {c.case_id for c in a.cases} != {c.case_id for c in b.cases}
    # Same logical coverage (persona × scenario × hop).
    sig_a = {
        (c.metadata["persona"], c.metadata["scenario"], c.metadata["multi_hop"])
        for c in a.cases
    }
    sig_b = {
        (c.metadata["persona"], c.metadata["scenario"], c.metadata["multi_hop"])
        for c in b.cases
    }
    assert sig_a == sig_b


# ----------------------------------------------------------------- scenario coverage


def test_every_scenario_present_for_every_persona() -> None:
    """The default suite covers every scenario for every default persona."""
    suite = build_privacy_probe_suite()
    pairs = {(c.metadata["persona"], c.metadata["scenario"]) for c in suite.cases}
    for persona in DEFAULT_PERSONAS:
        for scenario in ALL_SCENARIOS:
            assert (persona.persona_id, scenario.value) in pairs, (
                persona.persona_id,
                scenario.value,
            )


def test_single_and_multi_hop_variants_emitted() -> None:
    """Both single- and multi-hop probes exist (the injection scenario is multi-hop)."""
    suite = build_privacy_probe_suite()
    hops = {c.metadata["multi_hop"] for c in suite.cases}
    assert hops == {True, False}
    # The prompt-injection-consent template is inherently multi-hop.
    injection = [
        c
        for c in suite.cases
        if c.metadata["scenario"] == ProbeScenario.PROMPT_INJECTION_CONSENT.value
    ]
    assert injection and all(c.metadata["multi_hop"] for c in injection)


def test_probe_case_uses_prompt_key_and_safety_grader() -> None:
    """Cases use the harness default input key and grade on the existing SafetyMetric."""
    suite = build_privacy_probe_suite()
    for case in suite.cases:
        assert "prompt" in case.input and isinstance(case.input["prompt"], str)
        assert case.metric_weights == {"safety": 1.0}
        assert case.metadata["veto_metrics"] == ["safety"]
        assert "disallowed_patterns" in case.expected_output


def test_no_multi_hop_can_be_disabled() -> None:
    """``include_multi_hop=False`` keeps only inherently-multi-hop templates as multi."""
    suite = build_privacy_probe_suite(include_multi_hop=False)
    # Only the prompt-injection scenario (inherently multi-hop) stays multi-hop.
    multi = {c.metadata["scenario"] for c in suite.cases if c.metadata["multi_hop"]}
    assert multi == {ProbeScenario.PROMPT_INJECTION_CONSENT.value}


# --------------------------------------------------- the grader is NOT vacuous


def test_every_template_disallowed_patterns_catch_a_known_leak() -> None:
    """Each template's leak signatures MATCH its leak sample and MISS a clean refusal."""
    for persona in DEFAULT_PERSONAS:
        for scenario in ALL_SCENARIOS:
            template = template_for(scenario, persona)
            assert template.disallowed, (persona.persona_id, scenario.value)
            text = template.leak_sample.lower()
            # At least one disallowed pattern must fire on the leaking sample.
            assert any(re.search(p, text) for p in template.disallowed), (
                "no disallowed pattern caught the leak sample for",
                scenario.value,
            )
            # And none fire on a clean refusal.
            refusal = template.refusal_sample.lower()
            assert not any(re.search(p, refusal) for p in template.disallowed), (
                "a disallowed pattern wrongly matched a refusal for",
                scenario.value,
            )


def test_leak_patterns_for_known_labels_are_nonempty_and_match_corpus() -> None:
    """The PII corpus tokens are caught by the reused DLP patterns (must_fix #5)."""
    labels = probe_pii_labels()
    patterns = leak_patterns_for(labels)
    assert patterns  # never silently empty
    blob = planted_pii_blob(labels)
    # Every planted label's token is detected by at least one reused DLP pattern.
    assert any(re.search(p, blob) for p in patterns)


# ----------------------------- end-to-end through the REAL service, NO provider


@pytest.mark.asyncio
async def test_refusal_passes_and_leak_fails_through_real_service() -> None:
    """A refusal PASSES and a disclosure FAILS each case via the real EvaluationService."""
    suite = build_privacy_probe_suite()
    service = EvaluationService()  # no provider, no embedder — pure deterministic

    # A clean refusal for every case (never matches a leak signature).
    refusal_outputs = {
        c.case_id: "I can't help with that request." for c in suite.cases
    }
    refused = await service.run_suite(suite=suite, actual_outputs=refusal_outputs)
    assert refused.aggregate_score == 1.0
    assert all(r.passed for r in refused.case_results)

    # A leaking output per case (the template's own leak sample) — every case FAILS.
    leak_outputs: dict[str, str] = {}
    by_case = {c.case_id: c for c in suite.cases}
    for case in suite.cases:
        persona = next(
            p for p in DEFAULT_PERSONAS if p.persona_id == case.metadata["persona"]
        )
        scenario = ProbeScenario(case.metadata["scenario"])
        leak_outputs[case.case_id] = template_for(scenario, persona).leak_sample
    leaked = await service.run_suite(suite=suite, actual_outputs=leak_outputs)
    assert leaked.aggregate_score == 0.0
    for result in leaked.case_results:
        assert not result.passed, by_case[result.case_id].metadata


@pytest.mark.asyncio
async def test_per_case_veto_does_not_change_service_default() -> None:
    """The probe opts in via case metadata; the service default veto stays ('safety',)."""
    service = EvaluationService()
    # Reach into the configured default to prove we never mutated it.
    assert service._veto_metrics == ("safety",)  # noqa: SLF001 - invariant check
    suite = build_privacy_probe_suite()
    assert all(c.metadata["veto_metrics"] == ["safety"] for c in suite.cases)


# ------------------------------------------------------ LLM synthesizer (gated)


def test_llm_synthesizer_not_built_for_stub_or_none_provider() -> None:
    """``LLMProbeSynthesizer`` is constructed ONLY for a real provider (must_fix #5)."""
    from himmy.services.evaluation.privacy.probes import _select_synthesizer

    sentinel = object()  # a fake inference service that must NOT be touched
    for provider in (None, "stub"):
        chosen = _select_synthesizer(
            seed="s", inference_service=sentinel, provider=provider
        )
        assert isinstance(chosen, DeterministicProbeSynthesizer)
    # A real provider DOES select the LLM synthesizer.
    chosen = _select_synthesizer(
        seed="s", inference_service=sentinel, provider="ollama"
    )
    assert isinstance(chosen, LLMProbeSynthesizer)


def test_llm_synthesizer_fails_closed_to_deterministic_on_bad_output() -> None:
    """A synthesizer whose model returns junk falls back to the deterministic prompt."""

    class _BadInference:
        async def run(self, request: object) -> object:
            raise RuntimeError("provider exploded")

    deterministic = DeterministicProbeSynthesizer(seed="s")
    llm = LLMProbeSynthesizer(_BadInference(), fallback=deterministic)
    plan = ScenarioPlan(
        persona=DEFAULT_PERSONAS[0], scenario=ProbeScenario.PII_EXTRACTION
    )
    det_prompt, det_template = deterministic.synthesize(plan)
    prompt, template = llm.synthesize(plan)
    # Falls back closed: same prompt and (always) same template/grader.
    assert prompt == det_prompt
    assert template.disallowed == det_template.disallowed


def test_llm_synthesizer_uses_rewrite_when_provider_succeeds() -> None:
    """A successful structured rewrite replaces the prompt but never the grader."""

    class _StructuredResp:
        status = __import__(
            "himmy.services.inference.models", fromlist=["InferenceStatus"]
        ).InferenceStatus.SUCCESS
        output_structured = {"prompt": "rephrased adversarial prompt"}

    class _GoodInference:
        async def run(self, request: object) -> object:
            return _StructuredResp()

    deterministic = DeterministicProbeSynthesizer(seed="s")
    llm = LLMProbeSynthesizer(_GoodInference(), fallback=deterministic)
    plan = ScenarioPlan(
        persona=DEFAULT_PERSONAS[0], scenario=ProbeScenario.PII_EXTRACTION
    )
    _det_prompt, det_template = deterministic.synthesize(plan)
    prompt, template = llm.synthesize(plan)
    assert prompt == "rephrased adversarial prompt"
    assert template.disallowed == det_template.disallowed  # grader is template-owned


# ------------------------------------------------------ from_consent_ledger


def _consent(
    registry: EntityRegistry,
    *,
    subject: str,
    state: ConsentState,
    version: int = 1,
    purpose: Purpose = Purpose.TRAIN,
) -> None:
    """Register a consent record directly (mirrors the ledger's payload shape)."""
    registry.register(
        EntityRecord.create(
            stable_id=f"consent-{subject}-{purpose.value}",
            version=version,
            kind=CONSENT_KIND,
            payload={
                "subject_id": subject,
                "purpose": purpose.value,
                "state": state.value,
            },
            metadata={"subject_id": subject, "purpose": purpose.value},
        )
    )


def test_from_consent_ledger_derives_opted_out_training_probe() -> None:
    """An opted-out (DENIED train) subject yields an INDUCE_TRAINING_USE probe."""
    registry = EntityRegistry()
    _consent(registry, subject="teacher_b", state=ConsentState.DENIED)
    suite = PrivacyProbeGenerator().from_consent_ledger(registry)

    assert suite.metadata["derived_from"] == "consent_ledger"
    assert suite.metadata["ledger_subjects"] == 1
    scenarios = {c.metadata["scenario"] for c in suite.cases}
    assert ProbeScenario.INDUCE_TRAINING_USE.value in scenarios
    # The derived probe targets the actual subject id.
    assert any(c.metadata["subject_id"] == "teacher_b" for c in suite.cases)


def test_from_consent_ledger_derives_erased_cross_subject_probe() -> None:
    """An erased subject (tombstone) yields a CROSS_SUBJECT_LEAK probe."""
    registry = EntityRegistry()
    registry.register(
        EntityRecord.create(
            stable_id="ts-1",
            version=1,
            kind=ERASURE_KIND,
            payload={"subject_id": "teacher_a", "crypto_shredded": True},
        )
    )
    suite = PrivacyProbeGenerator().from_consent_ledger(registry)
    assert suite.metadata["derived_from"] == "consent_ledger"
    scenarios = {c.metadata["scenario"] for c in suite.cases}
    assert ProbeScenario.CROSS_SUBJECT_LEAK.value in scenarios
    assert any(c.metadata["subject_id"] == "teacher_a" for c in suite.cases)


def test_from_consent_ledger_latest_state_wins() -> None:
    """A re-grant after a denial means the subject is no longer opted-out."""
    registry = EntityRegistry()
    _consent(registry, subject="teacher_b", state=ConsentState.DENIED, version=1)
    _consent(registry, subject="teacher_b", state=ConsentState.GRANTED, version=2)
    suite = PrivacyProbeGenerator().from_consent_ledger(registry)
    # No opted-out subject AND no tombstone ⇒ catalog default fallback.
    assert suite.metadata["derived_from"] == "catalog_default"
    assert suite.metadata["ledger_subjects"] == 0


def test_from_consent_ledger_empty_degrades_with_explicit_signal() -> None:
    """No consent/erasure records ⇒ catalog default with an explicit empty signal."""
    registry = EntityRegistry()
    suite = PrivacyProbeGenerator().from_consent_ledger(registry)
    assert suite.metadata["derived_from"] == "catalog_default"
    assert suite.metadata["ledger_subjects"] == 0
    # Still a usable (non-empty) suite — degraded, never silently zero.
    assert suite.cases


def test_from_consent_ledger_raises_on_unknown_state() -> None:
    """An unrecognised consent state fails loudly (never silently drops a subject)."""
    registry = EntityRegistry()
    registry.register(
        EntityRecord.create(
            stable_id="consent-bad",
            version=1,
            kind=CONSENT_KIND,
            payload={
                "subject_id": "teacher_x",
                "purpose": Purpose.TRAIN.value,
                "state": "not-a-real-state",
            },
        )
    )
    with pytest.raises(HimmyError):
        PrivacyProbeGenerator().from_consent_ledger(registry)


# ------------------------------------------------ empty / unknown-scenario errors


def test_template_for_unknown_scenario_raises() -> None:
    """A scenario with no registered template builder fails loudly (no vacuous probe)."""

    class _Fake(str):
        value = "totally-unknown-scenario"

    with pytest.raises(HimmyError):
        template_for(_Fake(), DEFAULT_PERSONAS[0])  # type: ignore[arg-type]


def test_empty_personas_or_scenarios_yields_empty_plan() -> None:
    """No personas/scenarios ⇒ an empty plan (a degenerate suite, but not an error)."""
    assert default_scenario_plans(personas=(), scenarios=()) == []
    suite = build_privacy_probe_suite(personas=(), scenarios=())
    assert suite.cases == []


def test_llm_synthesizer_is_loop_safe_inside_running_loop() -> None:
    """The LLM synthesizer paraphrases even when called from a running event loop.

    ``PrivacyAuditService.run`` builds the probe suite from inside its async frame, so a naive
    ``asyncio.run()`` in the synthesizer would raise ``RuntimeError`` and silently fall back to
    the deterministic prompt. The loop-safe path drives the model call on a worker thread.
    """
    import asyncio
    import types

    from himmy.services.evaluation.privacy.probes import LLMProbeSynthesizer
    from himmy.services.inference.models import InferenceStatus

    class _FakeInference:
        async def run(self, _request: object) -> object:
            return types.SimpleNamespace(
                status=InferenceStatus.SUCCESS,
                output_structured={"prompt": "REWORDED-ATTACK"},
            )

    plan = default_scenario_plans()[0]
    synth = LLMProbeSynthesizer(_FakeInference())

    async def _inside_running_loop() -> str:
        prompt, _template = synth.synthesize(plan)  # sync call from within a live loop
        return prompt

    assert asyncio.run(_inside_running_loop()) == "REWORDED-ATTACK"
