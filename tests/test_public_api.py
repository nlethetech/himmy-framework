"""Public-symbol stability contract for ``import himmy``.

``himmy.__all__`` is the supported, SemVer-governed top-level API (see
``docs/STABILITY.md``). This test freezes that set: adding a name is allowed (extend
``_FROZEN_PUBLIC_API`` in the same PR, a minor bump), but DROPPING or RENAMING a name
reds the build — a consumer pinning ``himmy`` would break on upgrade otherwise.

It also resolves EVERY exported name, including the lazy PEP-562 (``__getattr__``)
paths, so a re-export that silently stops importing (a moved/renamed internal) is
caught here rather than in a downstream ``from himmy import X``.
"""

from __future__ import annotations

import himmy

# The frozen public surface. Keep in lockstep with himmy.__all__: a new export is a
# minor bump (add it here in the same PR); a removal/rename is a breaking change that
# must wait for a major bump per docs/STABILITY.md.
_FROZEN_PUBLIC_API = frozenset(
    {
        "__version__",
        "Persona",
        "Task",
        "Agent",
        "AgentSpec",
        "load_agent_spec",
        "ToolkitConfig",
        "register_packs",
        "build_embedder",
        "AgentTeam",
        "TeamMember",
        "MultiAgentOrchestrator",
        "GroupChatOrchestrator",
        "RoundRobinSelector",
        "LLMSelector",
        "CallableSelector",
        "SubtaskSpec",
        "SubtaskResult",
        "fan_out",
        "PlannerOrchestrator",
        "MemoryService",
        "AgentEvalHarness",
        "build_guardrail_pipeline",
        "build_runtime",
        "build_inference",
        "build_storage",
        "Skill",
        "SkillRegistry",
        "resolve_skills",
        "TypedAgent",
        "TypedAgentRunResult",
        "RunContext",
    }
)


def test_public_all_matches_frozen_contract() -> None:
    current = set(himmy.__all__)
    removed = _FROZEN_PUBLIC_API - current
    added = current - _FROZEN_PUBLIC_API
    assert not removed, (
        f"himmy.__all__ DROPPED public symbol(s) {sorted(removed)} — that is a "
        "breaking change. Restore them, or (for an intentional major-version "
        "removal) update _FROZEN_PUBLIC_API and docs/STABILITY.md."
    )
    assert not added, (
        f"himmy.__all__ ADDED public symbol(s) {sorted(added)} without updating the "
        "frozen contract. Add them to _FROZEN_PUBLIC_API (a minor bump) in this PR."
    )


def test_every_public_symbol_resolves() -> None:
    """Every name in __all__ (incl. lazy PEP-562 re-exports) must import cleanly."""
    unresolved: dict[str, str] = {}
    for name in himmy.__all__:
        try:
            getattr(himmy, name)
        except Exception as exc:  # noqa: BLE001 - we want to report ALL failures
            unresolved[name] = f"{type(exc).__name__}: {exc}"
    assert not unresolved, (
        "these himmy.__all__ exports do not resolve (a moved/renamed internal "
        f"behind a lazy re-export?): {unresolved}"
    )


def test_all_has_no_duplicates() -> None:
    assert len(himmy.__all__) == len(set(himmy.__all__)), (
        "himmy.__all__ has duplicate entries"
    )
