"""Two clean tools for the outcome-learning A/B.

Both tools RUN SUCCESSFULLY every time (no operational failures) and return a
plausible-looking answer. The ONLY difference is the QUALITY of what they produce,
which the experiment encodes out-of-band as seeded ``OUTCOME_SCORED`` events:

* ``quick_lookup`` — runs cleanly but historically produced BAD answers (low outcome
  grades). Operationally it looks perfect (never errors).
* ``cited_lookup`` — runs cleanly AND historically produced GOOD answers (high outcome
  grades).

Descriptions are deliberately symmetric and neutral so the model has no a-priori reason
to prefer either: the only behavioural lever is the learned reputation the runtime
injects. ``CALLS`` records every invocation (name + args) so the driver can read which
tool the model reached for first.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool

#: Ordered record of every tool invocation this process saw: (tool_name, args).
CALLS: list[tuple[str, dict[str, Any]]] = []


def reset() -> None:
    CALLS.clear()


def register(registry: ToolRegistry) -> None:
    """Register both clean lookup tools onto ``registry`` (the from_spec entrypoint)."""

    async def _quick_lookup(args: dict[str, Any]) -> dict[str, Any]:
        CALLS.append(("quick_lookup", dict(args)))
        # Runs cleanly, returns a (deliberately thin) answer. Never errors.
        return {"answer": "approximately 8,849 meters", "source": "quick_lookup"}

    register_local_tool(
        registry,
        name="quick_lookup",
        handler=_quick_lookup,
        description=(
            "Look up a factual answer to a question. Returns a single concise answer string."
        ),
        args_json_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
        read_only=True,
    )

    async def _cited_lookup(args: dict[str, Any]) -> dict[str, Any]:
        CALLS.append(("cited_lookup", dict(args)))
        return {"answer": "approximately 8,849 meters", "source": "cited_lookup"}

    register_local_tool(
        registry,
        name="cited_lookup",
        handler=_cited_lookup,
        description=(
            "Look up a factual answer to a question. Returns a single concise answer string."
        ),
        args_json_schema={
            "type": "object",
            "properties": {"question": {"type": "string"}},
            "required": ["question"],
        },
        read_only=True,
    )
