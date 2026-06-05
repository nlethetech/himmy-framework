"""Tool routing: pick the few relevant tools before the agent sees them (Tier 1.3).

Small models drown in choices — with a dozen tools bound they loop, pick the wrong
one, or refuse. :func:`select_tools` runs one cheap structured-output call that, given
the user's request and a tool catalog, returns the handful of relevant tool names. The
agent loop then binds only those, sharply improving tool-*selection* on 0.5–3B models.

Safe by construction: it never narrows below the model's own pick, returns *all* tools
on any failure, and is a no-op when there are already few tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Sequence

    from himmy.services.inference.service import InferenceService

_ROUTER_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["tools"],
    "additionalProperties": False,
}


async def select_tools(
    inference: InferenceService,
    query: str,
    candidates: Sequence[tuple[str, str]],
    *,
    max_tools: int = 4,
    model_key: str = "default",
) -> list[str]:
    """Return the relevant subset of tool names for ``query`` (or all on failure).

    ``candidates`` is ``[(name, description), ...]``. With ``len <= max_tools`` there is
    nothing to narrow, so all names are returned unchanged.
    """
    names = [name for name, _ in candidates]
    if len(names) <= max_tools:
        return names

    from himmy.services.inference.models import InferenceMessage, InferenceRequest

    catalog = "\n".join(f"- {name}: {desc}" for name, desc in candidates)
    system = (
        "You are a tool router. Given a user request and a catalog of tools, choose "
        f"the {max_tools} or fewer tools most relevant to fulfilling the request. "
        "Use ONLY names that appear in the catalog. If none are relevant, return an "
        "empty list."
    )
    user = f"Tool catalog:\n{catalog}\n\nUser request: {query}"
    request = InferenceRequest(
        model_key=model_key,
        messages=[
            InferenceMessage(role="system", content=system),
            InferenceMessage(role="user", content=user),
        ],
        output_json_schema=_ROUTER_SCHEMA,
    )
    try:
        response = await inference.run(request)
        chosen = (response.output_structured or {}).get("tools") or []
    except Exception:
        return names  # any failure → don't narrow (bind everything)

    valid = set(names)
    selected = [str(name) for name in chosen if str(name) in valid][:max_tools]
    return selected or names  # never strand the agent with zero tools


__all__ = ["select_tools"]
