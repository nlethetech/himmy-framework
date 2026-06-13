"""Programmatic plan mode for /v1 runs: a plan-first run that PAUSES at PLAN-READY (T2g).

Studio's plan-first surface (``himmy.api.studio_service``) registers a *non-gated*
``update_plan`` tool and renders each call as a checklist frame — the human watches the
plan evolve live but the run never stops. ``/v1`` is programmatic: there is no live viewer
to watch a streaming checklist, so plan mode must instead become a single reachable
PAUSE — the run publishes its plan, STOPS at ``AWAITING_APPROVAL`` with the plan in the run
metadata, and a caller resumes it via the SAME ``POST /v1/runs/{id}/approve`` machinery the
HITL gate uses (T2f). That makes plan mode programmatically reachable without inventing a
second resume path.

The mechanism reuses the existing approval gate end-to-end:

* :func:`register_plan_tool` registers an APPROVAL-GATED ``update_plan`` tool into the
  per-run runtime's registry. When the agent calls it to publish its plan, the tool is
  denied (``POLICY_BLOCKED``) exactly like any approval-gated tool, so the agent loop
  pauses into a durable checkpoint — no new pause path is added.
* :func:`apply_plan_mode_to_task` prepends the plan-first system nudge and ensures
  ``update_plan`` is bound for the turn (mirroring the Studio nudge), so the agent calls it
  first.

The plan steps are read back out of the paused checkpoint's ``update_plan`` call by
:meth:`himmy.application.services.RunAppService._extract_plan_from_checkpoint`, which
normalizes them via :func:`normalize_plan_steps` before stamping them into
``metadata['plan']`` — one extraction path, not two.

On approve, the run service rebuilds the SAME runtime (re-registering this tool via the
``plan_mode`` run-metadata flag) and ``resume_agent_loop`` executes the now-approved
``update_plan`` call (a harmless no-op handler) and the agent continues with the task.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.agents.base_agent.task import Task

#: The synthetic planning tool's name (shared with the Studio plan-first contract so a
#: human reading either surface sees the same tool).
PLAN_TOOL = "update_plan"
#: Bounds on a plan payload — a hostile/confused model cannot bloat the run metadata.
PLAN_MAX_STEPS = 12
PLAN_TITLE_MAX = 200
_PLAN_STATUSES = ("pending", "active", "done", "skipped")

_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "maxItems": PLAN_MAX_STEPS,
            "items": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "maxLength": PLAN_TITLE_MAX,
                        "description": "One short step of the plan.",
                    },
                    "status": {
                        "type": "string",
                        "enum": list(_PLAN_STATUSES),
                        "description": "The step's current status.",
                    },
                },
                "required": ["title", "status"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["steps"],
    "additionalProperties": False,
}

#: One-paragraph system nudge prepended when a /v1 run starts in plan mode. It instructs
#: the agent to publish its plan FIRST via ``update_plan`` — which (being approval-gated)
#: pauses the run at PLAN-READY for human review before any real work happens.
PLAN_NUDGE = (
    "Plan-first mode: before doing anything else, call the `update_plan` tool "
    "with a short ordered list of the steps you intend to take (each step a "
    "title plus a status of pending/active/done/skipped, at most "
    f"{PLAN_MAX_STEPS} steps). Do not take any other action until your plan has "
    "been reviewed. Then continue with the task itself."
)


def _cap(text: str, limit: int) -> str:
    """Trim ``text`` to ``limit`` characters."""
    return text if len(text) <= limit else text[:limit]


def normalize_plan_steps(raw: Any) -> list[dict[str, str]]:
    """Coerce a model-supplied ``steps`` payload into the bounded plan shape.

    Keeps at most :data:`PLAN_MAX_STEPS` entries; each must be a mapping with a
    non-empty ``title`` (capped at :data:`PLAN_TITLE_MAX` chars). An unknown or
    missing ``status`` degrades to ``pending`` rather than rejecting the call —
    small local models get the shape almost-right more often than exactly-right.
    """
    if not isinstance(raw, list):
        return []
    out: list[dict[str, str]] = []
    for item in raw[:PLAN_MAX_STEPS]:
        if not isinstance(item, dict):
            continue
        title = _cap(str(item.get("title", "")).strip(), PLAN_TITLE_MAX)
        if not title:
            continue
        status = str(item.get("status", "pending")).strip().lower()
        if status not in _PLAN_STATUSES:
            status = "pending"
        out.append({"title": title, "status": status})
    return out


def register_plan_tool(registry: Any) -> None:
    """Register the APPROVAL-GATED ``update_plan`` tool into a run's tool registry (T2g).

    Unlike the Studio plan tool (read-only, never gated), this one is
    ``requires_approval=True`` so the very first call to publish the plan is denied
    (``POLICY_BLOCKED``), which pauses the agent loop into a checkpoint — the same
    mechanism the HITL approval gate uses. The handler is a harmless validator: it only
    runs after a human approves the plan, and just echoes the bounded steps back.

    Idempotent: re-registering over an existing definition (e.g. a spec that itself
    declared a tool named ``update_plan``) replaces it with the gated version.
    """
    from himmy.services.tools.registry import register_local_tool

    def _handler(args: dict[str, Any]) -> dict[str, Any]:
        steps = normalize_plan_steps(args.get("steps"))
        return {"ok": bool(steps), "steps": steps}

    register_local_tool(
        registry,
        name=PLAN_TOOL,
        handler=_handler,
        description=(
            "Publish your step-by-step plan for this task. Call it first with the "
            "full plan; the run pauses for review before continuing."
        ),
        args_json_schema=_PLAN_SCHEMA,
        requires_approval=True,
        read_only=True,
        metadata={"synthetic": "planning"},
    )


def apply_plan_mode_to_task(task: Task) -> Task:
    """Prepend the plan-first nudge and bind ``update_plan`` for this task (T2g).

    Mirrors the Studio plan-first wiring: the system prefix renders into the system
    prompt on the first turn, and ``update_plan`` is added to any pinned ``tool_names``
    so a spec that pinned an explicit tool list still exposes the planning tool. Mutates
    and returns the same task for convenience.
    """
    prior = str(task.context.get("system_prefix") or "")
    task.context["system_prefix"] = (
        f"{PLAN_NUDGE}\n\n{prior}".strip() if prior else PLAN_NUDGE
    )
    pinned = task.context.get("tool_names")
    if isinstance(pinned, list) and PLAN_TOOL not in pinned:
        task.context["tool_names"] = [*pinned, PLAN_TOOL]
    return task


__all__ = [
    "PLAN_MAX_STEPS",
    "PLAN_NUDGE",
    "PLAN_TITLE_MAX",
    "PLAN_TOOL",
    "apply_plan_mode_to_task",
    "normalize_plan_steps",
    "register_plan_tool",
]
