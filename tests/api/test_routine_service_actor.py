"""A workspace-scoped routine fire stamps a least-privilege SERVICE actor (P0 #2).

The scheduled-routine side-door routes through ``RunAppService.create_run`` already; this
asserts that the actor it stamps now carries the routine SERVICE identity + roles + the
tool-authz enforce flag (so the run's tools are gated under a configured authenticator),
while remaining tenant-scoped to the routine's workspace.
"""

from __future__ import annotations

from himmy.api import routines as routines_mod
from himmy.api.auth.service_principal import ROUTINE_SERVICE_SUBJECT
from himmy.api.routines import Routine, Schedule, _run_headless_agent_id
from himmy.services.storage.models import RunRecord, RunStatus
from tests.conftest import run_async


class _AgentDef:
    agent_id = "ag-1"

    def agent_spec(self):  # noqa: ANN201 - test double
        from himmy.config.agent_spec import AgentSpec

        return AgentSpec(name="r", provider="stub")


class _AgentApp:
    async def get_agent_def(self, agent_id: str, *, workspace_id: str):  # noqa: ANN201
        return _AgentDef()


class _RunApp:
    def __init__(self) -> None:
        self.captured_actor: dict | None = None
        self.captured_workspace: str | None = None

    async def create_run(self, **kwargs):  # noqa: ANN003, ANN201 - test double
        self.captured_actor = kwargs.get("actor")
        self.captured_workspace = kwargs.get("workspace_id")
        return RunRecord(
            workspace_id=kwargs["workspace_id"],
            subject_id=kwargs["subject_id"],
            task_id="t",
            persona_name="p",
            status=RunStatus.SUCCEEDED,
        )

    async def get_run(self, run_id: str, *, workspace_id: str):  # noqa: ANN201
        return RunRecord(
            workspace_id=workspace_id,
            subject_id=workspace_id,
            task_id="t",
            persona_name="p",
            status=RunStatus.SUCCEEDED,
            output_text="done",
        )


class _Container:
    def __init__(self) -> None:
        self.run_app = _RunApp()
        self.agent_app = _AgentApp()


def test_routine_fire_stamps_service_actor() -> None:
    """The routine fire passes a SERVICE actor (subject + roles + enforce flag) to create_run."""
    container = _Container()
    routines_mod.set_routine_container_provider(lambda: container)
    try:
        routine = Routine(
            name="nightly",
            prompt="do it",
            agent_id="ag-1",
            workspace_id="ws-9",
            schedule=Schedule(kind="every", hours=1),
        )
        status, _output, _err = run_async(_run_headless_agent_id(routine))
    finally:
        routines_mod.set_routine_container_provider(None)

    assert status == "ok"
    actor = container.run_app.captured_actor
    assert actor is not None
    assert actor["subject"] == ROUTINE_SERVICE_SUBJECT
    assert actor["tool_authz_enforce"] is True
    assert "operator" in actor["roles"]
    # Still routed to the routine's own workspace (tenant-scoped).
    assert container.run_app.captured_workspace == "ws-9"
    # Routine lineage preserved alongside the service identity.
    assert actor["routine_id"] == routine.id
