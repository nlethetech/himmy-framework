"""API kernel: Studio Projects — sustained work gets a home.

Mounted under ``/api/studio/projects`` with the shared ``studio:use`` guard
(see :mod:`himmy.api.routers.studio_common`). A project groups chats and pins
an optional default knowledge base + default agent; the rows live in the chats
store (:mod:`himmy.api.studio_chats`), same SQLite file as the sessions they
group.

Offline-first notes:

* ``kb_id``/``agent_path`` are stored as opaque references and resolved to
  display names best-effort. The Studio KB registry is session-scoped, so a
  stored ``kb_id`` may stop resolving after a restart — that degrades to a
  ``null`` name, never an error.
* Deleting a project never deletes chats; they are merely ungrouped.
"""

from __future__ import annotations

from fastapi import HTTPException
from pydantic import BaseModel, Field

from himmy.api.routers.studio_common import build_studio_router
from himmy.api.studio_chats import ChatSession, ChatsStore, Project, get_chats_store

router = build_studio_router("projects", tag="studio-projects")


# ---- request / response models ----------------------------------------------


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    description: str = Field("", max_length=2000)
    kb_id: str | None = Field(None, max_length=100)
    agent_path: str | None = Field(None, max_length=500)


class ProjectUpdateRequest(BaseModel):
    """Partial update — only fields the caller actually sent are applied.

    Sending an explicit ``null`` clears ``kb_id``/``agent_path`` (detach);
    ``name`` can be changed but never cleared.
    """

    name: str | None = Field(None, min_length=1, max_length=120)
    description: str | None = Field(None, max_length=2000)
    kb_id: str | None = Field(None, max_length=100)
    agent_path: str | None = Field(None, max_length=500)


class ChatAssignRequest(BaseModel):
    chat_id: str = Field(..., min_length=1, max_length=100)


class ProjectDetail(Project):
    """A project plus its chats (newest first) and resolved display names."""

    chats: list[ChatSession] = []
    kb_name: str | None = None
    agent_name: str | None = None


# ---- best-effort name resolution ---------------------------------------------


def _resolve_kb_name(kb_id: str | None) -> str | None:
    """The KB's display name, or None when unset/unknown (e.g. post-restart)."""
    if not kb_id:
        return None
    try:
        from himmy.api.studio_knowledge import list_kbs

        for kb in list_kbs():
            if kb.kb_id == kb_id:
                return kb.name
    except Exception:  # noqa: BLE001 - resolution is cosmetic, never fatal
        return None
    return None


def _resolve_agent_name(agent_path: str | None) -> str | None:
    """The agent's display name from discovery, or None when unset/unknown."""
    if not agent_path:
        return None
    try:
        from himmy.api import studio_service

        for a in studio_service.list_agents():
            if a.path == agent_path:
                return a.name
    except Exception:  # noqa: BLE001 - resolution is cosmetic, never fatal
        return None
    return None


def _detail(store: ChatsStore, project: Project) -> ProjectDetail:
    chats = store.project_chats(project.id)
    return ProjectDetail(
        **project.model_dump(exclude={"chat_count"}),
        chat_count=len(chats),
        chats=chats,
        kb_name=_resolve_kb_name(project.kb_id),
        agent_name=_resolve_agent_name(project.agent_path),
    )


def _notify(kind_title: str, project: Project, body: str = "") -> None:
    from himmy.api.routers.studio_notify import record_notification

    record_notification(
        "project", kind_title, body=body, link=f"/projects/{project.id}"
    )


# ---- routes -------------------------------------------------------------------


@router.get("", response_model=list[Project])
async def projects_list() -> list[Project]:
    """Every project, most recently touched first."""
    return get_chats_store().list_projects()


@router.post("", response_model=Project)
async def projects_create(body: ProjectCreateRequest) -> Project:
    store = get_chats_store()
    project = store.create_project(
        name=body.name.strip(),
        description=body.description.strip(),
        kb_id=body.kb_id,
        agent_path=body.agent_path,
    )
    _notify(f"Project “{project.name}” created", project)
    return project


@router.get("/{project_id}", response_model=ProjectDetail)
async def projects_get(project_id: str) -> ProjectDetail:
    store = get_chats_store()
    project = store.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="unknown project")
    return _detail(store, project)


@router.patch("/{project_id}", response_model=Project)
async def projects_update(project_id: str, body: ProjectUpdateRequest) -> Project:
    store = get_chats_store()
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="unknown project")
    changes: dict[str, str | None] = {}
    for field in body.model_fields_set:
        value = getattr(body, field)
        changes[field] = value.strip() if isinstance(value, str) else value
    updated = store.update_project(project_id, changes)
    if updated is None:  # deleted between the check and the write
        raise HTTPException(status_code=404, detail="unknown project")
    return updated


@router.delete("/{project_id}")
async def projects_delete(project_id: str) -> dict[str, bool]:
    store = get_chats_store()
    project = store.get_project(project_id)
    if project is None:
        return {"ok": False}
    ok = store.delete_project(project_id)
    if ok:
        _notify(
            f"Project “{project.name}” deleted",
            project,
            body="Its chats were kept, just ungrouped.",
        )
    return {"ok": ok}


@router.post("/{project_id}/assign")
async def projects_assign(project_id: str, body: ChatAssignRequest) -> dict[str, bool]:
    store = get_chats_store()
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="unknown project")
    if not store.assign_chat(project_id, body.chat_id):
        raise HTTPException(status_code=404, detail="unknown chat session")
    return {"ok": True}


@router.post("/{project_id}/unassign")
async def projects_unassign(
    project_id: str, body: ChatAssignRequest
) -> dict[str, bool]:
    store = get_chats_store()
    if store.get_project(project_id) is None:
        raise HTTPException(status_code=404, detail="unknown project")
    return {"ok": store.unassign_chat(body.chat_id)}


__all__ = ["router"]
