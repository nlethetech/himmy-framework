"""Notes pack: ``list_notes`` + ``read_note`` + ``write_note``.

Lets an agent read and write the same durable notes the Studio Notes screen shows
(``.himmy/notes.db`` / ``HIMMY_NOTES_PATH``). A note written by an agent appears in the
GUI, and a note the user wrote is readable by the agent — a shared scratch space.
"""

from __future__ import annotations

from typing import Any

from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.toolkit.config import ToolkitConfig

_WRITE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Note title (a stable key)."},
        "body": {"type": "string", "description": "Markdown body to store."},
    },
    "required": ["title", "body"],
    "additionalProperties": False,
}

_READ_SCHEMA = {
    "type": "object",
    "properties": {"title": {"type": "string", "description": "Note title to read."}},
    "required": ["title"],
    "additionalProperties": False,
}


def register_notes_pack(registry: ToolRegistry, config: ToolkitConfig) -> None:
    """Register ``list_notes`` / ``read_note`` / ``write_note`` over the notes store.

    P1 tenancy: like the tasks pack, the ``notes`` tools reach the SAME singleton
    ``.himmy/notes.db`` (and its Postgres mirror) the scoped REST routes use, so a
    tenant-bound run would otherwise read + overwrite EVERY tenant's notes through the tool
    path. The run's owning workspace is threaded in via ``config.tenant_scope`` and passed as
    the store's ``workspace_id`` so the agent's reads are filtered and its writes stamped
    exactly like the REST caller. ``None`` (offline / one-shot CLI) is "no scoping", so the
    zero-config single-box path is byte-for-byte unchanged.
    """
    from himmy.api.studio_notes import Note, get_notes_store

    scope = config.tenant_scope

    def list_notes(args: dict[str, Any]) -> dict[str, Any]:
        return {
            "notes": [
                {"title": n.title, "updated_at": n.updated_at}
                for n in get_notes_store().list(workspace_id=scope)
            ]
        }

    def read_note(args: dict[str, Any]) -> dict[str, Any]:
        note = get_notes_store().find_by_title(str(args["title"]), workspace_id=scope)
        if note is None:
            return {"found": False}
        return {"found": True, "title": note.title, "body": note.body}

    def write_note(args: dict[str, Any]) -> dict[str, Any]:
        store = get_notes_store()
        title = str(args["title"])
        existing = store.find_by_title(title, workspace_id=scope)
        note = existing or Note(title=title)
        note.body = str(args["body"])
        store.upsert(note, workspace_id=scope)
        return {"saved": True, "title": title}

    register_local_tool(
        registry,
        name="list_notes",
        read_only=True,
        handler=list_notes,
        description="List the titles of saved notes.",
        args_json_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        metadata={"pack": "notes"},
    )
    register_local_tool(
        registry,
        name="read_note",
        read_only=True,
        handler=read_note,
        description="Read a saved note's markdown body by title.",
        args_json_schema=_READ_SCHEMA,
        metadata={"pack": "notes"},
    )
    register_local_tool(
        registry,
        name="write_note",
        read_only=False,
        handler=write_note,
        description="Create or overwrite a note (by title) with markdown.",
        args_json_schema=_WRITE_SCHEMA,
        metadata={"pack": "notes"},
    )


__all__ = ["register_notes_pack"]
