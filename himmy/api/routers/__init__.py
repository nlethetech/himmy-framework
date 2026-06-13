"""API kernel: HTTP routers mapping transport shapes to application services."""

from __future__ import annotations

from himmy.api.routers import (
    connectors,
    context,
    dashboard,
    evaluation,
    recommendations,
    runs,
    studio,
    studio_eval,
    studio_files,
    studio_knowledge_upload,
    studio_lineage,
    studio_mcp,
    studio_memory,
    studio_missions,
    studio_models,
    studio_notify,
    studio_privacy,
    studio_projects,
    studio_routines,
    studio_teams,
)

__all__ = [
    "connectors",
    "context",
    "runs",
    "recommendations",
    "dashboard",
    "evaluation",
    "studio",
    "studio_eval",
    "studio_files",
    "studio_knowledge_upload",
    "studio_lineage",
    "studio_mcp",
    "studio_memory",
    "studio_missions",
    "studio_models",
    "studio_notify",
    "studio_privacy",
    "studio_projects",
    "studio_routines",
    "studio_teams",
]
