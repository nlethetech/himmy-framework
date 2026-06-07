"""Tests for the centralized domain -> EntityRecord projection (entities.projection).

These cover the single ``project()`` algorithm directly plus the three behaviours
that the per-model ``to_record()`` wrappers must preserve after centralization:
a persona falling back to its name, a recommendation's curated payload, and a
chat thread taking its version from the model when none is passed. The remaining
nine models' wrappers are exercised implicitly across the suite wherever
``registry.register(x.to_record())`` runs.
"""

from __future__ import annotations

from pydantic import BaseModel

from himmy.agents.base_agent.thread import ChatThread
from himmy.agents.personas.persona import Persona
from himmy.entities import project, stable_id_for
from himmy.entities.records import record_id_for
from himmy.services.storage.models import RecommendationItem


class _Sample(BaseModel):
    name: str = "x"
    value: int = 1


def test_project_is_deterministic_and_content_addressed() -> None:
    """Same inputs -> identical record_id derived from (kind, stable_id, version)."""
    sample = _Sample(name="alpha", value=7)
    a = project(sample, stable_value="k1", namespace="sample", kind="sample")
    b = project(sample, stable_value="k1", namespace="sample", kind="sample")

    assert a.record_id == b.record_id
    assert a.kind == "sample"
    assert a.version == 1
    assert a.stable_id == stable_id_for("k1", namespace="sample")
    assert a.record_id == record_id_for(stable_id=a.stable_id, version=1, kind="sample")


def test_project_defaults_payload_to_model_dump_and_metadata_to_empty() -> None:
    """Without explicit payload/metadata, payload is the JSON dump and metadata is {}."""
    sample = _Sample(name="beta", value=3)
    record = project(sample, stable_value="k2", namespace="sample", kind="sample")

    assert record.payload == {"name": "beta", "value": 3}
    assert record.metadata == {}


def test_project_honours_explicit_payload_and_metadata() -> None:
    """An explicit payload/metadata overrides the model_dump default (curated subset)."""
    sample = _Sample(name="gamma", value=9)
    record = project(
        sample,
        stable_value="k3",
        namespace="sample",
        kind="sample",
        payload={"only": "this"},
        metadata={"tag": "t"},
    )

    assert record.payload == {"only": "this"}
    assert record.metadata == {"tag": "t"}


def test_project_fallback_key_used_when_stable_value_blank() -> None:
    """A blank stable_value falls back to ``fallback_key`` for id derivation."""
    record = project(
        _Sample(),
        stable_value="",
        namespace="persona",
        kind="persona",
        fallback_key="Helper",
    )
    assert record.stable_id == stable_id_for(
        "", namespace="persona", fallback_key="Helper"
    )


def test_persona_to_record_falls_back_to_name() -> None:
    """Persona with a blank agent_id derives its id from the name (special case)."""
    persona = Persona(agent_id="", name="Helper")
    record = persona.to_record()

    assert record.kind == "persona"
    assert record.stable_id == stable_id_for(
        "", namespace="persona", fallback_key="Helper"
    )


def test_recommendation_to_record_excludes_mutable_fields() -> None:
    """RecommendationItem payload omits status/notes so record_id is status-stable."""
    item = RecommendationItem(
        run_id="r1",
        workspace_id="w1",
        subject_id="s1",
        kind="action",
        title="Do the thing",
        notes="a mutable note",
    )
    record = item.to_record()

    assert record.kind == "recommendation"
    assert record.version == 1
    assert "status" not in record.payload
    assert "notes" not in record.payload
    assert record.payload["title"] == "Do the thing"
    assert record.metadata == {"run_id": "r1", "workspace_id": "w1"}


def test_chat_thread_to_record_takes_version_from_model_by_default() -> None:
    """ChatThread uses its own version when none is passed, but an arg overrides it."""
    thread = ChatThread(version=5)

    assert thread.to_record().version == 5
    assert thread.to_record(version=2).version == 2
