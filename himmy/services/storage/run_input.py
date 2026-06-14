"""Recoverable run-input serialization for the leased work-queue (Q0).

The leased queue's whole value is that a crashed ``QUEUED`` run is *re-runnable* from a
fresh process — not merely re-recorded. But the runtime's launch input (the ``Persona``,
the ``Task`` with its prompt, the optional ``LLMConfig`` and per-run ``AgentSpec``) lives
only in the in-process ``asyncio.create_task`` closure; the persisted :class:`RunRecord`
carries none of it. After a crash that closure is gone, so a dispatcher in a fresh process
has nothing to execute with. This module is the Q0 resolution: it serializes that launch
input into ONE opaque blob (``RunInput``) that rides the run record, so the recovery path
can rehydrate the exact same persona/task/config and re-dispatch.

DECISION — Option B (full input persistence), per the Q0 plan item. We persist the
serialized input for EVERY run that opts into recoverability, not only runs launched by a
stored ``agent_id``. The blob is sensitive (it contains the prompt), so it is routed
through the existing :class:`~himmy.services.storage.at_rest.StorePayloadCipher` at-rest
seam (``encrypt_run_input`` / ``decrypt_run_input``, bound to the ``run_id`` as AAD) exactly
like chat-thread content and run-event payloads — encrypted when ``HIMMY_ENCRYPTION_KEY`` /
``HIMMY_KEK_PROVIDER`` is configured, plaintext (offline path unchanged) otherwise.

SCOPE CAVEAT (state plainly, do not bury): the leased queue — and therefore this recoverable
input — only engages when the DURABLE run store is active (``HIMMY_DATABASE_URL`` /
``HIMMY_DURABLE_STORAGE``). The zero-config in-memory default does NOT persist runs across a
process exit at all, so it gets neither the queue nor crash-recovery; that is by design and
unchanged here. This module only builds/parses the blob; whether it is attached to a run is
the caller's (Q3 dispatcher's) decision.

The blob is a small, versioned JSON document of the four models' ``model_dump(mode="json")``
projections plus the agentic flags. Each model round-trips losslessly through JSON (verified),
and parsing rehydrates them with ``model_validate``. A blob whose ``v`` is newer than this
code understands, or that fails to rehydrate, raises :class:`RunInputError` so the recovery
path can mark the run FAILED with a clear reason rather than silently mis-executing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from himmy.core.errors import HimmyError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona
    from himmy.config.agent_spec import AgentSpec
    from himmy.services.inference.models import LLMConfig
    from himmy.services.storage.at_rest import StorePayloadCipher

#: The serialization version of the run-input blob. Bump only on a breaking shape change; the
#: parser refuses a blob from a strictly-newer version (forward-incompatible) so a downgrade
#: never silently mis-reads input.
RUN_INPUT_BLOB_VERSION = 1


class RunInputError(HimmyError):
    """A run-input blob could not be built or parsed (unknown version / malformed)."""


class RunInput:
    """The recoverable launch input of a run: persona, task, optional config + spec, flags.

    Holds rehydrated model instances (not dicts) so the recovery path can hand them straight
    to ``_execute_on_runtime`` exactly as the original ``create_run`` closure did.
    """

    __slots__ = ("persona", "task", "llm_config", "agent_spec", "hitl", "plan")

    def __init__(
        self,
        *,
        persona: Persona,
        task: Task,
        llm_config: LLMConfig | None = None,
        agent_spec: AgentSpec | None = None,
        hitl: bool = False,
        plan: bool = False,
    ) -> None:
        self.persona = persona
        self.task = task
        self.llm_config = llm_config
        self.agent_spec = agent_spec
        self.hitl = hitl
        self.plan = plan


def encode_run_input(
    *,
    persona: Persona,
    task: Task,
    llm_config: LLMConfig | None = None,
    agent_spec: AgentSpec | None = None,
    hitl: bool = False,
    plan: bool = False,
    run_id: str,
    cipher: StorePayloadCipher | None = None,
) -> str:
    """Serialize a run's launch input into the (optionally encrypted) ``input_blob`` string.

    The output is a JSON document; when ``cipher`` is provided the document is enveloped via
    :meth:`StorePayloadCipher.encrypt_run_input` (bound to ``run_id``). The ``agent_spec`` is
    expected to be ALREADY SANITIZED by the caller (the create boundary sanitizes before the
    record is built) — this layer is content-agnostic and only serializes.
    """
    doc: dict[str, Any] = {
        "v": RUN_INPUT_BLOB_VERSION,
        "persona": persona.model_dump(mode="json"),
        "task": task.model_dump(mode="json"),
        "llm_config": llm_config.model_dump(mode="json")
        if llm_config is not None
        else None,
        "agent_spec": agent_spec.model_dump(mode="json")
        if agent_spec is not None
        else None,
        "hitl": bool(hitl),
        "plan": bool(plan),
    }
    blob = json.dumps(doc, sort_keys=True)
    if cipher is not None:
        blob = cipher.encrypt_run_input(blob, run_id=run_id)
    return blob


def decode_run_input(
    blob: str,
    *,
    run_id: str,
    cipher: StorePayloadCipher | None = None,
) -> RunInput:
    """Parse + rehydrate a stored ``input_blob`` back into a :class:`RunInput`.

    Decrypts first when ``cipher`` is provided (idempotent on plaintext, so a blob written
    with encryption off still reads). Raises :class:`RunInputError` on an unknown/newer blob
    version or any rehydration failure, so the recovery path fails loud rather than executing
    garbled input.
    """
    from himmy.agents.base_agent.task import Task
    from himmy.agents.personas.persona import Persona
    from himmy.config.agent_spec import AgentSpec
    from himmy.services.inference.models import LLMConfig

    raw = blob
    if cipher is not None:
        raw = cipher.decrypt_run_input(blob, run_id=run_id)
    try:
        doc = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise RunInputError(f"run-input blob is not valid JSON for run {run_id}") from exc
    if not isinstance(doc, dict):
        raise RunInputError(f"run-input blob is not an object for run {run_id}")
    version = doc.get("v")
    if version is None or not isinstance(version, int) or version > RUN_INPUT_BLOB_VERSION:
        raise RunInputError(
            f"run-input blob version {version!r} is unsupported "
            f"(this build understands <= {RUN_INPUT_BLOB_VERSION})"
        )
    try:
        persona = Persona.model_validate(doc["persona"])
        task = Task.model_validate(doc["task"])
        llm_config = (
            LLMConfig.model_validate(doc["llm_config"])
            if doc.get("llm_config") is not None
            else None
        )
        agent_spec = (
            AgentSpec.model_validate(doc["agent_spec"])
            if doc.get("agent_spec") is not None
            else None
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise RunInputError(
            f"run-input blob failed to rehydrate for run {run_id}: {exc}"
        ) from exc
    return RunInput(
        persona=persona,
        task=task,
        llm_config=llm_config,
        agent_spec=agent_spec,
        hitl=bool(doc.get("hitl", False)),
        plan=bool(doc.get("plan", False)),
    )


__all__ = [
    "RUN_INPUT_BLOB_VERSION",
    "RunInput",
    "RunInputError",
    "encode_run_input",
    "decode_run_input",
]
