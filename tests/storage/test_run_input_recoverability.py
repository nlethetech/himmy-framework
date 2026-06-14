"""Q0: persist run INPUT so a crashed QUEUED run is re-runnable, not just re-recorded.

The leased queue's whole value is that a recovered ``QUEUED`` run can be EXECUTED from a fresh
process. These cover the recoverable-input blob: it round-trips persona/task/llm_config/spec/
flags losslessly, is encrypted at rest when a key is configured (the prompt is sensitive), is
re-loadable from a SEPARATE store instance on the same file (the crash-recovery proxy), and
fails LOUD on an unknown blob version rather than mis-executing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.config.agent_spec import AgentSpec
from himmy.services.inference.models import LLMConfig
from himmy.services.storage.at_rest import StorePayloadCipher
from himmy.services.storage.encryption import FieldEncryptor
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.run_input import (
    RunInputError,
    decode_run_input,
    encode_run_input,
)
from himmy.services.storage.sqlite import SqliteStorageService
from tests.conftest import run_async


def _inputs() -> tuple[Persona, Task, LLMConfig, AgentSpec]:
    persona = Persona(name="researcher", instructions=["be precise"])
    task = Task(title="t", prompt="SECRET-PROMPT-XYZ", context={"k": "v"})
    llm = LLMConfig(model_key="ollama:llama3", temperature=0.2)
    spec = AgentSpec(name="agent-a", description="d")
    return persona, task, llm, spec


def test_blob_round_trips_all_inputs_plaintext() -> None:
    persona, task, llm, spec = _inputs()
    blob = encode_run_input(
        persona=persona,
        task=task,
        llm_config=llm,
        agent_spec=spec,
        hitl=True,
        plan=False,
        run_id="run-1",
    )
    ri = decode_run_input(blob, run_id="run-1")
    assert ri.persona.name == "researcher"
    assert ri.task.prompt == "SECRET-PROMPT-XYZ"
    assert ri.task.context == {"k": "v"}
    assert ri.llm_config is not None and ri.llm_config.model_key == "ollama:llama3"
    assert ri.agent_spec is not None and ri.agent_spec.name == "agent-a"
    assert ri.hitl is True and ri.plan is False


def test_blob_round_trips_minimal_inputs() -> None:
    """A run with no llm_config / agent_spec (inline-persona one-shot) still round-trips."""
    persona = Persona(name="p")
    task = Task(title="t", prompt="hello")
    blob = encode_run_input(persona=persona, task=task, run_id="r")
    ri = decode_run_input(blob, run_id="r")
    assert ri.llm_config is None and ri.agent_spec is None
    assert ri.task.prompt == "hello"


def test_blob_encrypted_at_rest_when_key_configured() -> None:
    persona, task, _llm, _spec = _inputs()
    cipher = StorePayloadCipher(FieldEncryptor.generate())
    blob = encode_run_input(persona=persona, task=task, run_id="r", cipher=cipher)
    assert "SECRET-PROMPT-XYZ" not in blob  # ciphertext, not plaintext
    ri = decode_run_input(blob, run_id="r", cipher=cipher)
    assert ri.task.prompt == "SECRET-PROMPT-XYZ"


def test_encrypt_is_idempotent() -> None:
    persona, task, _llm, _spec = _inputs()
    cipher = StorePayloadCipher(FieldEncryptor.generate())
    once = encode_run_input(persona=persona, task=task, run_id="r", cipher=cipher)
    # Re-encrypting an already-enveloped blob must not double-wrap.
    twice = cipher.encrypt_run_input(once, run_id="r")
    assert twice == once
    assert decode_run_input(twice, run_id="r", cipher=cipher).task.prompt == task.prompt


def test_unknown_blob_version_fails_loud() -> None:
    bad = '{"v": 999, "persona": {}, "task": {}}'
    with pytest.raises(RunInputError):
        decode_run_input(bad, run_id="r")


def test_malformed_blob_fails_loud() -> None:
    with pytest.raises(RunInputError):
        decode_run_input("not json at all", run_id="r")
    with pytest.raises(RunInputError):
        decode_run_input('{"v":1}', run_id="r")  # missing persona/task


def test_crash_recovery_from_fresh_store_instance(tmp_path: Path) -> None:
    """A QUEUED run's input is recoverable + re-executable from a SEPARATE store instance.

    The crash-recovery proxy: instance #1 persists a QUEUED run carrying its input_blob, then
    is closed (process exit). Instance #2 opens the SAME file, claims the run, decodes the blob
    back to the exact launch input. This is the Q0 guarantee — re-runnable, not just re-recorded.
    """
    path = str(tmp_path / "storage.db")
    cipher = StorePayloadCipher(FieldEncryptor.generate())
    persona, task, llm, spec = _inputs()

    # --- process #1: enqueue with recoverable input, then "crash" (close) ---
    store1 = SqliteStorageService(path, cipher=cipher)
    run = RunRecord(workspace_id="w", subject_id="s", status=RunStatus.QUEUED)
    run.input_blob = encode_run_input(
        persona=persona,
        task=task,
        llm_config=llm,
        agent_spec=spec,
        run_id=run.run_id,
        cipher=cipher,
    )
    run_async(store1.save_run(run))
    # The prompt must not be on disk in cleartext.
    raw = store1._conn.execute(
        "SELECT payload FROM runs WHERE run_id = ?", (run.run_id,)
    ).fetchone()["payload"]
    assert "SECRET-PROMPT-XYZ" not in raw
    run_async(store1.close())

    # --- process #2: recover ---
    store2 = SqliteStorageService(path, cipher=cipher)
    claimed = run_async(store2.claim_next_queued_run("recovery-worker", 300))
    assert claimed is not None and claimed.run_id == run.run_id
    assert claimed.input_blob  # carried through the claim
    recovered = decode_run_input(
        claimed.input_blob, run_id=claimed.run_id, cipher=cipher
    )
    assert recovered.task.prompt == "SECRET-PROMPT-XYZ"
    assert recovered.persona.name == "researcher"
    assert recovered.llm_config.model_key == "ollama:llama3"
    assert recovered.agent_spec.name == "agent-a"
    run_async(store2.close())
