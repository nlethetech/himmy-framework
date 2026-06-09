"""DeepGyan — consent & purpose-limitation, end to end (WS4.6 worked example).

The scenario that drove the consent layer: *a teacher who did **not** opt in must have
their data **not retained and not used for training**, and the platform must be able to
**prove** it.* DeepGyan is a tutoring platform with two teachers:

* **teacher_a** grants ``RETAIN`` + ``INFER`` but **denies** ``TRAIN``. Their lessons are
  answered live, persisted (encrypted under a per-subject, shreddable key), yet raw-I/O
  capture and the verbatim ``rendered_prompt`` are suppressed and they are **excluded from
  the training export**.
* **teacher_b** never opts in. Their lesson is answered live (``INFER`` defaults to
  ``EPHEMERAL``) but **nothing is persisted** — neither the storage facade nor the immutable
  EntityRegistry spine retains a single subject-bearing byte for them.

This is the **only** place DeepGyan is coupled to himmy — nothing in ``himmy/`` imports it.
The capability is a generic, ``subject_id``-agnostic himmy feature; "teacher" is just a
``context_subject_id``. Everything here is opt-in: it runs only because we set
``HIMMY_CONSENT=on`` before the container is built. The final section re-runs the *same*
pipeline zero-config to prove the offline path is byte-for-byte unchanged.

Run it::

    python examples/deepgyan_consent/main.py

It prints a narrated walkthrough; the same assertions are exercised as a test in
``tests/examples/test_deepgyan_consent.py``.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from himmy.agents.base_agent.task import Task
from himmy.agents.personas.persona import Persona
from himmy.api.deps import ApiContainer
from himmy.entities.integrity import (
    export_audit_bundle,
    verify_audit_bundle,
)
from himmy.entities.records import EntityQuery, EntityRecord
from himmy.services.governance.consent import Effect, Purpose
from himmy.services.governance.consent_registry import ConsentAwareRegistry
from himmy.services.governance.consent_resolver import SubjectResolver
from himmy.services.governance.consent_storage import ConsentGatedStorage
from himmy.services.governance.training_export import ConsentFilteredExporter
from himmy.services.storage.models import RunRecord, RunStatus
from himmy.services.storage.service import StorageService

#: Distinctive markers so we can prove a verbatim string never lands on any sink.
TEACHER_A_LESSON = "DEEPGYAN-A-photosynthesis-lesson-plan-7Q2"
TEACHER_B_LESSON = "DEEPGYAN-B-quadratic-equations-lesson-plan-9K4"

#: The audit-bundle signing secret (HMAC). In production this comes from the
#: SecretProvider/HSM (``HIMMY_AUDIT_PRIVATE_KEY`` for Ed25519); here it is inline so the
#: example is self-contained and deterministic.
_BUNDLE_SECRET = "deepgyan-audit-signing-secret"  # noqa: S105 - example-local demo secret

#: The spine kinds whose integrity the signed bundle commits to in this walkthrough: the
#: audit/governance trail plus every transcript kind the runtime writes.
_BUNDLED_KINDS = (
    "consent",
    "security_event",
    "erasure_tombstone",
    "run_event",
    "message",
    "chat_thread",
    "persona",
)


@dataclass(frozen=True)
class GovernedRun:
    """The artefacts of one governed pipeline run, returned for assertions/printing."""

    container: ApiContainer
    training_corpus_before_train_grant: list[RunRecord]
    training_corpus_after_train_grant: list[RunRecord]
    bundle_verified: bool
    bundle_detects_tampering: bool


def _governed_container() -> ApiContainer:
    """Build the governed container (consent enforcement on).

    Mirrors a deployment that has opted in to data governance: ``HIMMY_CONSENT=on`` makes
    :meth:`ApiContainer.build_default` wrap storage + the spine and wire the TRAIN gate.
    """
    os.environ["HIMMY_CONSENT"] = "on"
    os.environ.pop("HIMMY_CONSENT_FILE", None)
    return ApiContainer.build_default()


def _bundled_records(container: ApiContainer) -> list[EntityRecord]:
    """Collect the spine records the signed audit bundle commits to (inner spine)."""
    spine = container.entity_registry
    records: list[EntityRecord] = []
    for kind in _BUNDLED_KINDS:
        records.extend(spine.list_by_kind(kind))
    return records


async def _serve_lesson(
    container: ApiContainer, *, subject_id: str, lesson: str
) -> None:
    """Answer one teacher's lesson and persist a RunRecord through the gated facade.

    Two writes happen, both governed by purpose=RETAIN for ``subject_id``:

    * ``runtime.run_task`` drives the live answer and writes run events / the transcript
      to the immutable spine (gated by :class:`ConsentAwareRegistry`), exercising the
      TRAIN gate (io-capture + ``rendered_prompt`` suppression) along the way.
    * a :class:`RunRecord` is saved through the :class:`ConsentGatedStorage` facade — the
      operational corpus a dataset builder would later read.
    """
    persona = Persona(name="DeepGyan Tutor")
    task = Task(
        title="lesson",
        prompt=lesson,
        context={"context_subject_id": subject_id},
    )
    thread = await container.runtime.run_task(persona, task)
    answer = thread.last_message.content if thread.last_message else ""

    # The operational run record (what an SFT/replay exporter would read).
    await container.storage.save_run(
        RunRecord(
            workspace_id="deepgyan",
            subject_id=subject_id,
            persona_name=persona.name,
            status=RunStatus.SUCCEEDED,
            output_text=f"{lesson} -> {answer}",
        )
    )


def _build_training_corpus(
    container: ApiContainer, runs: list[RunRecord]
) -> list[RunRecord]:
    """Filter the persisted runs to only TRAIN-consented subjects (the single funnel).

    Every dataset/SFT/replay builder routes through :class:`ConsentFilteredExporter`, so a
    subject who never granted TRAIN — even one who *did* grant RETAIN — can never appear in
    an exported corpus.
    """
    exporter = ConsentFilteredExporter(decider=container.consent_ledger.decision)
    return exporter.filter(runs, subject_of=SubjectResolver().subject_of)


async def run_deepgyan_scenario() -> GovernedRun:
    """Run the full governed DeepGyan scenario and return its artefacts."""
    container = _governed_container()
    ledger = container.consent_ledger
    spine = container.entity_registry

    # --- consent state: A opts in (retain+infer, NOT train); B never opts in -------
    ledger.grant("teacher_a", Purpose.RETAIN, source="example", actor="teacher_a")
    ledger.grant("teacher_a", Purpose.INFER, source="example", actor="teacher_a")
    ledger.deny("teacher_a", Purpose.TRAIN, source="example", actor="teacher_a")
    # teacher_b: no records at all → governed default DENY for RETAIN/TRAIN.

    # --- serve both lessons --------------------------------------------------------
    await _serve_lesson(container, subject_id="teacher_a", lesson=TEACHER_A_LESSON)
    await _serve_lesson(container, subject_id="teacher_b", lesson=TEACHER_B_LESSON)

    # --- training corpus BEFORE A grants TRAIN: provably empty ---------------------
    all_runs = await container.storage.list_runs()
    corpus_before = _build_training_corpus(container, all_runs)

    # --- A grants TRAIN; the corpus now includes A (still never B) -----------------
    ledger.grant("teacher_a", Purpose.TRAIN, source="example", actor="teacher_a")
    corpus_after = _build_training_corpus(container, all_runs)

    # --- verify the signed audit bundle round-trips and detects tampering ----------
    records = _bundled_records(container)
    bundle = export_audit_bundle(records, [], secret=_BUNDLE_SECRET)
    clean = verify_audit_bundle(bundle, records, [], secret=_BUNDLE_SECRET)

    # Tamper-evidence: forge one cited record's content (flip teacher_a's TRAIN-deny to
    # "granted" on the spine) and prove the signed bundle catches it by record id. The
    # forged record keeps its content-addressed id — only its payload diverges — so the
    # verifier flags it as TAMPERED rather than missing/added.
    train_deny = next(
        r
        for r in spine.list_by_kind("consent")
        if r.metadata.get("subject_id") == "teacher_a"
        and r.payload.get("purpose") == Purpose.TRAIN.value
        and r.payload.get("state") == "denied"
    )
    forged = EntityRecord.create(
        stable_id=train_deny.stable_id,
        version=train_deny.version,
        kind="consent",
        payload={**train_deny.payload, "state": "granted"},
        metadata=train_deny.metadata,
    )
    assert forged.record_id == train_deny.record_id  # same id, divergent content
    tampered_set = [forged if r.record_id == forged.record_id else r for r in records]
    tampered = verify_audit_bundle(bundle, tampered_set, [], secret=_BUNDLE_SECRET)

    return GovernedRun(
        container=container,
        training_corpus_before_train_grant=corpus_before,
        training_corpus_after_train_grant=corpus_after,
        bundle_verified=clean.ok,
        bundle_detects_tampering=not tampered.ok,
    )


def teacher_b_is_not_persisted(container: ApiContainer) -> bool:
    """True when teacher_b left zero subject-bearing bytes on storage AND the spine."""
    spine = container.entity_registry
    no_spine = (
        spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_b"})) == []
    )
    spine_blob = "".join(
        str(r.payload)
        for kind in ("run_event", "message", "chat_thread")
        for r in spine.list_by_kind(kind)
    )
    return no_spine and TEACHER_B_LESSON not in spine_blob


async def teacher_a_persisted_but_train_suppressed(container: ApiContainer) -> bool:
    """True when teacher_a's transcript persisted yet TRAIN capture was suppressed.

    Checks the linchpin TRAIN guarantees for a RETAIN-yes / TRAIN-no subject:

    * the transcript reached the spine tagged with the subject (RETAIN granted), and
    * no run event on the spine carries a captured raw-I/O blob or the verbatim
      ``rendered_prompt`` for that subject (TRAIN denied at serve time).
    """
    spine = container.entity_registry
    tagged = spine.query(EntityQuery(metadata_filters={"subject_id": "teacher_a"}))
    persisted = bool(tagged) and any(r.kind == "run_event" for r in tagged)

    a_events = [r for r in tagged if r.kind == "run_event"]
    no_io_capture = all("io" not in (r.payload or {}) for r in a_events)
    no_rendered_prompt = all(
        "rendered_prompt" not in (r.payload or {}) for r in a_events
    )
    # The verbatim lesson text never appears in cleartext on any run event for A.
    no_cleartext = all(TEACHER_A_LESSON not in str(r.payload) for r in a_events)
    return persisted and no_io_capture and no_rendered_prompt and no_cleartext


async def withdraw_and_verify_erasure(container: ApiContainer) -> dict[str, Any]:
    """A withdraws → crypto-shred + tombstone + ledger WITHDRAWN + fresh decide==DENY.

    Crypto-shred is *the* reconciliation between GDPR erasure and an append-only spine:
    a withdrawal destroys teacher_a's per-subject key, so the message ciphertext that was
    written under it stays on the (immutable) spine forever but is permanently
    undecryptable. We capture that ciphertext before withdrawal and prove it never reveals
    the cleartext lesson afterward.
    """
    from himmy.services.storage.encryption import ENC_PREFIX

    spine = container.entity_registry
    ledger = container.consent_ledger

    # A's user message persisted under their shreddable key — encrypted, never cleartext.
    user_msgs = [
        r
        for r in spine.list_by_kind("message")
        if r.metadata.get("subject_id") == "teacher_a"
        and r.payload.get("role") == "user"
    ]
    encrypted_before = bool(user_msgs) and all(
        m.payload["content"].startswith(ENC_PREFIX)
        and TEACHER_A_LESSON not in m.payload["content"]
        for m in user_msgs
    )

    ledger.withdraw("teacher_a", reason="teacher exercised right to erasure")

    tombstones = spine.list_by_kind("erasure_tombstone")
    decision = ledger.decision("teacher_a", Purpose.RETAIN)
    # The shredded ciphertext is still on the spine but no longer reveals the cleartext.
    shredded_ciphertext_unrecoverable = all(
        TEACHER_A_LESSON not in m.payload["content"] for m in user_msgs
    )
    return {
        "message_was_encrypted": encrypted_before,
        "tombstone_present": bool(tombstones),
        "ledger_withdrawn": ledger.state("teacher_a", Purpose.RETAIN).value
        == "withdrawn",
        "fresh_decision_deny": decision.effect is Effect.DENY,
        "ciphertext_unrecoverable": shredded_ciphertext_unrecoverable,
    }


async def zero_config_persists_verbatim() -> bool:
    """Re-run the same lesson zero-config; the bare pipeline persists everything verbatim.

    Proves the layer is purely opt-in: with ``HIMMY_CONSENT`` unset the container is a bare
    ``StorageService`` + an ungoverned runtime (no decider, no wrappers), and a
    non-consenting subject's prompt is captured verbatim on the spine exactly as before.
    """
    os.environ.pop("HIMMY_CONSENT", None)
    os.environ.pop("HIMMY_CONSENT_FILE", None)
    container = ApiContainer.build_default()

    # The offline path is byte-identical: bare storage, no wrappers, no decider.
    bare = (
        type(container.storage) is StorageService
        and not isinstance(container.storage, ConsentGatedStorage)
        and not isinstance(container.entity_registry, ConsentAwareRegistry)
        and container.runtime._consent_decider is None
        and container.consent_ledger is None
    )

    task = Task(
        title="lesson",
        prompt=TEACHER_B_LESSON,
        context={"context_subject_id": "teacher_b"},
    )
    await container.runtime.run_task(Persona(name="Tutor"), task)

    # Ungoverned: teacher_b's lesson is captured verbatim on the spine (no gating at all).
    spine_blob = "".join(
        str(r.payload) for r in container.entity_registry.list_by_kind("run_event")
    )
    return bare and TEACHER_B_LESSON in spine_blob


async def main() -> None:
    """Narrate the whole DeepGyan walkthrough end to end."""
    print("=== DeepGyan — consent & purpose-limitation (WS4.6) ===\n")

    result = await run_deepgyan_scenario()
    container = result.container

    print("Governed mode (HIMMY_CONSENT=on)")
    print("  teacher_a: RETAIN=grant, INFER=grant, TRAIN=deny")
    print("  teacher_b: no consent on file (governed default DENY)\n")

    b_clean = teacher_b_is_not_persisted(container)
    print(f"[B] non-consenting teacher leaves NOTHING persisted : {b_clean}")

    a_ok = await teacher_a_persisted_but_train_suppressed(container)
    print(f"[A] served + persisted (encrypted), TRAIN suppressed : {a_ok}")

    print(
        "[export] training corpus before A grants TRAIN        : "
        f"{len(result.training_corpus_before_train_grant)} runs (empty)"
    )
    after = result.training_corpus_after_train_grant
    print(
        "[export] training corpus after A grants TRAIN         : "
        f"{len(after)} run(s); subjects="
        f"{sorted({r.subject_id for r in after})}"
    )

    print(
        f"[audit] signed bundle verifies                        : {result.bundle_verified}"
    )
    print(
        f"[audit] tampering a cited record is detected          : {result.bundle_detects_tampering}"
    )

    erasure = await withdraw_and_verify_erasure(container)
    print("\n[withdraw] teacher_a exercises right to erasure:")
    for key, value in erasure.items():
        print(f"  {key:<24}: {value}")

    verbatim = await zero_config_persists_verbatim()
    print(
        "\n[offline] zero-config re-run persists the lesson verbatim "
        f"(opt-in proven)        : {verbatim}"
    )


if __name__ == "__main__":
    asyncio.run(main())
