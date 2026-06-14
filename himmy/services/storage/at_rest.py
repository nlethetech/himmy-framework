"""Field encryption at rest for the durable stores (wires WS4.4 into SQLite/Postgres).

The envelope :class:`~himmy.services.storage.encryption.FieldEncryptor` machinery is
opt-in and was previously unreferenced by any storage backend — configuring
``HIMMY_ENCRYPTION_KEY`` / ``HIMMY_KEK_PROVIDER`` changed nothing and the durable stores
wrote plaintext regardless. :class:`StorePayloadCipher` is the single seam that closes
that gap: it encrypts the *sensitive* fields of the records the durable backends persist
(chat-message content and run-event payloads) before they hit the ``payload`` column, and
decrypts them transparently on read.

It is **opt-in** exactly as documented: :func:`build_store_cipher` returns ``None`` when no
key/provider is configured, so the offline plaintext path (and the entire existing test
suite) is byte-identical. When a key *is* configured the cipher actually encrypts:

* **chat threads** — each message's ``content`` (nested in ``messages[*]``) is encrypted,
  bound to the ``thread_id`` as AAD so a ciphertext can't be moved to another thread.
* **run events** — the ``payload`` dict (tool I/O, inference snippets) is JSON-encoded and
  encrypted as one value, bound to the ``event_id`` as AAD.

Decryption is idempotent on plaintext (legacy rows written before a key was configured
round-trip untouched), so enabling encryption never bricks existing data. Only the
``payload`` column content changes — the indexed filter columns (``thread_id`` /
``trace_id`` / ...) stay clear so list queries are unaffected.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.storage.encryption import FieldEncryptor

#: Wrapper key under which an encrypted run-event payload is stashed. A payload dict that
#: contains this single key is a ciphertext envelope (so decrypt can recognise it); a plain
#: dict is left untouched (legacy/plaintext rows).
_ENC_PAYLOAD_FIELD = "__himmy_enc_payload__"

#: Prefix marking an encrypted run-input blob (Q0). The blob is a single opaque STRING (the
#: encrypted run-input JSON), not a dict envelope like the event payload, so a sentinel prefix
#: is what distinguishes a ciphertext blob from a legacy/plaintext one on the decrypt path.
_RUN_INPUT_ENC_PREFIX = "enc1:"


class StorePayloadCipher:
    """Encrypt/decrypt the sensitive fields of durable storage payloads (at rest).

    Holds an envelope :class:`FieldEncryptor` and applies it to the two record shapes the
    durable backends persist with sensitive content. All methods return a *new* dict and
    never mutate their argument, and are idempotent on already-plaintext input so they are
    safe to apply unconditionally on the read path.
    """

    def __init__(self, encryptor: FieldEncryptor) -> None:
        self._enc = encryptor

    # ------------------------------------------------------------------ chat threads
    def encrypt_thread_payload(
        self, payload: dict[str, Any], *, thread_id: str
    ) -> dict[str, Any]:
        """Return a copy of a thread payload with each message's ``content`` encrypted.

        ``thread_id`` is the AAD, binding every ciphertext to its thread. Already-encrypted
        content is left as-is (idempotent), so re-saving a thread never double-encrypts.
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload
        aad = thread_id.encode("utf-8")
        out = dict(payload)
        out["messages"] = [
            self._encrypt_message(msg, aad) if isinstance(msg, dict) else msg
            for msg in messages
        ]
        return out

    def decrypt_thread_payload(
        self, payload: dict[str, Any], *, thread_id: str
    ) -> dict[str, Any]:
        """Return a copy of a thread payload with each message's ``content`` decrypted.

        Plaintext content (legacy rows) round-trips untouched.
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return payload
        aad = thread_id.encode("utf-8")
        out = dict(payload)
        out["messages"] = [
            self._decrypt_message(msg, aad) if isinstance(msg, dict) else msg
            for msg in messages
        ]
        return out

    def _encrypt_message(self, message: dict[str, Any], aad: bytes) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, str) or self._enc.is_encrypted(content):
            return message
        out = dict(message)
        out["content"] = self._enc.encrypt(content, aad=aad)
        return out

    def _decrypt_message(self, message: dict[str, Any], aad: bytes) -> dict[str, Any]:
        content = message.get("content")
        if not isinstance(content, str) or not self._enc.is_encrypted(content):
            return message
        out = dict(message)
        out["content"] = self._enc.decrypt(content, aad=aad)
        return out

    # -------------------------------------------------------------------- run events
    def encrypt_event_payload(
        self, payload: dict[str, Any], *, event_id: str
    ) -> dict[str, Any]:
        """Return an encrypted envelope for a run-event ``payload`` dict.

        The whole payload is JSON-encoded and encrypted as one value (bound to the
        ``event_id`` as AAD), then wrapped under :data:`_ENC_PAYLOAD_FIELD`. An empty
        payload and an already-wrapped payload are returned untouched.
        """
        if not payload or _ENC_PAYLOAD_FIELD in payload:
            return payload
        token = self._enc.encrypt(
            json.dumps(payload, sort_keys=True), aad=event_id.encode("utf-8")
        )
        return {_ENC_PAYLOAD_FIELD: token}

    def decrypt_event_payload(
        self, payload: dict[str, Any], *, event_id: str
    ) -> dict[str, Any]:
        """Return the cleartext run-event ``payload`` dict from an envelope (or as-is).

        A plaintext payload (legacy row, or one written with encryption off) is returned
        unchanged.
        """
        token = payload.get(_ENC_PAYLOAD_FIELD)
        if not isinstance(token, str):
            return payload
        decoded = json.loads(self._enc.decrypt(token, aad=event_id.encode("utf-8")))
        return decoded if isinstance(decoded, dict) else payload

    # ------------------------------------------------------------------- run input (Q0)
    def encrypt_run_input(self, blob_json: str, *, run_id: str) -> str:
        """Encrypt a serialized run-input JSON string, bound to ``run_id`` as AAD.

        Returns a single opaque string carrying the :data:`_RUN_INPUT_ENC_PREFIX` sentinel.
        Idempotent: an already-encrypted blob (one already carrying the prefix) is returned
        untouched so re-saving a run never double-encrypts.
        """
        if blob_json.startswith(_RUN_INPUT_ENC_PREFIX):
            return blob_json
        token = self._enc.encrypt(blob_json, aad=run_id.encode("utf-8"))
        return _RUN_INPUT_ENC_PREFIX + token

    def decrypt_run_input(self, blob: str, *, run_id: str) -> str:
        """Return the cleartext run-input JSON from an encrypted blob (or as-is).

        A plaintext blob (a run persisted with encryption off, then read after a key was
        configured, or vice-versa) lacks the sentinel prefix and is returned unchanged.
        """
        if not blob.startswith(_RUN_INPUT_ENC_PREFIX):
            return blob
        token = blob[len(_RUN_INPUT_ENC_PREFIX) :]
        return self._enc.decrypt(token, aad=run_id.encode("utf-8"))


def build_store_cipher() -> StorePayloadCipher | None:
    """Build a :class:`StorePayloadCipher` from the configured KEK, or ``None`` (off).

    Delegates to :func:`~himmy.services.storage.encryption.build_field_encryptor`, so the
    backend (local key / aws-kms / ...) is selected exactly as documented and the offline
    plaintext path is preserved when nothing is configured.
    """
    from himmy.services.storage.encryption import build_field_encryptor

    encryptor = build_field_encryptor()
    return StorePayloadCipher(encryptor) if encryptor is not None else None


__all__ = ["StorePayloadCipher", "build_store_cipher"]
