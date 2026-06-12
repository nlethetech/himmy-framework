"""Field encryption at rest: the ``StorePayloadCipher`` + its wiring into the stores.

Proves the previously-dead envelope encryption actually fires once a key is configured:
round-trips for both record shapes, ciphertext leaves no plaintext on disk, AAD binds a
ciphertext to its record, decryption is idempotent on legacy plaintext rows, and the
durable SQLite store encrypts message content + run-event payloads at rest while keeping
the indexed filter columns clear.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

pytest.importorskip("cryptography")

from himmy.agents.base_agent.thread import (  # noqa: E402
    ChatThread,
    Message,
    MessageRole,
)
from himmy.core.events import EventType, RunEvent  # noqa: E402
from himmy.services.storage.at_rest import (  # noqa: E402
    StorePayloadCipher,
    build_store_cipher,
)
from himmy.services.storage.encryption import ENC_PREFIX, FieldEncryptor  # noqa: E402
from himmy.services.storage.sqlite import SqliteStorageService  # noqa: E402
from tests.conftest import run_async  # noqa: E402


def _cipher() -> StorePayloadCipher:
    return StorePayloadCipher(FieldEncryptor.generate())


# --- StorePayloadCipher: thread message content -----------------------------------


def test_thread_payload_round_trips_and_hides_content() -> None:
    cipher = _cipher()
    payload = {
        "thread_id": "t1",
        "messages": [
            {"role": "user", "content": "buy 100 shares of ACME"},
            {"role": "assistant", "content": "order placed"},
        ],
    }
    enc = cipher.encrypt_thread_payload(payload, thread_id="t1")
    blob = json.dumps(enc)
    assert "buy 100 shares" not in blob and "order placed" not in blob
    assert enc["messages"][0]["content"].startswith(ENC_PREFIX)
    dec = cipher.decrypt_thread_payload(enc, thread_id="t1")
    assert [m["content"] for m in dec["messages"]] == [
        "buy 100 shares of ACME",
        "order placed",
    ]


def test_thread_aad_binds_content_to_its_thread() -> None:
    cipher = _cipher()
    enc = cipher.encrypt_thread_payload(
        {"messages": [{"role": "user", "content": "secret"}]}, thread_id="t1"
    )
    # Decrypting under a different thread_id (AAD) must fail, not silently return junk.
    from cryptography.exceptions import InvalidTag

    with pytest.raises(InvalidTag):
        cipher.decrypt_thread_payload(enc, thread_id="other-thread")


def test_thread_encrypt_is_idempotent_on_plaintext_and_ciphertext() -> None:
    cipher = _cipher()
    once = cipher.encrypt_thread_payload(
        {"messages": [{"role": "user", "content": "hi"}]}, thread_id="t1"
    )
    twice = cipher.encrypt_thread_payload(once, thread_id="t1")
    assert once == twice  # re-saving never double-encrypts
    # A payload with no messages list is returned untouched.
    assert cipher.encrypt_thread_payload({"thread_id": "t1"}, thread_id="t1") == {
        "thread_id": "t1"
    }


# --- StorePayloadCipher: run-event payloads ---------------------------------------


def test_event_payload_round_trips_and_hides_content() -> None:
    cipher = _cipher()
    payload = {"prompt": "transfer funds", "tokens": 42}
    enc = cipher.encrypt_event_payload(payload, event_id="ev1")
    assert "transfer funds" not in json.dumps(enc)
    assert cipher.decrypt_event_payload(enc, event_id="ev1") == payload


def test_event_aad_binds_payload_to_its_event() -> None:
    cipher = _cipher()
    enc = cipher.encrypt_event_payload({"k": "v"}, event_id="ev1")
    from cryptography.exceptions import InvalidTag

    with pytest.raises(InvalidTag):
        cipher.decrypt_event_payload(enc, event_id="ev2")


def test_event_empty_payload_and_plaintext_round_trip_untouched() -> None:
    cipher = _cipher()
    assert cipher.encrypt_event_payload({}, event_id="ev1") == {}
    # A legacy plaintext payload (no envelope) decrypts to itself.
    assert cipher.decrypt_event_payload({"k": "v"}, event_id="ev1") == {"k": "v"}


# --- build_store_cipher: opt-in -----------------------------------------------------


def test_build_store_cipher_off_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_KEK_PROVIDER", raising=False)
    monkeypatch.delenv("HIMMY_ENCRYPTION_KEY", raising=False)
    from himmy.config.secrets import configure_secrets

    class _Secrets:
        def get(self, name: str) -> str | None:
            return None

    configure_secrets(_Secrets())
    try:
        assert build_store_cipher() is None
    finally:
        configure_secrets(None)


def test_build_store_cipher_on_with_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_KEK_PROVIDER", raising=False)
    from himmy.config.secrets import configure_secrets
    from himmy.services.storage.encryption import LocalKekProvider

    key = LocalKekProvider.generate().key_b64()

    class _Secrets:
        def get(self, name: str) -> str | None:
            return key if name == "HIMMY_ENCRYPTION_KEY" else None

    configure_secrets(_Secrets())
    try:
        cipher = build_store_cipher()
        assert isinstance(cipher, StorePayloadCipher)
        enc = cipher.encrypt_event_payload({"k": "v"}, event_id="e")
        assert cipher.decrypt_event_payload(enc, event_id="e") == {"k": "v"}
    finally:
        configure_secrets(None)


# --- SQLite store: encryption actually lands on disk ------------------------------


def _encrypted_store(tmp_path: Path) -> SqliteStorageService:
    return SqliteStorageService(
        str(tmp_path / "storage.db"), cipher=StorePayloadCipher(FieldEncryptor.generate())
    )


def test_sqlite_thread_content_is_ciphertext_on_disk(tmp_path: Path) -> None:
    storage = _encrypted_store(tmp_path)
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="my SSN is 123-45-6789"))
    run_async(storage.save_thread(thread))

    # The raw payload column never holds the plaintext.
    raw = sqlite3.connect(str(tmp_path / "storage.db"))
    (payload,) = raw.execute(
        "SELECT payload FROM chat_threads WHERE thread_id = ?", (thread.thread_id,)
    ).fetchone()
    raw.close()
    assert "123-45-6789" not in payload

    # But load_thread transparently decrypts it back.
    loaded = run_async(storage.load_thread(thread.thread_id))
    assert loaded is not None
    assert loaded.messages[0].content == "my SSN is 123-45-6789"


def test_sqlite_event_payload_is_ciphertext_on_disk(tmp_path: Path) -> None:
    storage = _encrypted_store(tmp_path)
    event = RunEvent(
        event_type=EventType.AGENT_RUN_FINISHED,
        thread_id="t1",
        trace_id="tr1",
        payload={"secret_prompt": "wire $1M to account 999"},
    )
    run_async(storage.append_event(event))

    raw = sqlite3.connect(str(tmp_path / "storage.db"))
    (payload,) = raw.execute(
        "SELECT payload FROM run_events WHERE event_id = ?", (event.event_id,)
    ).fetchone()
    raw.close()
    assert "wire $1M" not in payload

    # list_events transparently decrypts. The indexed filter columns stayed clear.
    events = run_async(storage.list_events(thread_id="t1"))
    assert len(events) == 1
    assert events[0].payload == {"secret_prompt": "wire $1M to account 999"}


def test_sqlite_plaintext_store_is_unchanged(tmp_path: Path) -> None:
    """With no cipher (default offline path) the store writes plaintext as before."""
    storage = SqliteStorageService(str(tmp_path / "plain.db"), cipher=None)
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="hello"))
    run_async(storage.save_thread(thread))

    raw = sqlite3.connect(str(tmp_path / "plain.db"))
    (payload,) = raw.execute(
        "SELECT payload FROM chat_threads WHERE thread_id = ?", (thread.thread_id,)
    ).fetchone()
    raw.close()
    assert "hello" in payload  # plaintext when encryption is off

    loaded = run_async(storage.load_thread(thread.thread_id))
    assert loaded is not None and loaded.messages[0].content == "hello"


def test_sqlite_cross_instance_decrypt_with_shared_cipher(tmp_path: Path) -> None:
    """A second instance on the same file + same key reads the encrypted thread."""
    enc = FieldEncryptor.from_key_b64(FieldEncryptor.generate().key_b64())
    path = str(tmp_path / "shared.db")
    writer = SqliteStorageService(path, cipher=StorePayloadCipher(enc))
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="durable secret"))
    run_async(writer.save_thread(thread))

    reader = SqliteStorageService(path, cipher=StorePayloadCipher(enc))
    loaded = run_async(reader.load_thread(thread.thread_id))
    assert loaded is not None and loaded.messages[0].content == "durable secret"
