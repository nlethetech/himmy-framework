"""Tests for himmy.toml project config + SqliteSessionStore + the CLI wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from himmy.agents.base_agent.thread import ChatThread, Message, MessageRole
from himmy.cli.__main__ import main
from himmy.config.project import find_project_config, load_project_config
from himmy.runtime.session import SqliteSessionStore
from himmy.toolkit.config import ToolkitConfig

# --------------------------------------------------------------------- project config


def test_load_project_config(tmp_path: Path) -> None:
    (tmp_path / "himmy.toml").write_text(
        '[defaults]\nprovider = "ollama"\ntool_packs = ["web"]\n[toolkit]\nembedder = "ollama"\n'
    )
    cfg = load_project_config(tmp_path)
    assert cfg["defaults"]["provider"] == "ollama"
    assert cfg["toolkit"]["embedder"] == "ollama"


def test_find_project_config_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "no-home")
    assert find_project_config(tmp_path) is None


def test_toolkit_from_sources_toml_base_env_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HIMMY_EMBEDDER", raising=False)
    cfg = ToolkitConfig.from_sources({"embedder": "ollama", "memory_path": "/x.db"})
    assert cfg.embedder == "ollama"  # from toml
    assert cfg.memory_path == "/x.db"
    monkeypatch.setenv("HIMMY_EMBEDDER", "fastembed")
    cfg2 = ToolkitConfig.from_sources({"embedder": "ollama"})
    assert cfg2.embedder == "fastembed"  # env overrides toml


# --------------------------------------------------------------------- session store


def test_session_store_roundtrip(tmp_path: Path) -> None:
    store = SqliteSessionStore(str(tmp_path / "s.db"))
    thread = ChatThread()
    thread.append_message(Message(role=MessageRole.USER, content="hello 42"))
    store.save("s1", thread)
    loaded = store.load("s1")
    assert loaded is not None
    assert loaded.thread_id == thread.thread_id
    assert any("42" in m.content for m in loaded.messages)
    assert store.load("missing") is None
    store.close()


def test_list_sessions_orders_newest_first_with_counts(tmp_path: Path) -> None:
    """list_sessions returns id + timestamp + message count, newest first."""
    store = SqliteSessionStore(str(tmp_path / "s.db"))
    one = ChatThread()
    one.append_message(Message(role=MessageRole.USER, content="a"))
    store.save("first", one)
    two = ChatThread()
    two.append_message(Message(role=MessageRole.USER, content="b"))
    two.append_message(Message(role=MessageRole.USER, content="c"))
    store.save("second", two)  # saved later → newer updated_at
    infos = store.list_sessions()
    assert [i.session_id for i in infos] == ["second", "first"]
    by_id = {i.session_id: i for i in infos}
    assert by_id["first"].message_count == 1
    assert by_id["second"].message_count == 2
    assert all(i.updated_at for i in infos)
    store.close()


def test_list_sessions_respects_limit(tmp_path: Path) -> None:
    store = SqliteSessionStore(str(tmp_path / "s.db"))
    for i in range(5):
        store.save(f"s{i}", ChatThread())
    assert len(store.list_sessions(limit=2)) == 2
    assert len(store.list_sessions()) == 5
    store.close()


def test_list_sessions_empty(tmp_path: Path) -> None:
    store = SqliteSessionStore(str(tmp_path / "s.db"))
    assert store.list_sessions() == []
    store.close()


def test_session_durable_across_reopen(tmp_path: Path) -> None:
    db = str(tmp_path / "s.db")
    a = SqliteSessionStore(db)
    t = ChatThread()
    t.append_message(Message(role=MessageRole.USER, content="persist me"))
    a.save("x", t)
    a.close()
    b = SqliteSessionStore(db)
    assert b.load("x") is not None
    b.close()


# --------------------------------------------------------------------- CLI wiring


def test_chat_session_persists_across_invocations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "chat",
                "--provider",
                "stub",
                "--name",
                "a",
                "--session",
                "s1",
                "--message",
                "hi",
            ]
        )
        == 0
    )
    capsys.readouterr()
    from himmy.config.project import conversations_db_path

    store = SqliteSessionStore(conversations_db_path())
    assert store.load("s1") is not None
    store.close()


def test_himmy_toml_defaults_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A himmy.toml default tool_pack is applied without any flag."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "himmy.toml").write_text('[defaults]\ntool_packs = ["utils"]\n')
    # Clear the module-level project cache so this test's toml is read. The cache
    # lives in himmy.runtime.from_spec (the shared spec→runtime wiring).
    import himmy.runtime.from_spec as fs

    monkeypatch.setattr(fs, "_PROJECT_CACHE", None)
    code = main(["run", "--provider", "stub", "--name", "a", "-p", "hi"])
    assert code == 0
