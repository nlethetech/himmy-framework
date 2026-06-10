"""Live tool-pack matrix: every GUI capability proven against a real model.

The 2026-06-10 report "tool usage may not be working — agents not researching,
not able to fetch news" turned out to be a spec problem (the onboarding wizard
created agents with NO tool packs). This suite makes the real thing
unfakeable: for each pack, a purpose-built agent runs on live Ollama through
the SAME studio stream the GUI uses, and the test asserts MECHANICS —

  * a tool frame from the expected pack actually fired,
  * the run terminated successfully with a non-empty answer,
  * and wherever a pack mutates state, the state is verified OUT-OF-BAND:
    notes/tasks/memory through their own APIs, files on disk, arithmetic by
    exact value.

Prose quality is never asserted. Skips as a block when Ollama is unreachable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

_MODEL = "qwen2.5:7b-instruct"
_OLLAMA = "http://localhost:11434"


def _ollama_up() -> bool:
    try:
        resp = httpx.get(f"{_OLLAMA}/api/tags", timeout=2.0)
        if resp.status_code != 200:
            return False
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        return any(n.startswith(_MODEL.split(":")[0]) for n in names)
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_up(), reason="local Ollama not reachable"),
]


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def client(project: Path) -> TestClient:
    from himmy.api.app import create_app

    with TestClient(create_app()) as c:
        yield c


def _agent(project: Path, packs: list[str], extra_instruction: str = "") -> str:
    """Write a single-purpose agent spec carrying exactly the packs under test."""
    spec: dict[str, Any] = {
        "name": "verifier",
        "description": "A tool-verification agent.",
        "instructions": [
            "You verify tools. ALWAYS use the tool the user names — never answer "
            "from memory. After the tool returns, answer briefly from its result."
        ]
        + ([extra_instruction] if extra_instruction else []),
        "tool_packs": packs,
    }
    path = "verifier.agent.yaml"
    (project / path).write_text(yaml.safe_dump(spec, sort_keys=False), "utf-8")
    return path


def _run(client: TestClient, agent_path: str, prompt: str) -> tuple[list[dict], dict]:
    """Stream one studio run; return (all frames, the done frame)."""
    frames: list[dict[str, Any]] = []
    with client.stream(
        "POST",
        "/api/studio/run",
        json={
            "agent_path": agent_path,
            "prompt": prompt,
            "provider": "ollama",
            "model": _MODEL,
        },
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    done = next((f for f in frames if f.get("type") == "done"), None)
    assert done is not None, f"no done frame; kinds={[f.get('type') for f in frames]}"
    return frames, done


def _run_with_approval(
    client: TestClient, agent_path: str, prompt: str
) -> tuple[list[dict], dict]:
    """Stream a run that pauses at an approval gate, approve it, and stream the
    resumed run to completion — the full HITL cycle, live."""
    frames: list[dict[str, Any]] = []
    with client.stream(
        "POST",
        "/api/studio/run",
        json={
            "agent_path": agent_path,
            "prompt": prompt,
            "provider": "ollama",
            "model": _MODEL,
        },
    ) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    paused = next((f for f in frames if f.get("type") == "paused"), None)
    assert paused is not None, (
        f"expected an approval pause; kinds={[f.get('type') for f in frames]}"
    )
    assert any(f.get("type") == "approval_required" for f in frames)
    checkpoint = paused.get("checkpoint_id")
    assert checkpoint, paused
    with client.stream("POST", f"/api/studio/approvals/{checkpoint}/approve") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
    done = next((f for f in frames if f.get("type") == "done"), None)
    assert done is not None, (
        f"no done frame after approval; kinds={[f.get('type') for f in frames]}"
    )
    return frames, done


def _tools_fired(frames: list[dict]) -> set[str]:
    return {f.get("name", "") for f in frames if f.get("type") == "tool"}


def _assert_grounded(done: dict, frames: list[dict], expected: set[str]) -> set[str]:
    fired = _tools_fired(frames)
    assert fired & expected, f"no tool from {expected} fired; fired={fired or 'NONE'}"
    assert done.get("succeeded") is True, f"run failed: {done}"
    assert str(done.get("output_text", "")).strip(), "empty final answer"
    return fired


# --------------------------------------------------------------------- utils
def test_calculator_returns_the_exact_value(client: TestClient, project: Path) -> None:
    path = _agent(project, ["utils"])
    frames, done = _run(
        client, path, "Use the calculator tool to compute 17*23. Reply with the number."
    )
    _assert_grounded(done, frames, {"calculator"})
    assert "391" in done["output_text"], done["output_text"]


def test_current_time_tool_fires(client: TestClient, project: Path) -> None:
    path = _agent(project, ["utils"])
    frames, done = _run(
        client, path, "Use the current_time tool and tell me today's date."
    )
    _assert_grounded(done, frames, {"current_time"})


# ---------------------------------------------------------------------- news
def test_news_pack_fetches_real_headlines(client: TestClient, project: Path) -> None:
    """THE reported failure: 'not able to fetch news'. A news tool must fire
    and the answer must come from it (non-empty, successful run)."""
    path = _agent(project, ["news"])
    frames, done = _run(
        client,
        path,
        "Use the news_search tool to find one recent headline about Nepal, then "
        "tell me the headline and its source.",
    )
    _assert_grounded(done, frames, {"news_search", "news_sources", "news_fetch"})
    # placeholders like "[News Article 1]" are the hallucination signature
    assert "[News Article" not in done["output_text"]


def test_news_search_never_dead_ends_on_natural_phrasing(
    client: TestClient, project: Path
) -> None:
    """The exact reported failure: 'use news tool to search about recent
    political news of Nepal' returned count=0 (the old ALL-words match) and the
    agent floundered. The tool must now return material for natural phrasing."""
    path = _agent(project, ["news"])
    frames, done = _run(
        client,
        path,
        "use the news_search tool to search about recent political news of "
        "Nepal, then summarize what you find.",
    )
    _assert_grounded(done, frames, {"news_search", "news_fetch", "news_sources"})
    tool = next(
        (
            f
            for f in frames
            if f.get("type") == "tool" and f.get("name") == "news_search"
        ),
        None,
    )
    if tool is not None and isinstance(tool.get("result"), str):
        # the search result must not be a dead end
        assert '"count": 0' not in tool["result"], tool["result"][:200]


# ----------------------------------------------------------------------- web
def test_web_pack_searches_the_open_web(client: TestClient, project: Path) -> None:
    path = _agent(project, ["web"])
    frames, done = _run(
        client,
        path,
        "Use the web_search tool to look up 'capital of Nepal' and answer from "
        "the results.",
    )
    _assert_grounded(done, frames, {"web_search", "web_fetch", "http_request"})


# -------------------------------------------------------------- data-sources
def test_wikipedia_tool_grounds_the_answer(client: TestClient, project: Path) -> None:
    path = _agent(project, ["data-sources"])
    frames, done = _run(
        client,
        path,
        "Use the wikipedia tool to get a summary of Mount Everest and give me "
        "one sentence from it.",
    )
    _assert_grounded(done, frames, {"wikipedia", "weather", "geocode"})


# --------------------------------------------------------------------- notes
def test_write_note_persists_to_the_shared_board(
    client: TestClient, project: Path
) -> None:
    path = _agent(project, ["notes"])
    frames, done = _run(
        client,
        path,
        "Use the write_note tool to save a note titled 'ToolCheck' whose content "
        "is 'verified'. Then confirm you saved it.",
    )
    _assert_grounded(done, frames, {"write_note", "list_notes", "read_note"})
    # out-of-band proof: the note exists in the SAME store the GUI reads
    notes = client.get("/api/studio/notes").json()
    titles = [n.get("title", "") for n in notes]
    assert any("toolcheck" in t.lower() for t in titles), titles


# --------------------------------------------------------------------- tasks
def test_add_task_persists_to_the_shared_board(
    client: TestClient, project: Path
) -> None:
    path = _agent(project, ["tasks"])
    frames, done = _run(
        client,
        path,
        "Use the add_task tool to add a task titled 'Water the plants'. Then "
        "confirm it was added.",
    )
    _assert_grounded(done, frames, {"add_task", "list_tasks"})
    tasks = client.get("/api/studio/tasks").json()
    titles = [t.get("title", "") for t in tasks]
    assert any("water the plants" in t.lower() for t in titles), titles


# -------------------------------------------------------------------- memory
def test_remember_persists_a_durable_memory(client: TestClient, project: Path) -> None:
    path = _agent(project, ["memory"])
    frames, done = _run(
        client,
        path,
        "Use the remember tool to store this fact: 'favorite tea is masala "
        "chiya'. Then confirm.",
    )
    _assert_grounded(done, frames, {"remember"})
    listing = client.get("/api/studio/memory").json()
    body = json.dumps(listing).lower()
    assert "masala" in body, "stored memory not found in the memory store"


# --------------------------------------------------------------------- files
def test_write_file_lands_in_the_sandbox(client: TestClient, project: Path) -> None:
    path = _agent(project, ["files"])
    # write_file mutates the filesystem, so it is approval-gated: approve the
    # pause and let the resumed run land the file.
    frames, done = _run_with_approval(
        client,
        path,
        "Use the write_file tool to create a file named toolcheck.txt containing "
        "exactly: HELLO. Then confirm.",
    )
    fired = _tools_fired(frames)
    assert fired & {"write_file"}, f"write_file never fired: {fired or 'NONE'}"
    # out-of-band proof: the file exists under the sandbox root on disk
    matches = list(project.rglob("toolcheck.txt"))
    assert matches, "toolcheck.txt not found under the project sandbox"
    assert "HELLO" in matches[0].read_text("utf-8")
