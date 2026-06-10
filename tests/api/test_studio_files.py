"""Studio files router: sandbox listing + traversal/symlink-safe downloads.

Drives the in-process FastAPI app via TestClient. The sandbox root is pinned to
a tmp dir through ``HIMMY_FS_ROOT`` (the same env knob the toolkit honors), so
every case is isolated and offline.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app

FILES = "/api/studio/files"
DOWNLOAD = "/api/studio/files/download"


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "sandbox"
    root.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_FS_ROOT", str(root))
    return root


@pytest.fixture
def client(sandbox: Path) -> TestClient:
    return TestClient(create_app())


def _write(root: Path, rel: str, content: str = "hello") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


# ---- listing -------------------------------------------------------------


def test_list_empty_sandbox(client: TestClient, sandbox: Path) -> None:
    r = client.get(FILES)
    assert r.status_code == 200
    body = r.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["truncated"] is False
    assert body["root"] == str(sandbox.resolve())


def test_list_missing_root_is_empty_not_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HIMMY_FS_ROOT", str(tmp_path / "does-not-exist"))
    r = TestClient(create_app()).get(FILES)
    assert r.status_code == 200
    assert r.json()["items"] == []


def test_list_files_shape_and_order(client: TestClient, sandbox: Path) -> None:
    old = _write(sandbox, "old.txt", "a")
    _write(sandbox, "reports/new.md", "# fresh")
    past = time.time() - 3600
    os.utime(old, (past, past))

    body = client.get(FILES).json()
    assert body["total"] == 2
    assert body["truncated"] is False
    paths = [i["path"] for i in body["items"]]
    assert paths == ["reports/new.md", "old.txt"]  # newest first
    newest = body["items"][0]
    assert newest["name"] == "new.md"
    assert newest["size"] == len("# fresh")
    assert "T" in newest["mtime"] and newest["mtime"].endswith("+00:00")


def test_list_skips_hidden_skipdirs_and_symlinks(
    client: TestClient, sandbox: Path, tmp_path: Path
) -> None:
    _write(sandbox, "visible.txt")
    _write(sandbox, ".env", "SECRET=1")
    _write(sandbox, ".himmy/approvals.db", "db")
    _write(sandbox, "node_modules/pkg/index.js", "x")
    outside = tmp_path / "outside.txt"
    outside.write_text("leak")
    (sandbox / "link.txt").symlink_to(outside)
    linkdir = tmp_path / "outdir"
    linkdir.mkdir()
    (linkdir / "inner.txt").write_text("leak")
    (sandbox / "linkdir").symlink_to(linkdir)

    body = client.get(FILES).json()
    assert [i["path"] for i in body["items"]] == ["visible.txt"]
    assert body["total"] == 1


def test_list_limit_bounds(client: TestClient, sandbox: Path) -> None:
    for i in range(3):
        _write(sandbox, f"f{i}.txt", str(i))
    body = client.get(f"{FILES}?limit=2").json()
    assert len(body["items"]) == 2
    assert body["total"] == 3
    assert body["truncated"] is True
    # out-of-range limits are rejected by validation, not clamped silently
    assert client.get(f"{FILES}?limit=0").status_code == 422
    assert client.get(f"{FILES}?limit=1001").status_code == 422


# ---- download ------------------------------------------------------------


def test_download_happy_path(client: TestClient, sandbox: Path) -> None:
    _write(sandbox, "reports/answer.md", "# the answer\n42\n")
    r = client.get(DOWNLOAD, params={"path": "reports/answer.md"})
    assert r.status_code == 200
    assert r.content == b"# the answer\n42\n"
    assert 'filename="answer.md"' in r.headers["content-disposition"]
    assert r.headers["content-length"] == str(len(r.content))


def test_download_known_extension_gets_text_type(
    client: TestClient, sandbox: Path
) -> None:
    _write(sandbox, "notes.txt", "plain")
    r = client.get(DOWNLOAD, params={"path": "notes.txt"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")


def test_download_unknown_extension_is_octet_stream(
    client: TestClient, sandbox: Path
) -> None:
    _write(sandbox, "blob.weirdext", "data")
    r = client.get(DOWNLOAD, params={"path": "blob.weirdext"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"


def test_download_missing_file_404(client: TestClient, sandbox: Path) -> None:
    assert client.get(DOWNLOAD, params={"path": "nope.txt"}).status_code == 404


def test_download_directory_404(client: TestClient, sandbox: Path) -> None:
    (sandbox / "adir").mkdir()
    assert client.get(DOWNLOAD, params={"path": "adir"}).status_code == 404


def test_download_rejects_absolute_path(client: TestClient, sandbox: Path) -> None:
    secret = sandbox.parent / "secret.txt"
    secret.write_text("top secret")
    r = client.get(DOWNLOAD, params={"path": str(secret)})
    assert r.status_code == 400
    assert "top secret" not in r.text


def test_download_rejects_traversal(client: TestClient, sandbox: Path) -> None:
    (sandbox.parent / "outside.txt").write_text("leak")
    for evil in ("../outside.txt", "a/../../outside.txt", "..", "~/x"):
        r = client.get(DOWNLOAD, params={"path": evil})
        assert r.status_code == 400, evil
        assert "leak" not in r.text


def test_download_rejects_symlink(client: TestClient, sandbox: Path) -> None:
    outside = sandbox.parent / "outside.txt"
    outside.write_text("leak")
    (sandbox / "alias.txt").symlink_to(outside)
    r = client.get(DOWNLOAD, params={"path": "alias.txt"})
    assert r.status_code == 400
    assert "leak" not in r.text


def test_download_rejects_symlinked_dir_component(
    client: TestClient, sandbox: Path
) -> None:
    real = sandbox / "real"
    real.mkdir()
    (real / "inner.txt").write_text("inside")
    (sandbox / "shortcut").symlink_to(real)
    # even a link pointing INSIDE the sandbox is refused (retarget-after-check)
    r = client.get(DOWNLOAD, params={"path": "shortcut/inner.txt"})
    assert r.status_code == 400


def test_download_rejects_hidden_components(client: TestClient, sandbox: Path) -> None:
    _write(sandbox, ".env", "SECRET=1")
    _write(sandbox, ".himmy/approvals.db", "db")
    assert client.get(DOWNLOAD, params={"path": ".env"}).status_code == 400
    assert (
        client.get(DOWNLOAD, params={"path": ".himmy/approvals.db"}).status_code == 400
    )


def test_download_input_bounds(client: TestClient, sandbox: Path) -> None:
    assert client.get(DOWNLOAD).status_code == 422  # path is required
    too_long = "a/" * 600 + "f.txt"
    assert client.get(DOWNLOAD, params={"path": too_long}).status_code == 422
    assert client.get(DOWNLOAD, params={"path": "  padded.txt "}).status_code == 400


def test_download_nul_byte_rejected(client: TestClient, sandbox: Path) -> None:
    assert client.get(DOWNLOAD, params={"path": "a\x00b"}).status_code == 400
