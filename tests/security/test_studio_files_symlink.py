"""Regression: studio file-download TOCTOU on an intermediate path component.

Finding #14: the download endpoint validated the relative path per-component
(rejecting symlinks) and then opened the file with ``O_NOFOLLOW``. But
``O_NOFOLLOW`` only guards the *final* component, so an attacker who can write
inside the sandbox could swap an INTERMEDIATE directory for a symlink to (say)
``/etc`` in the window between the check and the open, and have the endpoint
serve ``/etc/passwd``.

These tests drive the in-process app via TestClient with the sandbox pinned to a
tmp dir (the same isolation the existing suite uses). The TOCTOU is reproduced
deterministically by monkeypatching ``_validate_rel_path`` so that the malicious
directory→symlink swap happens AFTER validation returns but BEFORE the open —
exactly the race the fix must close. Against the old single-``O_NOFOLLOW``-open
code these fail (the file outside the sandbox is served); with the segment-wise
``O_NOFOLLOW`` traversal they pass (the request is rejected).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.api.routers import studio_files

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


def _swap_dir_for_symlink(victim: Path, swap_target: Path) -> None:
    """Replace the real directory ``victim`` with a symlink to ``swap_target``."""
    if victim.is_dir() and not victim.is_symlink():
        for child in victim.iterdir():
            child.unlink()
        victim.rmdir()
        victim.symlink_to(swap_target, target_is_directory=True)


def _arm_intermediate_swap(
    monkeypatch: pytest.MonkeyPatch,
    sandbox: Path,
    inner_rel: str,
    swap_target: Path,
) -> None:
    """Swap ``inner_rel`` for a symlink in the check-to-USE window (right before open).

    Wraps the segment-wise opener so validation AND the existence/dir checks all
    pass against the real directory, then the directory is replaced by a symlink
    to ``swap_target`` in the instant before the file is opened — exactly the
    TOCTOU the fix must close. The real opener runs after the swap.
    """
    real_open = studio_files._open_nofollow_segments

    def racing_open(root_resolved: Path, rel: str) -> int:
        _swap_dir_for_symlink(sandbox / inner_rel, swap_target)
        return real_open(root_resolved, rel)

    monkeypatch.setattr(studio_files, "_open_nofollow_segments", racing_open)


def test_intermediate_symlink_to_outside_is_rejected(
    client: TestClient,
    sandbox: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Attacker-controlled location outside the sandbox holding a secret.
    secret_dir = tmp_path / "etc"
    secret_dir.mkdir()
    (secret_dir / "passwd").write_text("root:x:0:0:LEAKED-SECRET")

    # A legitimate-looking inner directory + file inside the sandbox so that
    # validation passes on "sub/passwd".
    inner = sandbox / "sub"
    inner.mkdir()
    (inner / "passwd").write_text("harmless placeholder")

    _arm_intermediate_swap(monkeypatch, sandbox, "sub", secret_dir)

    r = client.get(DOWNLOAD, params={"path": "sub/passwd"})
    assert r.status_code == 400, r.status_code
    assert "LEAKED-SECRET" not in r.text


def test_intermediate_symlink_inside_sandbox_also_rejected(
    client: TestClient,
    sandbox: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Even a retargeted-after-check link that points back INSIDE the sandbox is
    # refused: any symlinked intermediate component fails closed.
    other = sandbox / "other"
    other.mkdir()
    (other / "passwd").write_text("inside-but-redirected")

    inner = sandbox / "sub"
    inner.mkdir()
    (inner / "passwd").write_text("placeholder")

    _arm_intermediate_swap(monkeypatch, sandbox, "sub", other)

    r = client.get(DOWNLOAD, params={"path": "sub/passwd"})
    assert r.status_code == 400, r.status_code


def test_deep_intermediate_symlink_rejected(
    client: TestClient,
    sandbox: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The vulnerable component is a *non-leaf, non-first* segment (a/b/c/file):
    # confirms every intermediate is O_NOFOLLOW-guarded, not just the first.
    secret_dir = tmp_path / "loot"
    secret_dir.mkdir()
    (secret_dir / "file.txt").write_text("DEEP-LEAK")

    deep = sandbox / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "file.txt").write_text("placeholder")

    _arm_intermediate_swap(monkeypatch, sandbox, "a/b/c", secret_dir)

    r = client.get(DOWNLOAD, params={"path": "a/b/c/file.txt"})
    assert r.status_code == 400, r.status_code
    assert "DEEP-LEAK" not in r.text


def test_legitimate_nested_download_still_works(
    client: TestClient, sandbox: Path
) -> None:
    # Guardrail: the segment-wise opener must not break normal nested downloads.
    nested = sandbox / "reports" / "q1"
    nested.mkdir(parents=True)
    (nested / "answer.md").write_text("# fine\n42\n")

    r = client.get(DOWNLOAD, params={"path": "reports/q1/answer.md"})
    assert r.status_code == 200
    assert r.content == b"# fine\n42\n"
