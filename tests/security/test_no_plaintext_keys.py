"""Regression test for finding #6: no plaintext OpenRouter key on disk.

Greps the repo working tree (excluding ``.git`` and any ``.venv``) and asserts
that no tracked-or-untracked file embeds a real ``sk-or-v1-...`` literal. Fails
against the old behavior (key in ``.env`` + the three ``scripts/bfcl/workdir*/.env``
files) and passes once the secret is scrubbed.

Also exercises the harness guard (``scripts/bfcl/env_guard.py``) to prove it
refuses to write a real key into a workdir ``.env``.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

# Real OpenRouter secret-key shape.
_KEY_RE = re.compile(rb"sk-or-v1-[A-Za-z0-9]{8,}")

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Directories that are not part of the source tree we audit (vendored deps,
# build artifacts, vcs internals, byte-compiled caches).
_SKIP_DIRS = {".git", ".venv", "node_modules", "build", "dist", "__pycache__"}


# This test file legitimately contains fake ``sk-or-v1-...`` fixtures, so it is
# excluded from its own scan.
_SELF = Path(__file__).resolve()


def _iter_candidate_files():
    for path in _REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.resolve() == _SELF:
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(_REPO_ROOT).parts):
            continue
        yield path


def test_no_plaintext_openrouter_key_in_working_tree():
    offenders: list[str] = []
    for path in _iter_candidate_files():
        try:
            data = path.read_bytes()
        except OSError:
            continue
        if _KEY_RE.search(data):
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        "Plaintext OpenRouter API key (sk-or-v1-...) found on disk in: "
        + ", ".join(sorted(offenders))
    )


def _load_env_guard():
    guard_path = _REPO_ROOT / "scripts" / "bfcl" / "env_guard.py"
    spec = importlib.util.spec_from_file_location("bfcl_env_guard", guard_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_harness_reads_key_from_env(monkeypatch):
    guard = _load_env_guard()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    # No env key and (by default) no secret store entry -> must refuse, not crash.
    with pytest.raises(RuntimeError):
        guard.read_openrouter_key()
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-fakefakefake0000")
    assert guard.read_openrouter_key() == "sk-or-v1-fakefakefake0000"


def test_harness_refuses_to_write_real_key(tmp_path):
    guard = _load_env_guard()
    # Placeholder write must never contain a real key.
    env_path = guard.write_workdir_env(
        tmp_path / "wd", extra={"OPENROUTER_MODEL": "openai/gpt-4o-mini"}
    )
    assert not _KEY_RE.search(env_path.read_bytes())
    # Attempting to smuggle a real key in must be refused.
    with pytest.raises(RuntimeError):
        guard.write_workdir_env(
            tmp_path / "wd2",
            extra={"OPENROUTER_API_KEY": "sk-or-v1-deadbeefdeadbeef"},
        )
