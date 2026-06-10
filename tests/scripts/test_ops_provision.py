"""Tests for ``scripts/ops_provision.py`` (no docker; tmp dirs only)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load() -> ModuleType:
    path = _REPO_ROOT / "scripts" / "ops_provision.py"
    spec = importlib.util.spec_from_file_location("ops_provision", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ops = _load()


def test_provision_writes_env_and_secret(tmp_path: Path) -> None:
    rc = ops.main(["--dir", str(tmp_path)])
    assert rc == 0

    env = tmp_path / ".env"
    pg = tmp_path / "secrets" / "postgres_password"
    assert env.is_file()
    assert pg.is_file()

    pg_value = pg.read_text().strip()
    assert pg_value  # non-empty random
    env_text = env.read_text()
    # the .env's POSTGRES_PASSWORD must match the secret file (no divergence)
    assert f"POSTGRES_PASSWORD={pg_value}" in env_text
    assert "HIMMY_INTERNAL_API_KEY=" in env_text
    # api key line is non-empty
    api_line = next(
        ln for ln in env_text.splitlines() if ln.startswith("HIMMY_INTERNAL_API_KEY=")
    )
    assert len(api_line.split("=", 1)[1]) > 10


def test_provision_refuses_overwrite_without_force(tmp_path: Path) -> None:
    assert ops.main(["--dir", str(tmp_path)]) == 0
    with pytest.raises(SystemExit):
        ops.main(["--dir", str(tmp_path)])


def test_provision_force_rotates(tmp_path: Path) -> None:
    ops.main(["--dir", str(tmp_path)])
    first = (tmp_path / "secrets" / "postgres_password").read_text()
    assert ops.main(["--dir", str(tmp_path), "--force"]) == 0
    second = (tmp_path / "secrets" / "postgres_password").read_text()
    assert first != second  # a new random value


def test_provision_partial_state_does_not_rotate_secret(tmp_path: Path) -> None:
    """Pre-existing .env but absent secret (a partial prior run): a no-force rerun must
    refuse WITHOUT rotating the secret file, so .env and the secret never diverge."""
    # First, a clean provision, then delete only the secret to simulate partial state.
    assert ops.main(["--dir", str(tmp_path)]) == 0
    pg = tmp_path / "secrets" / "postgres_password"
    env = tmp_path / ".env"
    env_before = env.read_text()
    pg.unlink()  # secret missing, .env present → divergent partial state
    assert not pg.exists()

    # A no-force rerun must SystemExit and write nothing (no secret rotation).
    with pytest.raises(SystemExit):
        ops.main(["--dir", str(tmp_path)])
    assert not pg.exists()  # secret was NOT (re)written
    assert env.read_text() == env_before  # .env untouched


def test_provision_partial_state_secret_present_env_absent(tmp_path: Path) -> None:
    """The mirror case: secret present, .env absent. A no-force rerun must refuse
    before rewriting the secret (which would orphan it from any future .env)."""
    assert ops.main(["--dir", str(tmp_path)]) == 0
    pg = tmp_path / "secrets" / "postgres_password"
    env = tmp_path / ".env"
    pg_before = pg.read_text()
    env.unlink()  # .env missing, secret present

    with pytest.raises(SystemExit):
        ops.main(["--dir", str(tmp_path)])
    assert pg.read_text() == pg_before  # secret NOT rotated
    assert not env.exists()  # .env NOT written


def test_provision_encryption_key(tmp_path: Path) -> None:
    rc = ops.main(["--dir", str(tmp_path), "--encryption-key"])
    assert rc == 0
    enc = tmp_path / "secrets" / "HIMMY_ENCRYPTION_KEY"
    assert enc.is_file()
    assert enc.read_text().strip()
