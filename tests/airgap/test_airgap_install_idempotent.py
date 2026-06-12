"""Idempotency tests for scripts/airgap_install.sh (step 4: install compose files).

The installer must be safe to re-run for an upgrade-in-place: refresh the bundled
compose files but NEVER clobber the operator's provisioned deploy/compose/.env or
deploy/compose/secrets/ (losing the KEK makes encrypted-at-rest data unreadable).

docker / sha256sum are stubbed on PATH so no real container engine or network is
touched; the test exercises the real shell script against a minimal fake bundle.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _REPO_ROOT / "scripts" / "airgap_install.sh"


def _stub(path: Path, name: str) -> None:
    """Write a no-op executable shim named ``name`` under ``path`` (always exit 0)."""
    p = path / name
    p.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_bundle(bundle: Path) -> None:
    """A minimal unpacked bundle the installer accepts (no models, one image)."""
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    (bundle / "SHA256SUMS").write_text("", encoding="utf-8")
    images = bundle / "images"
    images.mkdir()
    (images / "studio.tar.gz").write_bytes(b"\x00")  # satisfies images/*.tar.gz glob
    compose = bundle / "compose"
    compose.mkdir()
    (compose / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (compose / ".env.example").write_text("EXAMPLE=1\n", encoding="utf-8")


def _run_installer(bundle: Path, stub_bin: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{stub_bin}{os.pathsep}{env.get('PATH', '')}"
    return subprocess.run(
        ["sh", str(_SCRIPT)],
        cwd=str(bundle),
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def staged(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / "bundle"
    stub_bin = tmp_path / "bin"
    stub_bin.mkdir()
    for tool in ("docker", "sha256sum"):
        _stub(stub_bin, tool)
    _make_bundle(bundle)
    return bundle, stub_bin


def test_fresh_install_drops_compose(staged: tuple[Path, Path]) -> None:
    bundle, stub_bin = staged
    res = _run_installer(bundle, stub_bin)
    assert res.returncode == 0, res.stderr
    assert (bundle / "deploy" / "compose" / "docker-compose.yml").is_file()


def test_rerun_preserves_provisioned_env_and_secrets(
    staged: tuple[Path, Path],
) -> None:
    bundle, stub_bin = staged
    # First install.
    assert _run_installer(bundle, stub_bin).returncode == 0

    # Operator provisions real secrets the installer must never destroy.
    compose = bundle / "deploy" / "compose"
    (compose / ".env").write_text("POSTGRES_PASSWORD=supersecret\n", encoding="utf-8")
    secrets = compose / "secrets"
    secrets.mkdir(exist_ok=True)
    (secrets / "HIMMY_ENCRYPTION_KEY").write_text("the-only-kek\n", encoding="utf-8")

    # Re-run the installer (simulating an upgrade-in-place with a fresh bundle).
    (bundle / "compose" / "docker-compose.yml").write_text(
        "services: {updated: true}\n", encoding="utf-8"
    )
    res = _run_installer(bundle, stub_bin)
    assert res.returncode == 0, res.stderr

    # The operator's .env + KEK survive untouched...
    assert (compose / ".env").read_text() == "POSTGRES_PASSWORD=supersecret\n"
    assert (secrets / "HIMMY_ENCRYPTION_KEY").read_text() == "the-only-kek\n"
    # ...while the bundled compose file IS refreshed to the new bundle's content.
    assert "updated: true" in (compose / "docker-compose.yml").read_text()
    # ...and a backup of the prior install was taken.
    backups = list((bundle / "deploy").glob("compose.bak-*"))
    assert backups, "expected a deploy/compose.bak-<ts> backup on re-run"


def test_rerun_does_not_seed_env_over_existing(staged: tuple[Path, Path]) -> None:
    """An existing deploy/compose/.env is never overwritten by the bundle."""
    bundle, stub_bin = staged
    assert _run_installer(bundle, stub_bin).returncode == 0
    compose = bundle / "deploy" / "compose"
    (compose / ".env").write_text("MINE=1\n", encoding="utf-8")
    # Even if the bundle were to ship a .env, the existing one wins.
    shutil.copy(bundle / "compose" / ".env.example", bundle / "compose" / ".env")
    assert _run_installer(bundle, stub_bin).returncode == 0
    assert (compose / ".env").read_text() == "MINE=1\n"
