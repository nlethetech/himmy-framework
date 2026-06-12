"""Packaging contract for the Himmy Studio GUI.

Closes the "a wheel can silently ship without the GUI" gap. The CI ``Build Wheel``
workflow builds the SPA, builds the wheel, and asserts the built shell is inside
it; these tests pin the two pieces of config that gate makes load-bearing so they
can't silently regress between CI runs:

* the ``[tool.setuptools.package-data]`` globs that pull ``_studio_static`` into
  the wheel (a fast, always-on contract — no build tooling required), and
* the end-to-end "a built wheel actually contains ``index.html``" assertion (the
  same check the CI gate makes), guarded to skip cleanly when the ``build``
  package or a built SPA is absent so the default suite is never affected.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from himmy.api.app import STUDIO_STATIC_DIR, studio_is_built

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_PYPROJECT = _REPO_ROOT / "pyproject.toml"

# The single artifact whose presence proves the GUI shipped: `himmy studio` and
# `studio_is_built()` both key off this exact path inside the package.
_SHELL_RELPATH = "himmy/api/_studio_static/index.html"


def _package_data() -> list[str]:
    """The ``[tool.setuptools.package-data]`` globs declared for the ``himmy`` pkg."""
    with _PYPROJECT.open("rb") as fh:
        data = tomllib.load(fh)
    return data["tool"]["setuptools"]["package-data"]["himmy"]


# --------------------------------------------------------------------------- #
# Static config contract — always runs, no build tooling needed.
# --------------------------------------------------------------------------- #
def test_package_data_declares_studio_static() -> None:
    """pyproject must glob the built SPA into the wheel, or `python -m build`
    produces an importable wheel whose `himmy studio` can't serve the GUI."""
    globs = _package_data()
    # index.html + hashed assets + the bundled fonts must all be packaged; a
    # wheel with only one of these still leaves the SPA broken at runtime.
    assert "api/_studio_static/*" in globs
    assert "api/_studio_static/assets/*" in globs
    assert "api/_studio_static/fonts/*" in globs


def test_studio_is_built_keys_off_index_html() -> None:
    """The build-detection predicate is anchored to the exact packaged path the
    wheel assertion checks — so CI and runtime can never disagree about whether
    the GUI is present."""
    assert STUDIO_STATIC_DIR == _REPO_ROOT / "himmy" / "api" / "_studio_static"
    # The relpath the CI gate asserts is precisely the file studio_is_built reads.
    assert (STUDIO_STATIC_DIR / "index.html") == _REPO_ROOT / _SHELL_RELPATH


# --------------------------------------------------------------------------- #
# End-to-end wheel contract — the same assertion the CI gate makes.
# Skips cleanly when the SPA isn't built or the `build` frontend is absent, so
# the default offline suite is unaffected; it runs for real wherever both exist
# (CI's Build Wheel job, and locally after `cd studio && npm run build`).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not studio_is_built(),
    reason="Studio SPA not built (run `cd studio && npm run build`)",
)
def test_built_wheel_contains_studio_gui(tmp_path: Path) -> None:
    """A wheel built from this checkout must actually contain the Studio shell
    and its hashed assets — the gap the CI build gate exists to catch."""
    pytest.importorskip("build", reason="the `build` package is not installed")

    out = tmp_path / "dist"
    env = {**os.environ, "PIP_NO_BUILD_ISOLATION": "0"}
    proc = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out)],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"`python -m build` failed:\n{proc.stderr}"

    wheels = glob.glob(str(out / "*.whl"))
    assert len(wheels) == 1, f"expected one wheel, found {wheels!r}"
    names = set(zipfile.ZipFile(wheels[0]).namelist())

    assert _SHELL_RELPATH in names, (
        f"built wheel is MISSING {_SHELL_RELPATH} — the Studio GUI was not "
        f"packaged. _studio_static entries: "
        f"{sorted(n for n in names if '_studio_static' in n)}"
    )
    assets = [n for n in names if n.startswith("himmy/api/_studio_static/assets/")]
    assert assets, "wheel ships index.html but no built assets — SPA build incomplete"
