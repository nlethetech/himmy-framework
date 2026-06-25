"""Regression: the SHIPPED Studio bundle must carry the postMessage origin guard.

Proven bypass: the origin-check fix lived only in the TypeScript source
(``studio/src/components/GoogleCard.tsx``), but the minified artifact that
``himmy studio`` actually serves out of ``himmy/api/_studio_static/`` was never
rebuilt. The live app therefore kept serving the OLD handler::

    j => { j.data === "himmy-google-connected" && (...) }

with no ``e.origin`` check, so any cross-origin window could ``postMessage`` the
"connected" signal and spoof a successful Google OAuth connection.

This test inspects the real emitted JS bundle (not the source) and fails if the
served handler reacts to ``himmy-google-connected`` without first comparing the
event origin against the app's own origin. It is intentionally text-based on the
artifact because the bug was precisely a stale artifact: the source can be
correct while the shipped bundle is not.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# himmy/api/_studio_static/assets — the directory `himmy studio` serves.
_ASSETS_DIR = (
    Path(__file__).resolve().parents[2]
    / "himmy"
    / "api"
    / "_studio_static"
    / "assets"
)

_SIGNAL = "himmy-google-connected"
# Origin guard, as minified by Vite: `j.origin===window.location.origin` (the
# exact identifiers are mangled, so match the structural comparison instead).
_ORIGIN_GUARD = re.compile(
    r"\.origin\s*(===|!==)\s*\w+\.location\.origin"
    r"|\w+\.location\.origin\s*(===|!==)\s*\w+\.origin"
)


def _bundle_files() -> list[Path]:
    if not _ASSETS_DIR.is_dir():
        pytest.skip(f"studio bundle not present at {_ASSETS_DIR}")
    return sorted(_ASSETS_DIR.glob("*.js"))


def _handler_window(text: str) -> str:
    """Return the slice of minified JS immediately around the signal check."""
    idx = text.find(_SIGNAL)
    assert idx != -1
    # The origin guard must sit just before the data comparison in the same
    # handler; a generous-but-local window catches the minified `&&` chain
    # without matching unrelated origin checks elsewhere in the bundle.
    start = max(0, idx - 200)
    return text[start:idx]


def test_shipped_bundle_handler_checks_origin_before_signal() -> None:
    """The artifact that handles the OAuth signal must guard on origin first."""
    matched = False
    for js in _bundle_files():
        text = js.read_text(encoding="utf-8", errors="ignore")
        if _SIGNAL not in text:
            continue
        matched = True
        before = _handler_window(text)
        assert _ORIGIN_GUARD.search(before), (
            f"{js.name}: handler reacts to {_SIGNAL!r} without an origin guard "
            f"preceding it — stale/unrebuilt bundle still vulnerable to a "
            f"cross-origin postMessage spoof. Rebuild Studio (npm run build)."
        )
    assert matched, (
        f"no shipped JS bundle references {_SIGNAL!r}; the GoogleCard handler "
        f"appears to be missing from the build output entirely."
    )


def test_no_unguarded_signal_handler_anywhere_in_bundle() -> None:
    """No served JS may contain the old unguarded `j.data===signal` shape.

    Reproduces the exact stale-artifact bypass: the pre-fix minified handler was
    ``j.data==="himmy-google-connected"`` with NO origin term anywhere near it.
    """
    # Vulnerable minified shape: a `<id>.data===` comparison to the signal with
    # no `.origin` token in the immediately-preceding handler body.
    data_check = re.compile(r"\w+\.data\s*===\s*[\"']" + re.escape(_SIGNAL))
    for js in _bundle_files():
        text = js.read_text(encoding="utf-8", errors="ignore")
        for m in data_check.finditer(text):
            preceding = text[max(0, m.start() - 200) : m.start()]
            assert _ORIGIN_GUARD.search(preceding), (
                f"{js.name}: found `data==={_SIGNAL}` with no origin guard "
                f"before it — this is the unrebuilt vulnerable handler."
            )
