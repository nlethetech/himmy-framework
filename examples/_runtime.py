"""Examples bootstrap: make the repo importable, then re-export the package builder.

The full offline-first stack builder now lives in the package as
:func:`himmy.build_runtime` (see :mod:`himmy.runtime.builder`). This shim
exists only so ``python examples/0N_*.py`` runs WITHOUT a prior ``pip install -e .``
by putting the repo root on ``sys.path``, then re-exports the builder so the
example scripts can ``from _runtime import build_runtime``.

New code should prefer ``from himmy import build_runtime`` directly.
"""

from __future__ import annotations

import os
import sys

# Examples-only convenience: make the repo root importable so the scripts run as
# plain ``python examples/0N_*.py`` even without installing the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from himmy.runtime.builder import (  # noqa: E402 - after the sys.path bootstrap
    build_inference,
    build_runtime,
    build_storage,
)

__all__ = ["build_runtime", "build_storage", "build_inference"]
