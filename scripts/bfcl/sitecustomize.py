"""Interpreter-startup hook for the BFCL OpenRouter harness.

Python imports a module named ``sitecustomize`` automatically at startup if it
is found on ``sys.path``. We point the bfcl venv at this file by putting
``scripts/bfcl/`` on ``PYTHONPATH`` (see ``scripts/bfcl/run.sh`` /
``scripts/bfcl/README.md``). That guarantees our custom models are registered
into BFCL's ``MODEL_CONFIG_MAPPING`` BEFORE ``bfcl_eval.__main__`` reads it,
WITHOUT editing any site-packages file.

Responsibilities:
  1. Load ``OPENROUTER_API_KEY`` (and any other vars) from the himmy repo
     ``.env`` into the process environment if not already set. The secret value
     is NEVER printed.
  2. Register the RAW OpenRouter models (scripts/bfcl/register.py).
  3. Register the HIMMY OpenRouter models if scripts/bfcl/register_himmy.py
     exists (added by the HIMMY arm; optional so the RAW arm works standalone).

This module is intentionally fail-soft: if BFCL is not importable (e.g. running
outside the bfcl venv) it does nothing, so it cannot break unrelated Python.
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = three levels up from this file: scripts/bfcl/sitecustomize.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DOTENV = _REPO_ROOT / ".env"


def _load_dotenv_keys(path: Path) -> None:
    """Minimal .env loader. Only sets keys that are not already in os.environ.

    Avoids a hard dependency on python-dotenv at startup and never prints values.
    """
    if not path.exists():
        return
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        pass


def _bootstrap() -> None:
    _load_dotenv_keys(_DOTENV)

    # Only proceed if BFCL is importable (i.e. we are in the bfcl venv).
    try:
        import bfcl_eval.constants.model_config  # noqa: F401
    except Exception:
        return

    try:
        import register  # scripts/bfcl/register.py (on PYTHONPATH)

        register.register()
    except Exception as exc:  # pragma: no cover - surfaced, never swallowed silently
        # Do not crash the interpreter; surface so the operator sees it.
        print(f"[bfcl sitecustomize] RAW model registration failed: {exc!r}")

    # Optional: HIMMY arm registration, if present.
    try:
        import importlib.util

        himmy_reg = Path(__file__).resolve().parent / "register_himmy.py"
        if himmy_reg.exists():
            import register_himmy  # type: ignore

            register_himmy.register()
    except Exception as exc:  # pragma: no cover
        print(f"[bfcl sitecustomize] HIMMY model registration failed: {exc!r}")


_bootstrap()
