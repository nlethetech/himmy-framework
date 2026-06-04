"""Himmy services namespace.

Each service kernel lives in its own subpackage (inference, prompts, context,
storage, tools, knowledge, evaluation, observability). This package is intentionally
empty so subpackages can be imported independently and offline.
"""

from __future__ import annotations

__all__: list[str] = []
