"""Core kernel: the shared base exception type for Himmy."""

from __future__ import annotations


class HimmyError(Exception):
    """Base class for all Himmy-raised errors.

    Kernel-specific error enums live in their own kernels; this is the single
    exception type callers can broadly catch for Himmy-originated failures.
    """
