"""Core kernel: the shared base exception type for OpenSims."""

from __future__ import annotations


class OpenSimsError(Exception):
    """Base class for all OpenSims-raised errors.

    Kernel-specific error enums live in their own kernels; this is the single
    exception type callers can broadly catch for OpenSims-originated failures.
    """
