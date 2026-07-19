"""Per-workspace admission + execution-concurrency gate (T0.4).

Extracted from :mod:`himmy.application.services` so the per-tenant back-pressure
state (outstanding-run counters + lazily-bound execution semaphores) lives in one
cohesive collaborator that :class:`~himmy.application.services.RunAppService` holds
and delegates to. Behaviour is BYTE-IDENTICAL to the former inline implementation:

- single-threaded read-modify-write in the event loop (so admit/release is atomic),
- semaphores created lazily on first use so they bind the running loop,
- ``max_outstanding == 0`` disables the outstanding cap entirely.
"""

from __future__ import annotations

import asyncio


class WorkspaceRunQuotaExceeded(Exception):
    """A workspace exceeded its outstanding-run cap (T0.4 — surfaces as HTTP 429).

    Raised by :meth:`RunAppService.create_run` when a workspace already has its
    maximum number of in-flight (created-but-not-terminal) background runs. Carries
    the workspace + the cap so the router can return an informative 429.
    """

    def __init__(self, workspace_id: str, *, cap: int, outstanding: int) -> None:
        """Record the workspace and the cap it hit for the API error surface."""
        self.workspace_id = workspace_id
        self.cap = cap
        self.outstanding = outstanding
        super().__init__(
            f"workspace {workspace_id!r} has {outstanding} outstanding run(s); "
            f"the per-workspace cap is {cap}. Retry once in-flight runs complete."
        )


class WorkspaceQuota:
    """Owns the per-workspace outstanding-run counters + execution semaphores (T0.4).

    ``concurrency`` bounds how many of one workspace's runs execute the runtime at
    once; ``max_outstanding`` caps how many created-but-not-terminal runs a workspace
    may hold before admission rejects with :class:`WorkspaceRunQuotaExceeded`. Both are
    clamped on assignment to the same safe floors the former inline code used.
    """

    def __init__(self, *, concurrency: int, max_outstanding: int) -> None:
        self._concurrency = max(1, int(concurrency))
        self._max_outstanding = max(0, int(max_outstanding))
        # Keyed on workspace_id, created lazily so an unused workspace costs nothing.
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._outstanding: dict[str, int] = {}

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @property
    def max_outstanding(self) -> int:
        return self._max_outstanding

    @max_outstanding.setter
    def max_outstanding(self, value: int) -> None:
        self._max_outstanding = max(0, int(value))

    @property
    def outstanding_counts(self) -> dict[str, int]:
        """The live per-workspace outstanding-run counter dict (mutated in place)."""
        return self._outstanding

    def admit(self, workspace_id: str) -> None:
        """Reserve one outstanding-run slot for ``workspace_id`` or reject (T0.4).

        Raises :class:`WorkspaceRunQuotaExceeded` when the workspace already holds
        ``max_outstanding`` in-flight runs. ``0`` disables the cap. Runs single-threaded
        in the event loop, so the read-modify-write is atomic.
        """
        if self._max_outstanding <= 0:
            return
        current = self._outstanding.get(workspace_id, 0)
        if current >= self._max_outstanding:
            raise WorkspaceRunQuotaExceeded(
                workspace_id,
                cap=self._max_outstanding,
                outstanding=current,
            )
        self._outstanding[workspace_id] = current + 1

    def release(self, workspace_id: str) -> None:
        """Release one outstanding-run slot for ``workspace_id`` (floors at 0)."""
        if self._max_outstanding <= 0:
            return
        current = self._outstanding.get(workspace_id, 0)
        if current <= 1:
            self._outstanding.pop(workspace_id, None)
        else:
            self._outstanding[workspace_id] = current - 1

    def semaphore(self, workspace_id: str) -> asyncio.Semaphore:
        """Lazily get/create the per-workspace execution semaphore (T0.4).

        Created on first use inside the running loop so it binds the correct event
        loop (a semaphore created at construction could bind the wrong loop under
        ``asyncio.run`` test harnesses).
        """
        sem = self._semaphores.get(workspace_id)
        if sem is None:
            sem = asyncio.Semaphore(self._concurrency)
            self._semaphores[workspace_id] = sem
        return sem

    def outstanding(self, workspace_id: str) -> int:
        """Return the current count of in-flight runs for a workspace (introspection)."""
        return self._outstanding.get(workspace_id, 0)
