"""Sandbox backend selection: subprocess (default), container (hardened), or off.

:func:`build_sandbox` picks the execution backend from a mode string so the ``code``
toolkit pack and any served deployment can choose their isolation posture by config
(``HIMMY_CODE_EXEC``). ``off`` returns a :class:`DisabledSandbox` that refuses to run —
the safe default we recommend for multi-tenant served deployments, where running
model-authored code on the bare subprocess backend is not an acceptable risk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from himmy.services.sandbox.models import SandboxResult

if TYPE_CHECKING:  # pragma: no cover - typing only
    from himmy.services.sandbox.base import Sandbox
    from himmy.services.sandbox.models import SandboxLimits

#: Recognized backend modes.
CODE_EXEC_MODES = ("off", "subprocess", "container")


class DisabledSandbox:
    """A sandbox that refuses every run (code execution disabled by policy)."""

    async def run_code(
        self,
        code: str,
        *,
        stdin: str | None = None,
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Return a structured refusal instead of executing anything."""
        return SandboxResult(
            ok=False,
            exit_code=None,
            stdout="",
            stderr=(
                "code execution is disabled by policy "
                "(set HIMMY_CODE_EXEC=subprocess|container to enable)"
            ),
        )


def build_sandbox(
    mode: str,
    *,
    limits: SandboxLimits | None = None,
    image: str = "python:3.12-slim",
    engine: str = "docker",
) -> Sandbox:
    """Build the configured sandbox backend.

    ``off`` → :class:`DisabledSandbox`; ``container`` →
    :class:`~himmy.services.sandbox.container_sandbox.ContainerSandbox` (hardened);
    anything else → the portable
    :class:`~himmy.services.sandbox.subprocess_sandbox.SubprocessSandbox` (default).
    """
    normalized = (mode or "subprocess").strip().lower()
    if normalized == "off":
        return DisabledSandbox()
    if normalized == "container":
        from himmy.services.sandbox.container_sandbox import ContainerSandbox

        return ContainerSandbox(image=image, engine=engine, limits=limits)
    from himmy.services.sandbox.subprocess_sandbox import SubprocessSandbox

    return SubprocessSandbox(limits=limits)


__all__ = ["build_sandbox", "DisabledSandbox", "CODE_EXEC_MODES"]
