"""Sandbox kernel: the limits requested, the policy enforced, and the result."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SandboxLimits(BaseModel):
    """The resource envelope enforced around a sandboxed execution.

    ``cpu_seconds`` and ``memory_mb`` are enforced via POSIX ``setrlimit`` in the
    child (best-effort on platforms that reject a given limit); ``timeout_seconds``
    is a hard wall-clock kill; ``max_output_bytes`` bounds captured stdout/stderr;
    ``file_size_mb`` caps any single file the code writes. ``network`` is
    **advisory** — a bare subprocess cannot block sockets on every OS, so a
    Sandbox backend that can (container/namespace) should honor it and the default
    :class:`SubprocessSandbox` documents it as not enforced. ``allow_env`` is an
    allow-list of environment variables passed through to the child (everything
    else is stripped).
    """

    cpu_seconds: float = 5.0
    timeout_seconds: float = 10.0
    memory_mb: int = 256
    file_size_mb: int = 16
    max_output_bytes: int = 64 * 1024
    network: bool = False
    allow_env: list[str] = Field(default_factory=list)


class NetworkMode(str, Enum):
    """How a hardened sandbox is allowed to reach the network.

    * ``none`` — default-deny: the container is created with ``--network none``,
      so there is **no** interface at all and egress is impossible. This is the
      only posture a bare ``docker run`` can fully enforce by itself.
    * ``egress_proxy`` — a constrained allow-list: the container is attached to an
      operator-provisioned, otherwise-isolated Docker network whose only route off
      the host is a forward proxy. The sandbox injects ``HTTP_PROXY``/``HTTPS_PROXY``
      pointing at that proxy and a ``NO_PROXY`` covering loopback, so the *only*
      reachable destinations are the hosts the proxy itself allows. There is never
      a general bridge to the host network — even in this mode egress is mediated.

    A plain bridge ("just give it the internet") is intentionally **not** a value:
    the policy never grants general, unmediated network access to untrusted code.
    """

    NONE = "none"
    EGRESS_PROXY = "egress_proxy"


class SandboxPolicy(BaseModel):
    """The enterprise isolation contract a hardened sandbox must enforce per run.

    One object carries all five guarantees so a deployment chooses its posture by
    config, with safe defaults (deny network, ephemeral, capped, no host access,
    audited). The :class:`SandboxLimits` envelope (cpu/mem/pids/timeout/output) is
    nested; the remaining fields are the OS-level isolation knobs the container
    backend applies. Defaults are the *locked-down* posture — relaxing any of them
    is an explicit, reviewable change.
    """

    # --- resource envelope (cpu / mem / pids / timeout / output) ----------------
    limits: SandboxLimits = Field(default_factory=SandboxLimits)
    #: CPU cores the container may use (``docker run --cpus``).
    cpus: float = 1.0
    #: Max processes/threads inside the container (``--pids-limit``; fork-bomb cap).
    pids_limit: int = 128
    #: POSIX rlimits applied in the container as ``--ulimit name=soft[:hard]``.
    #: Defaults cap open files and forbid core dumps; ``nproc`` reinforces pids.
    ulimits: dict[str, str] = Field(
        default_factory=lambda: {"nofile": "256:512", "nproc": "128:128", "core": "0:0"}
    )

    # --- network (default DENY; optional mediated egress) -----------------------
    network: NetworkMode = NetworkMode.NONE
    #: For ``egress_proxy`` mode: the operator-provisioned, otherwise-isolated Docker
    #: network the container attaches to (its only route off-host is the proxy).
    egress_network: str | None = None
    #: For ``egress_proxy`` mode: the proxy URL injected as HTTP(S)_PROXY (e.g.
    #: ``http://egress-proxy:3128``). The proxy is what actually enforces the host
    #: allow-list; without it the container still has no usable route.
    egress_proxy_url: str | None = None
    #: Hosts/CIDRs the code may reach directly without the proxy (always loopback).
    egress_no_proxy: list[str] = Field(
        default_factory=lambda: ["127.0.0.1", "localhost"]
    )

    # --- ephemeral + no-host-access ---------------------------------------------
    #: Remove the container as soon as it exits (``--rm``) — nothing persists.
    auto_remove: bool = True
    #: Mount the root filesystem read-only (``--read-only``).
    read_only_rootfs: bool = True
    #: Run the writable workspace (``/sandbox``) on a tmpfs so it never touches a
    #: host disk and is gone when the container is removed. Size = file_size_mb.
    workspace_tmpfs: bool = True
    #: Drop every Linux capability (``--cap-drop ALL``).
    drop_all_capabilities: bool = True
    #: Forbid privilege escalation (``--security-opt no-new-privileges``).
    no_new_privileges: bool = True
    #: The non-root uid:gid the code runs as (``--user``). ``nobody:nogroup``.
    user: str = "65534:65534"

    # --- audit ------------------------------------------------------------------
    #: Record every execution to the event/audit spine. On by default; a hardened
    #: run that cannot be audited is a posture violation, not a silent allowance.
    audit: bool = True

    def with_limits(self, limits: SandboxLimits) -> SandboxPolicy:
        """Return a copy of this policy carrying ``limits`` (override the envelope)."""
        return self.model_copy(update={"limits": limits})


class SandboxResult(BaseModel):
    """The outcome of one sandboxed execution."""

    ok: bool
    exit_code: int | None
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_ms: float = 0.0
    truncated: bool = False
    limits: SandboxLimits = Field(default_factory=SandboxLimits)


__all__ = ["SandboxLimits", "SandboxPolicy", "NetworkMode", "SandboxResult"]
