# Sandbox backends — code-execution isolation (WS2)

> **See [`sandbox.md`](./sandbox.md) for the full operator guide** — the security ladder,
> which backend per deployment tier, and how to deploy/verify gVisor & Firecracker on a
> Linux+KVM host. This page is the concise threat-model reference.

Himmy can run model-authored Python (`run_python` in the `code` toolkit pack) under five
backends, selected by `HIMMY_CODE_EXEC`. Pick by your threat model.

| `HIMMY_CODE_EXEC` | Backend | Boundary | Use it for |
|---|---|---|---|
| `off` | `DisabledSandbox` | n/a — refuses to run | **Recommended default for multi-tenant served deployments.** No code execution at all. |
| `subprocess` *(default)* | `SubprocessSandbox` | resource/fault only | Trusted, single-tenant, local/CLI use. **Not** a security boundary against hostile code. |
| `container` | `ContainerSandbox` | OS-level, shared host kernel | Untrusted / model-authored code, single tenant. Hardened container per run. |
| `gvisor` | `GVisorSandbox` | user-space kernel (`runsc`) | Untrusted / multi-tenant. **Linux only.** |
| `firecracker` | `FirecrackerSandbox` | per-run KVM microVM | Hostile / multi-tenant, strongest isolation. **Linux + KVM only.** |

`run_python` is **always approval-gated** regardless of backend (a human-in-the-loop
decision). The default stays `subprocess` for backward compatibility — but a served,
multi-tenant deployment should set `off` or `container`. In a **server context** the
`subprocess` backend is **refused outright** (fail-closed; see below), so a server can
never silently run untrusted code unsandboxed.

## The five guarantees `ContainerSandbox` enforces (per run, by default)

Driven by a `SandboxPolicy` with locked-down defaults — relaxing any field is an
explicit, reviewable change.

1. **Network — default DENY** (`--network none`): no interface, egress impossible (the
   #1 exfiltration path). An optional *mediated* allow-list
   (`HIMMY_SANDBOX_NETWORK=egress_proxy`) attaches the container to an
   operator-provisioned isolated network and injects `HTTP(S)_PROXY` pointing at a
   forward proxy that enforces the host allow-list — **never a general bridge**. *(verified)*
2. **Ephemeral** (`--rm` + a fresh container + a tmpfs workspace): nothing the code
   writes survives, no two runs share state. *(verified)*
3. **Resource limits**: `--memory` / `--memory-swap`, `--cpus`, `--pids-limit`,
   `--ulimit` (open files / nproc / no core dumps), plus a **hard in-container
   wall-clock kill** (coreutils `timeout -s KILL`) backed by an outer watchdog that
   `docker rm -f`s an overrunning container. *(verified)*
4. **No host access**: `--read-only` rootfs, `--cap-drop ALL`,
   `--security-opt no-new-privileges`, a non-root `--user 65534:65534`, and **no host
   bind mount** — the input bundle (code + files) is streamed **out-of-band on the
   container's stdin** (length-prefixed) and written to the in-container tmpfs, so no host
   path is ever shared *and* large inputs never hit the kernel's argv size ceiling. Total
   input is bounded by the `file_size_mb` workspace envelope (default 16 MB); an oversized
   bundle is refused fast with a clear `HimmyError` naming the cap. *(verified)*
5. **Logged**: every execution is recorded through the event/audit spine. *(verified)*

Config: `HIMMY_SANDBOX_IMAGE` (default `python:3.12-slim`, must be pre-pulled),
`HIMMY_SANDBOX_ENGINE` (`docker`/`podman`), `HIMMY_SANDBOX_NETWORK`
(`none`/`egress_proxy`) with `HIMMY_SANDBOX_EGRESS_NETWORK` +
`HIMMY_SANDBOX_EGRESS_PROXY` for the mediated allow-list. Because there is no bind
mount, **no host path needs sharing** — this runs unmodified on Docker Desktop/macOS.

## Fail-closed: a server never runs the unsandboxed backend

`SubprocessSandbox` is not a security boundary against hostile code. `build_sandbox(...)`
therefore **raises** for the `subprocess` backend when `server_context=True` (set by
every served entrypoint) or when `HIMMY_REQUIRE_SANDBOX=1`. A served deployment must run
`container` (hardened) or `off` (disabled) — the same fail-closed posture as the API's
non-loopback/auth checks. The dev/CLI default is unchanged.

> These guarantees are exercised by `tests/sandbox/test_container_sandbox.py`, which runs
> live against a real engine (and skips cleanly when none is present).

## Stronger boundaries: gVisor / Firecracker (WS2.3)

For **hostile multi-tenant** workloads, a standard container shares the host kernel — a
kernel exploit is a host compromise. These are now **first-class backends** selected by
`HIMMY_CODE_EXEC` (not just a `runtime=` knob), each enforcing the *same five-guarantee
`SandboxPolicy`* as `container`:

- **`gvisor`** → `GVisorSandbox` — the hardened container run under gVisor's `runsc`
  user-space kernel (`docker run --runtime=runsc`); syscalls are intercepted, so the host
  kernel surface shrinks ~100×. **Linux only.**
- **`firecracker`** → `FirecrackerSandbox` — a throwaway KVM microVM per run via
  firecracker-containerd (the `aws.firecracker` runtime): a separate guest kernel per run,
  the strongest practical boundary. **Linux + KVM only.**

Both are drop-in via the same `Sandbox` protocol — switching is a config change. They are
**not executable on macOS**; `build_sandbox` capability-detects them at startup when
`HIMMY_SANDBOX_VERIFY_RUNTIME=1`. Deploy/verify steps and the per-tier recommendation are
in [`sandbox.md`](./sandbox.md).

## Per-tenant execution quotas

Code execution counts toward the same per-principal / per-tenant rate limits and quotas as
the rest of the BFF (WS3.2). Cap executions and CPU-seconds per tenant there; the sandbox
enforces the per-run envelope (cpu/mem/pids/timeout) described above.
