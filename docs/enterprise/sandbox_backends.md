# Sandbox backends — code-execution isolation (WS2)

Himmy can run model-authored Python (`run_python` in the `code` toolkit pack) under three
backends, selected by `HIMMY_CODE_EXEC`. Pick by your threat model.

| `HIMMY_CODE_EXEC` | Backend | Boundary | Use it for |
|---|---|---|---|
| `off` | `DisabledSandbox` | n/a — refuses to run | **Recommended default for multi-tenant served deployments.** No code execution at all. |
| `subprocess` *(default)* | `SubprocessSandbox` | resource/fault only | Trusted, single-tenant, local/CLI use. **Not** a security boundary against hostile code. |
| `container` | `ContainerSandbox` | OS-level | Untrusted / model-authored code. Hardened container per run. |

`run_python` is **always approval-gated** regardless of backend (a human-in-the-loop
decision). The default stays `subprocess` for backward compatibility — but a served,
multi-tenant deployment should set `off` or `container`.

## What `ContainerSandbox` enforces (per run, by default)

- **`--network none`** — egress denied (the #1 exfiltration path). *(verified)*
- **`--read-only`** root filesystem + a small writable `/tmp` tmpfs. *(verified)*
- **`--cap-drop ALL`** + **`--security-opt no-new-privileges`**.
- **`--user 65534:65534`** (non-root), so an escape isn't immediately root. *(verified)*
- **`--pids-limit` / `--memory` / `--cpus`** — fork-bomb, OOM, and CPU containment.
- A **hard in-container wall-clock kill** (coreutils `timeout -s KILL`), backed by an
  outer watchdog that `docker rm -f`s an overrunning container. *(verified)*
- A throwaway, world-readable bind-mounted workdir (`/work:ro`) for input files only.

Config: `HIMMY_SANDBOX_IMAGE` (default `python:3.12-slim`, must be pre-pulled),
`HIMMY_SANDBOX_ENGINE` (`docker`/`podman`), `HIMMY_SANDBOX_TMP` (a host path the engine
shares — needed on Docker Desktop/macOS, where `$TMPDIR` isn't shared by default).

> These guarantees are exercised by `tests/sandbox/test_container_sandbox.py`, which runs
> live against a real engine (and skips cleanly when none is present).

## Stronger boundaries: gVisor / microVM (WS2.3)

For **hostile multi-tenant** workloads, a standard container shares the host kernel — a
kernel exploit is a host compromise. Run the same image under a stronger runtime by
passing `runtime=` to `ContainerSandbox` (wired to `docker run --runtime`):

- **gVisor** (`runtime="runsc"`) — a user-space kernel; syscalls are intercepted, so the
  host kernel surface is dramatically reduced. Install gVisor + register the `runsc`
  runtime with Docker, then `ContainerSandbox(runtime="runsc")`.
- **Kata Containers / Firecracker** (`runtime="kata-runtime"` / a Firecracker shim) — each
  run gets its own micro-VM with a separate guest kernel: the strongest practical boundary.

These are drop-in via the same `Sandbox` protocol; the only change is the configured
runtime, so the rest of the stack is unaffected.

## Per-tenant execution quotas

Code execution counts toward the same per-principal / per-tenant rate limits and quotas as
the rest of the BFF (WS3.2). Cap executions and CPU-seconds per tenant there; the sandbox
enforces the per-run envelope (cpu/mem/pids/timeout) described above.
