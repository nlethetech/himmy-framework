# Sandbox backends — running model-authored code safely

Himmy can execute model-authored Python (the `run_python` tool in the `code` toolkit
pack). Because that code is, in the worst case, **hostile** — a prompt-injected or
adversarial model trying to exfiltrate data, escape the host, or burn resources — the
sandbox is a real security control, not a convenience wrapper. You pick how strong a
boundary you want with one config value, `HIMMY_CODE_EXEC`.

This page explains the four backends in plain English, the **security ladder** they form,
**which to use per deployment tier**, and the **Linux + KVM requirement** for the two
strongest ones — plus how to deploy and verify them on a real host.

> `run_python` is **always approval-gated** regardless of backend (running code is a
> human-in-the-loop decision). The backend just decides *how isolated* an approved run is.

## The four backends, plain English

| `HIMMY_CODE_EXEC` | What it is | The boundary | OS |
|---|---|---|---|
| `off` | Refuses to run any code | n/a — nothing executes | any |
| `subprocess` *(default)* | A child Python process with rlimits, a wall-clock kill, a throwaway dir, and a stripped env | **Resources & faults only.** NOT a security boundary against hostile code. | any |
| `container` | A hardened, throwaway container per run | **OS-level**, but the workload **shares the host kernel** | Linux/macOS w/ Docker/Podman |
| `gvisor` | The same hardened container, run under gVisor's `runsc` runtime | **A user-space kernel** (the Sentry) sits between the code and the host; syscalls are intercepted, so the host-kernel surface shrinks ~100× | **Linux only** |
| `firecracker` | A throwaway **microVM** per run via firecracker-containerd | **A separate guest kernel per run**, hardware-isolated by KVM — the strongest practical boundary | **Linux + KVM only** |

All four are interchangeable behind the same `Sandbox` protocol and (for the three
hardened ones) the same five-guarantee `SandboxPolicy` — so switching is a config change,
nothing in the rest of the stack moves.

### What "subprocess" actually protects against (and what it doesn't)

`subprocess` is genuine defense-in-depth for *trusted* code: it stops runaway CPU/RAM,
fork bombs, accidental huge writes, and env/secret leakage. It does **not** stop a
determined adversary — a plain process can still attempt network I/O and read
world-readable files, and `RLIMIT_AS` is silently ignored on some platforms (macOS). Use
it only for code you trust, on a single tenant, locally. A **served deployment refuses it**
(see "Fail-closed posture" below).

## The security ladder

Weakest → strongest, each rung a strictly bigger wall around the same untrusted code:

```
subprocess        container            gvisor                  firecracker
(no boundary)  →  (shared host kernel) → (user-space kernel) →  (per-run microVM,
                                          ~100× less host-                separate guest
                                          kernel surface)                 kernel, KVM-isolated)
```

* **container** removes file/network/host access and caps resources, but a **host-kernel
  exploit** from inside the container is a host compromise — one shared kernel.
* **gvisor** keeps the same container ergonomics but interposes a user-space kernel: the
  workload's syscalls are serviced in user space, so it only ever touches a tiny, hardened
  slice of the real kernel. A kernel-bug exploit has almost nothing to aim at.
* **firecracker** gives each run its **own kernel** inside a KVM microVM. Even a full
  guest-kernel compromise is contained in the VM; the host kernel is never directly
  reachable. This is what multi-tenant code-execution services (AWS Lambda, etc.) use.

## The five guarantees (all three hardened backends, every run)

Driven by a `SandboxPolicy` with **locked-down defaults** — relaxing any field is an
explicit, reviewable change. gVisor and Firecracker enforce the *identical* policy as
`container`; only the boundary underneath changes.

1. **Network — default DENY** (`--network none`): no interface at all, egress impossible
   (the #1 exfiltration path). An optional *mediated* allow-list
   (`HIMMY_SANDBOX_NETWORK=egress_proxy`) attaches to an operator-provisioned isolated
   network and forces all egress through a forward proxy that enforces a host allow-list —
   **never a general bridge**.
2. **Ephemeral** (`--rm` + a fresh container/VM + a tmpfs workspace): nothing the code
   writes survives, and no two runs share state.
3. **Resource limits** (`--memory` / `--cpus` / `--pids-limit` / `--ulimit`, which also
   *size the microVM* under Firecracker) plus a **hard in-guest wall-clock kill** (coreutils
   `timeout -s KILL`), backed by an outer watchdog that force-removes an overrunning
   container/VM.
4. **No host access**: `--read-only` rootfs, `--cap-drop ALL`,
   `--security-opt no-new-privileges`, a non-root `--user 65534:65534`, and **no host bind
   mount** — input code+files are streamed **out-of-band on stdin** (length-prefixed) and
   written to the in-guest tmpfs, so no host path is ever shared. Total input is bounded by
   the `file_size_mb` workspace; an oversized bundle is refused fast with a clear error.
5. **Logged**: every execution is recorded through the event/audit spine, labelled with
   the real backend (`container` / `gvisor` / `firecracker`) so an auditor can tell which
   boundary a given run used.

## Which backend per tier

| Deployment tier | Recommended `HIMMY_CODE_EXEC` | Why |
|---|---|---|
| **OSS / self-host, single tenant** | `container` | Strong, portable, runs on any Docker/Podman host (incl. Docker Desktop on macOS). The right default once you enable code execution. |
| **Local dev / CLI, trusted code** | `subprocess` (default) | Zero infra; resource/fault isolation is enough for code you wrote. |
| **Hosted / multi-tenant SaaS** | `gvisor` or `firecracker` | Untrusted tenant code on shared hardware: you must not let one tenant's kernel exploit reach the host or another tenant. gVisor for density, Firecracker for the hardest isolation. |
| **Enterprise in-VPC, untrusted/regulated** | `gvisor` or `firecracker` | Same reasoning; pick per your kernel-isolation and compliance bar. Firecracker's per-run guest kernel is the strongest story for an auditor. |
| **Any tier that doesn't need code exec** | `off` | The safest setting of all — no execution surface. |

Rule of thumb: **trusted + single tenant → container; untrusted or multi-tenant → gVisor
or Firecracker; unsure → off.**

## Fail-closed posture (a server never runs the unsandboxed backend)

`subprocess` is not a security boundary, so `build_sandbox(...)` **raises** for it when
`server_context=True` (set by every served entrypoint) or `HIMMY_REQUIRE_SANDBOX=1`. A
served deployment must run one of the hardened backends (`container` / `gvisor` /
`firecracker`) or disable execution with `off`. This is the same fail-closed pattern as
the API's non-loopback/auth checks — the dev/CLI default is unchanged.

## Linux + KVM requirement (gVisor & Firecracker)

> **gVisor (`runsc`) and Firecracker are Linux-only and CANNOT run on macOS.** Firecracker
> additionally requires hardware virtualization via `/dev/kvm`. On a macOS developer box
> both backends report themselves **unavailable** by design. They are implemented and
> contract-tested in full (the exact runtime command, isolation flags, network-deny,
> ephemeral teardown, and resource limits are unit-asserted in
> `tests/sandbox/test_gvisor_sandbox.py` and `tests/sandbox/test_firecracker_sandbox.py`),
> but their **live behaviour must be verified on a Linux+KVM host.**

`build_sandbox(...)` **capability-detects** the chosen runtime when
`HIMMY_SANDBOX_VERIFY_RUNTIME=1` (recommended in production): if `gvisor` is selected but
`runsc` is not installed/registered, or `firecracker` is selected without `/dev/kvm` +
the `firecracker` binary + the containerd driver, startup fails with a clear, actionable
`HimmyError` — never an opaque error on the first run.

### Deploy + verify gVisor on a Linux host

```bash
# 1. Install gVisor's runsc runtime (see gvisor.dev/docs/user_guide/install).
#    Then register it with Docker in /etc/docker/daemon.json:
#      { "runtimes": { "runsc": { "path": "/usr/local/bin/runsc" } } }
sudo systemctl restart docker

# 2. Confirm the daemon sees the runtime.
docker info --format '{{json .Runtimes}}'      # must contain "runsc"

# 3. Point Himmy at it and turn on the startup capability probe.
export HIMMY_CODE_EXEC=gvisor
export HIMMY_SANDBOX_VERIFY_RUNTIME=1

# 4. Verify the live boundary (these run only on a Linux+gVisor host):
HIMMY_RUN_GVISOR_LIVE=1 pytest tests/sandbox/test_gvisor_live.py -q
```

A quick manual smoke test (the user-space kernel reports a gVisor-specific uname):

```bash
docker run --rm --runtime=runsc --network none python:3.12-slim \
  python -c "import os;print(os.uname().release)"   # gVisor reports its own kernel string
```

### Deploy + verify Firecracker on a Linux+KVM host

```bash
# 0. Confirm hardware virtualization is available.
ls -l /dev/kvm                      # must exist; user must be in the 'kvm' group

# 1. Install firecracker-containerd (github.com/firecracker-microvm/firecracker-containerd):
#    the 'firecracker' binary, the dedicated containerd, the aws.firecracker runtime,
#    and a guest kernel + rootfs image. Start its containerd on
#    /run/firecracker-containerd/containerd.sock.

# 2. Confirm the runtime + a microVM boot end-to-end.
sudo nerdctl --address /run/firecracker-containerd/containerd.sock \
  --namespace firecracker-containerd run --rm --runtime aws.firecracker \
  --network none python:3.12-slim python -c "print('hello from a microVM')"

# 3. Point Himmy at it.
export HIMMY_CODE_EXEC=firecracker
export HIMMY_SANDBOX_VERIFY_RUNTIME=1
# Override the socket/namespace only if you deviated from the defaults
# (/run/firecracker-containerd/containerd.sock, namespace firecracker-containerd).
```

The Firecracker backend drives firecracker-containerd through the docker-compatible
`nerdctl` CLI so the **same hardened OCI flag set** (network-deny, `--memory`, `--cpus`,
`--pids-limit`, `--read-only`, `--cap-drop ALL`, `--user`, tmpfs workspace, `--rm`) maps
straight onto a microVM. Jailer confinement, cgroups, and seccomp are applied by the
firecracker-containerd runtime; the per-run vCPU/memory come from `SandboxLimits`.

## Configuration reference

| Env var | Default | Meaning |
|---|---|---|
| `HIMMY_CODE_EXEC` | `subprocess` | `off` / `subprocess` / `container` / `gvisor` / `firecracker` |
| `HIMMY_SANDBOX_IMAGE` | `python:3.12-slim` | OCI image (must be pre-pulled; needs `python` + coreutils `timeout`) |
| `HIMMY_SANDBOX_ENGINE` | `docker` | container engine for `container`/`gvisor` (`docker`/`podman`) |
| `HIMMY_SANDBOX_NETWORK` | `none` | `none` (deny) or `egress_proxy` (mediated allow-list) |
| `HIMMY_SANDBOX_EGRESS_NETWORK` | — | for `egress_proxy`: the isolated network to attach to |
| `HIMMY_SANDBOX_EGRESS_PROXY` | — | for `egress_proxy`: the forward proxy URL injected as HTTP(S)_PROXY |
| `HIMMY_SANDBOX_VERIFY_RUNTIME` | off | probe that the chosen hardened runtime is present at **startup** (recommended in prod) |
| `HIMMY_REQUIRE_SANDBOX` | off | opt a non-server caller into the fail-closed posture (refuse `subprocess`) |

Per-tenant execution quotas (cap executions / CPU-seconds per principal) live in the BFF
rate-limit/quota layer; the sandbox enforces the *per-run* envelope above.

---

*Backends are implemented in `himmy/services/sandbox/` (`container_sandbox.py`,
`gvisor_sandbox.py`, `firecracker_sandbox.py`) and selected in `factory.py`. The container
guarantees are live-verified by `tests/sandbox/test_container_sandbox.py`; the gVisor and
Firecracker command/flag/policy contracts by `tests/sandbox/test_gvisor_sandbox.py` and
`tests/sandbox/test_firecracker_sandbox.py`.*
