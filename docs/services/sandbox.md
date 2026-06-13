# Sandbox

> Isolated, resource-limited execution of model-authored code, behind a single backend-agnostic protocol.

## Overview

The sandbox service runs a Python snippet inside a resource/fault (and, with the
container backend, OS-level security) envelope and returns a structured result. It
exists because letting an agent execute code is the highest-severity surface in the
framework: the design separates the *contract* (`Sandbox` protocol) from the
*isolate* (subprocess / container / disabled), so the strength of isolation is a
deployment choice — selected by the `HIMMY_CODE_EXEC` environment variable — without
any caller change.

`run_code` never raises for code-level failures (a crash, a non-zero exit, a
timeout); those are reported in the returned `SandboxResult`. It may raise only for
*infrastructure* failures (the isolate could not be created), surfaced as
`HimmyError`.

The default backend (`SubprocessSandbox`) is portable defense-in-depth against
*buggy/runaway* code, **not** a security boundary against hostile code. The
`ContainerSandbox` adds the OS-level boundary for untrusted, model-authored code, and
the same protocol is the seam where gVisor / Kata / Firecracker plug in.

## Module map

| File | Responsibility |
| --- | --- |
| `himmy/services/sandbox/base.py` | The `Sandbox` runtime-checkable Protocol (`run_code` → `SandboxResult`). |
| `himmy/services/sandbox/models.py` | `SandboxLimits` (the requested resource envelope) and `SandboxResult` (the outcome). |
| `himmy/services/sandbox/subprocess_sandbox.py` | `SubprocessSandbox` — portable, default backend (rlimits + wall-clock kill + fs/env isolation). |
| `himmy/services/sandbox/container_sandbox.py` | `ContainerSandbox` — hardened Docker/Podman isolate for untrusted code (shared host kernel). |
| `himmy/services/sandbox/gvisor_sandbox.py` | `GVisorSandbox` — the hardened container under gVisor's `runsc` user-space kernel. **Linux only.** |
| `himmy/services/sandbox/firecracker_sandbox.py` | `FirecrackerSandbox` — a throwaway KVM microVM per run via firecracker-containerd. **Linux + KVM only.** |
| `himmy/services/sandbox/factory.py` | `build_sandbox(mode)` selection + hardened-runtime capability detection, `DisabledSandbox`, `CODE_EXEC_MODES`. |
| `himmy/services/sandbox/tools.py` | `register_sandbox_tool` — expose a sandbox as a policy-gated, audited agent tool. |
| `himmy/services/sandbox/__init__.py` | Public surface re-export. |

## Key abstractions

### `Sandbox` protocol (`base.py`)

```python
@runtime_checkable
class Sandbox(Protocol):
    async def run_code(
        self,
        code: str,
        *,
        stdin: str | None = None,
        files: dict[str, str] | None = None,
    ) -> SandboxResult: ...
```

`files` are written into the working directory before the run (data the code can
`open()` by relative name); `stdin` is fed to the process.

### `SandboxLimits` / `SandboxResult` (`models.py`)

`SandboxLimits` (all defaults shown):

| Field | Default | Meaning |
| --- | --- | --- |
| `cpu_seconds` | `5.0` | POSIX `RLIMIT_CPU` in the child (best-effort). |
| `timeout_seconds` | `10.0` | Hard wall-clock kill. |
| `memory_mb` | `256` | `RLIMIT_AS` (silently not enforced on some platforms, e.g. macOS). |
| `file_size_mb` | `16` | `RLIMIT_FSIZE` — caps any single file the code writes (also the `/sandbox` tmpfs size in the container, and the **ceiling on total input** code+files the container backend will accept — see below). |
| `max_output_bytes` | `64 * 1024` | Bounds captured stdout/stderr. |
| `network` | `False` | **Advisory** for subprocess (not enforced). The container backend's network posture is the `SandboxPolicy.network` mode (`none` default / `egress_proxy`), not this flag. |
| `allow_env` | `[]` | Allow-list of env vars passed to the child; everything else is stripped. |

`SandboxResult` carries `ok`, `exit_code`, `stdout`, `stderr`, `timed_out`,
`duration_ms`, `truncated`, and the `limits` that were applied.

## How it works / data flow

### Backend selection (`factory.py`)

`build_sandbox(mode, *, limits, policy, image, engine, server_context, event_sink,
capability_check)` normalizes the mode string
(`CODE_EXEC_MODES = ("off", "subprocess", "container", "gvisor", "firecracker")`) and
returns:

- `off` → `DisabledSandbox` — every `run_code` returns a structured refusal
  (`ok=False`, a message telling the operator to set `HIMMY_CODE_EXEC`). This is the
  recommended default for multi-tenant served deployments.
- `container` → `ContainerSandbox` (passed the `SandboxPolicy` + optional `event_sink`).
- `gvisor` → `GVisorSandbox` — the same hardened container under the `runsc` runtime.
- `firecracker` → `FirecrackerSandbox` — a KVM microVM per run (firecracker-containerd).
- anything else (incl. `subprocess`) → `SubprocessSandbox` (the default) — **unless**
  `server_context` (or `HIMMY_REQUIRE_SANDBOX=1`) is set, in which case it raises
  (fail-closed; the unsandboxed backend is refused in a server posture).

For the hardened Linux-only backends, `build_sandbox` **capability-detects** the runtime
at startup when `HIMMY_SANDBOX_VERIFY_RUNTIME=1` (or `capability_check=True`): `gvisor`
without a registered `runsc`, or `firecracker` without `/dev/kvm` + the driver, raises a
clear `HimmyError` instead of failing on the first run. The probe is off by default so
constructing a backend for inspection on any host (incl. macOS) is unchanged.

### `SubprocessSandbox`

For each run it creates a throwaway `tempfile.TemporaryDirectory`, writes input
`files` (path-traversal rejected by `_safe_path`) and `__sandbox_main__.py`, then
launches `python -I -B __sandbox_main__.py` (`-I` isolated mode ignores `PYTHON*`
env and user site; `-B` writes no pyc). On POSIX it applies a `preexec_fn` that
calls `setrlimit` for CPU/address-space/file-size/core, runs with
`start_new_session=True`, and feeds a minimal env (`PATH`/`LANG`/`LC_ALL` plus
allow-listed passthroughs). A `wait_for(timeout_seconds)` enforces the wall-clock
limit; on timeout the **whole process group** is `SIGKILL`ed via
`os.killpg(os.getpgid(pid), ...)`. Output is decoded and truncated to
`max_output_bytes`.

Threat model (documented in the module): isolates resources and faults (infinite
loops, fork bombs, runaway writes, env/secret leakage), but a plain subprocess can
still attempt network I/O and read world-readable files, and `RLIMIT_AS` is not
enforced everywhere. Use the container backend (or a custom `Sandbox`) for untrusted
code.

### `ContainerSandbox`

Shells out to the Docker/Podman CLI (no SDK; the image, default
`python:3.12-slim`, must already be present). It enforces all **five enterprise
guarantees** per run, driven by a `SandboxPolicy` (locked-down defaults). Input files
are **not** bind-mounted — a host bind mount is host access we refuse — they are
base64-embedded in a JSON bundle the in-container bootstrap materializes onto the tmpfs
workspace (`/sandbox`), so the run owns its only writable surface and shares no host
path.

The bundle is delivered **out-of-band, streamed on the container's stdin** (length-prefixed:
the first line is the decimal byte length, then exactly that many bundle bytes, then the
caller's own `stdin`), never on the `docker run` argv. This matters: embedding a large
bundle in a `sh -c` argv string hit the kernel's per-arg size ceiling
(`MAX_ARG_STRLEN`, ~128 KB/arg on Linux) — and because the bundle is double-base64
(content → base64 → JSON → base64, ~1.78× expansion) even a ~70 KB code snippet or input
file failed with an opaque `exec /usr/bin/sh: argument list too long` (`ok=False`,
`exit_code=255`). Streaming on stdin removes that ceiling; the bootstrap reads the bundle
with raw `os.read` (no buffered read-ahead) so the snippet still receives the caller's
remaining `stdin` untouched. The total input (decoded code + files) is bounded by the
`/sandbox` tmpfs envelope (`file_size_mb`, default 16 MB) it must land on;
`_build_bundle` pre-validates and raises a clear `HimmyError` naming `file_size_mb` if it
is exceeded, rather than failing opaquely mid-run.

The hardened `docker run` argv:

- **(1) Network — default DENY**: `--network none` (no interface, egress impossible).
  The optional mediated allow-list (`NetworkMode.EGRESS_PROXY`) instead attaches to an
  operator-provisioned isolated network and injects `HTTP(S)_PROXY` pointing at a
  forward proxy — never a general bridge.
- **(2) Ephemeral**: `--rm` + a fresh container per run + the `/sandbox` workspace on
  `--tmpfs ...,mode=1777,size=<file_size_mb>m`. Nothing the code writes survives and no
  two runs share state.
- **(3) Resource limits**: `--pids-limit` (default 128), `--memory` / `--memory-swap`,
  `--cpus` (default 1.0), and `--ulimit` (default `nofile`, `nproc`, `core=0`). Plus a
  **hard in-container wall-clock kill** — `timeout -k 1 -s KILL <timeout_seconds> python
  -I -B __sandbox_main__.py` — backed by an **outer watchdog**
  (`wait_for(timeout + startup_grace)`) that `docker rm -f`s an overrunning container.
  Exit codes `124`/`137` normalize to `timed_out=True`.
- **(4) No host access**: `--read-only` rootfs, `--cap-drop ALL`,
  `--security-opt no-new-privileges`, `--user 65534:65534` (non-root), and **no `-v`
  bind mount** of any host path.
- **(5) Logged**: when an `event_sink` is supplied, every run emits `TOOL_CALLED` then
  `TOOL_COMPLETED`/`TOOL_FAILED` through the event/audit spine (the `code` pack's tool
  path already records this for `run_python`).
- Optional `--runtime <runtime>` (the gVisor/Kata seam, see below); allow-listed env
  vars passed via `-e`.

`ContainerSandbox.available(engine="docker")` reports whether the engine CLI is on
`PATH` (used to skip-gate tests). Because there is no bind mount, no host path needs to
be shared — the backend runs unmodified on Docker Desktop / macOS (the old
`HIMMY_SANDBOX_TMP` / `workdir_base` knob is no longer needed).

### Stronger boundaries: gVisor / Firecracker

A standard container shares the host kernel — a kernel exploit is a host compromise. For
hostile multi-tenant workloads, select a stronger backend by config (not just a `runtime=`
knob); each enforces the *same five-guarantee `SandboxPolicy`*:

- `HIMMY_CODE_EXEC=gvisor` → `GVisorSandbox` — the hardened container under gVisor's
  `runsc` user-space kernel (`docker run --runtime=runsc`); syscalls are intercepted, so
  the host-kernel surface shrinks ~100×. The audit spine labels these runs `gvisor`.
- `HIMMY_CODE_EXEC=firecracker` → `FirecrackerSandbox` — a throwaway KVM microVM per run
  via firecracker-containerd (the `aws.firecracker` runtime), driven through the
  docker-compatible `nerdctl` CLI so the same OCI isolation flags map onto a microVM.

Both require **Linux** (and **KVM** for Firecracker) and are **not executable on macOS** —
they are implemented and contract-tested in full, but live behaviour must be verified on a
Linux+KVM host. They are drop-in via the same `Sandbox` protocol. See
[`../enterprise/sandbox.md`](../enterprise/sandbox.md) for the security ladder, per-tier
recommendation, and deploy/verify steps, and
[`../enterprise/sandbox_backends.md`](../enterprise/sandbox_backends.md) for the concise
threat model + per-tenant quota notes.

### How code tools route to the sandbox

The `code` toolkit pack (`himmy/toolkit/code.py::register_code_pack`) is the only
caller in the framework. It reads `ToolkitConfig` (`code_exec`, `sandbox_limits`,
`sandbox_image`, `sandbox_engine` — all from `HIMMY_CODE_EXEC` / `HIMMY_SANDBOX_*`),
calls `build_sandbox(...)`, and registers the result via
`register_sandbox_tool(registry, sandbox, name="run_python", requires_approval=True)`.

`register_sandbox_tool` (`tools.py`) wraps the sandbox as a **local tool** so it
flows through the `ToolService` pipeline — approval gating, arg validation
(`SANDBOX_TOOL_ARGS_SCHEMA`), event emission, lineage. Its handler also runs
`echo_last_expression` on the code (rewrites a trailing bare expression to print its
`repr`, REPL/Jupyter-style, so a model that ends with `result` instead of
`print(result)` still produces stdout). The tool returns the `SandboxResult` as a
JSON dict and is `read_only=False` (executing code is an action). Approval is on by
default — running code is a human-in-the-loop decision.

## Configuration

| Var | Default | Effect |
| --- | --- | --- |
| `HIMMY_CODE_EXEC` | `subprocess` | Backend: `off` / `subprocess` / `container`. |
| `HIMMY_SANDBOX_IMAGE` | `python:3.12-slim` | Container image (must be pre-pulled; needs `python` + coreutils `timeout`). |
| `HIMMY_SANDBOX_ENGINE` | `docker` | `docker` / `podman`. |
| `HIMMY_SANDBOX_NETWORK` | `none` | Hardened network posture: `none` (deny) / `egress_proxy` (mediated allow-list). |
| `HIMMY_SANDBOX_EGRESS_NETWORK` | (unset) | `egress_proxy` mode: the isolated Docker network to attach. |
| `HIMMY_SANDBOX_EGRESS_PROXY` | (unset) | `egress_proxy` mode: the forward-proxy URL injected as `HTTP(S)_PROXY`. |
| `HIMMY_REQUIRE_SANDBOX` | `0` | Fail-closed: when truthy (always in a server context), refuse the unsandboxed `subprocess` backend. |

`SandboxLimits` is configured via `ToolkitConfig.sandbox_limits` (see
`himmy/toolkit/config.py`); there is no individual env var per limit field. The full
isolation contract (`SandboxPolicy`) is assembled by `ToolkitConfig.build_sandbox_policy()`.

### Fail-closed server posture

`SubprocessSandbox` isolates resources and faults but is **not** a security boundary
against hostile code. To make that impossible to forget, `build_sandbox(...)` **refuses**
the `subprocess` backend in a server context (`server_context=True`, set by every served
entrypoint) or when `HIMMY_REQUIRE_SANDBOX=1` — raising rather than silently running
untrusted code unsandboxed. A served/multi-tenant deployment must therefore run
`HIMMY_CODE_EXEC=container` (hardened isolate) or `off`. The dev/CLI default is
unchanged.

## Extension points

- Implement the `Sandbox` protocol to add a new isolate; register it via
  `register_sandbox_tool` (or return it from a customized factory) and the rest of
  the stack is unaffected.
- Configure `ContainerSandbox(runtime=...)` for gVisor/Kata/Firecracker.
- Pass a custom `SandboxLimits` to tighten/loosen the per-run envelope.

## Gotchas & invariants

- `run_code` must not raise for code-level failures — only infrastructure failures.
- `SubprocessSandbox` is **not** a security boundary; `network` is advisory there.
- `RLIMIT_AS` (memory) is silently dropped on platforms that reject it (macOS) —
  degrade-gracefully, not fail.
- `ContainerSandbox` requires the engine CLI and the image to be present; it does not
  pull images.
- `ContainerSandbox` streams the input bundle on the container's stdin (not the argv), so
  there is **no `MAX_ARG_STRLEN` ceiling** on code/file size; total input is instead
  bounded by the `file_size_mb` workspace envelope, and an oversized bundle raises a clear
  `HimmyError` naming the cap (an infra-class failure, like path traversal — not a
  code-level `SandboxResult`).
- `run_python` is always approval-gated regardless of backend.
- Default `HIMMY_CODE_EXEC` stays `subprocess` for backward compatibility, but a
  served, multi-tenant deployment should set `off` or `container`.

## Related docs

- [Sandbox backends (enterprise, WS2)](../enterprise/sandbox_backends.md)
- [Guardrails](guardrails.md) — the other safety control on the tool surface.
- [Enterprise hardening plan (WS2)](../enterprise/HARDENING_PLAN.md)
