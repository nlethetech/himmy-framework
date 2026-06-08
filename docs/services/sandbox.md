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
| `himmy/services/sandbox/container_sandbox.py` | `ContainerSandbox` — hardened Docker/Podman isolate for untrusted code. |
| `himmy/services/sandbox/factory.py` | `build_sandbox(mode)` selection, `DisabledSandbox`, `CODE_EXEC_MODES`. |
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
| `file_size_mb` | `16` | `RLIMIT_FSIZE` — caps any single file the code writes (also the `/tmp` tmpfs size in the container). |
| `max_output_bytes` | `64 * 1024` | Bounds captured stdout/stderr. |
| `network` | `False` | **Advisory** for subprocess (not enforced); honored by the container backend (`none` vs `bridge`). |
| `allow_env` | `[]` | Allow-list of env vars passed to the child; everything else is stripped. |

`SandboxResult` carries `ok`, `exit_code`, `stdout`, `stderr`, `timed_out`,
`duration_ms`, `truncated`, and the `limits` that were applied.

## How it works / data flow

### Backend selection (`factory.py`)

`build_sandbox(mode, *, limits, image, engine)` normalizes the mode string
(`CODE_EXEC_MODES = ("off", "subprocess", "container")`) and returns:

- `off` → `DisabledSandbox` — every `run_code` returns a structured refusal
  (`ok=False`, a message telling the operator to set `HIMMY_CODE_EXEC`). This is the
  recommended default for multi-tenant served deployments.
- `container` → `ContainerSandbox`.
- anything else (incl. `subprocess`) → `SubprocessSandbox` (the default).

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
`python:3.12-slim`, must already be present). Per run it materializes the workdir
(`chmod 0755`/`0644` so the non-root container user can traverse + read the
read-only bind mount) and builds a hardened `docker run` argv:

- `--network none` (or `bridge` when `limits.network`) — egress denied by default.
- `--read-only` root filesystem + a `--tmpfs /tmp:rw,nosuid,nodev,size=<file_size_mb>m`.
- `--cap-drop ALL` + `--security-opt no-new-privileges`.
- `--user 65534:65534` (non-root) so a container escape isn't immediately root.
- `--pids-limit` (default 128), `--memory` / `--memory-swap`, `--cpus` (default 1.0)
  — fork-bomb, OOM, and CPU containment.
- `-v {workdir}:/work:ro -w /work` — input files only, read-only.
- Optional `--runtime <runtime>` (the gVisor/Kata seam, see below).
- Allow-listed env vars passed via `-e`.
- The in-container command is `timeout -k 1 -s KILL <timeout_seconds> python -I -B
  /work/__sandbox_main__.py` — a **hard in-container wall-clock kill** (coreutils
  `timeout`), backed by an **outer watchdog** (`wait_for(timeout + startup_grace)`)
  that force-removes (`docker rm -f`) an overrunning container. Exit codes `124`/`137`
  are normalized to `timed_out=True`.

`ContainerSandbox.available(engine="docker")` reports whether the engine CLI is on
`PATH` (used to skip-gate tests). `HIMMY_SANDBOX_TMP` / the `workdir_base` arg points
the throwaway workdir at a host path the engine shares (needed on Docker Desktop /
macOS, where `$TMPDIR` isn't shared by default).

### The gVisor / Kata seam

A standard container shares the host kernel — a kernel exploit is a host compromise.
For hostile multi-tenant workloads, pass `runtime=` to `ContainerSandbox` (wired to
`docker run --runtime`):

- gVisor — `ContainerSandbox(runtime="runsc")` (a user-space kernel; syscalls
  intercepted).
- Kata / Firecracker — `runtime="kata-runtime"` / a Firecracker shim (each run gets
  its own micro-VM with a separate guest kernel).

These are drop-in via the same `Sandbox` protocol; only the configured runtime
changes. See [`../enterprise/sandbox_backends.md`](../enterprise/sandbox_backends.md)
for the full threat model and per-tenant quota notes.

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
| `HIMMY_SANDBOX_TMP` | (unset) | Host path for the bind-mounted workdir (Docker Desktop / macOS). |

`SandboxLimits` is configured via `ToolkitConfig.sandbox_limits` (see
`himmy/toolkit/config.py`); there is no individual env var per limit field.

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
- `run_python` is always approval-gated regardless of backend.
- Default `HIMMY_CODE_EXEC` stays `subprocess` for backward compatibility, but a
  served, multi-tenant deployment should set `off` or `container`.

## Related docs

- [Sandbox backends (enterprise, WS2)](../enterprise/sandbox_backends.md)
- [Guardrails](guardrails.md) — the other safety control on the tool surface.
- [Enterprise hardening plan (WS2)](../enterprise/HARDENING_PLAN.md)
