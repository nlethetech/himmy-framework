#!/usr/bin/env python3
"""Build a self-contained, no-network install bundle for an air-gapped site.

himmy is pitched offline-first, but a real air-gapped install still needs the
*bits* carried in: container images, a Python wheelhouse, and the Ollama models.
This script assembles all of that into one uncompressed outer tar
(``himmy-airgap-<version>-linux-amd64.tar``) that the bundled POSIX installer
(``scripts/airgap_install.sh``) unpacks on a machine with only docker + compose.

The bundle contains:
  - ``images/*.tar.gz``  — gzipped ``docker save`` of studio / pgvector / ollama
                           / busybox (busybox is also needed on the target to
                           untar the models volume)
  - ``wheels/``          — a manylinux2014_x86_64 / cpython-3.12 wheelhouse
  - ``models/ollama-models.tar`` — the Ollama models volume, tarred via busybox
  - ``compose/``         — a copy of ``deploy/compose/``
  - ``airgap_install.sh``— the installer
  - ``manifest.json`` + ``SHA256SUMS`` — versions, image digests, per-file sha256

Everything that touches docker / pip / the network goes through an injected
``run`` callable, so the orchestration is unit-tested with a fake recorder and
``--dry-run`` does ZERO side effects (it only prints the plan + a size estimate,
which is what CI exercises).

Usage:
    python scripts/airgap_bundle.py build --dry-run
    python scripts/airgap_bundle.py build --out dist/
    python scripts/airgap_bundle.py build --out dist/ --models qwen2.5:3b-instruct
    python scripts/airgap_bundle.py build --studio-image himmy-studio:0.2.0 --platform linux/amd64
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
_COMPOSE_DIR = _REPO_ROOT / "deploy" / "compose"
_INSTALLER = Path(__file__).resolve().parent / "airgap_install.sh"

# Schema version of the emitted manifest.json (bump on incompatible changes).
MANIFEST_SCHEMA_VERSION = 1

# Image used to tar the Ollama models volume from inside the container runtime
# (no host-side mount of a docker-managed volume needed). Also SAVED into the
# bundle so the air-gapped installer can untar the models volume without a
# registry pull (busybox is ~2 MB gzipped).
BUSYBOX_IMAGE = "busybox:1.36"

# Ollama runtime image. MUST stay byte-identical to the tag pinned in
# deploy/compose/docker-compose.yml — otherwise `docker load` registers one tag
# on the air-gapped target and `compose up` tries to pull the other and fails.
# A test parses the compose file and enforces that every image: tag it references
# is covered by the bundle's saved image set, so this constant cannot drift.
OLLAMA_IMAGE = "ollama/ollama:0.13.5"

# The Ollama models a default himmy stack expects (chat + two embedders).
DEFAULT_MODELS: tuple[str, ...] = (
    "qwen2.5:3b-instruct",
    "nomic-embed-text",
    "qwen3-embedding",
)

# Python extras baked into the wheelhouse (must stay in sync with the runbook).
WHEELHOUSE_EXTRAS = "studio,knowledge,postgres,encryption,auth"

# pip download targets for a linux/amd64 served deployment on cpython 3.12.
PIP_PLATFORM = "manylinux2014_x86_64"
PIP_PYTHON_VERSION = "312"

# Volume name MUST match deploy/compose so `docker load`ed models land where the
# studio stack reads them.
OLLAMA_VOLUME = "himmy_ollama"

# A ``run`` callable: mirrors subprocess.run's keyword surface we use, returns an
# object exposing ``.stdout`` (str) when ``capture`` is requested.
RunFn = Callable[..., Any]


def _eprint(*args: Any) -> None:
    print(*args, file=sys.stderr)


# --------------------------------------------------------------------------- #
# Pure logic (no side effects) — fully unit-tested.
# --------------------------------------------------------------------------- #
def bundle_name(version: str) -> str:
    """Outer-tar filename for a given himmy version."""
    return f"himmy-airgap-{version}-linux-amd64.tar"


def image_specs(version: str, studio_image: str | None = None) -> dict[str, str]:
    """Map of logical name -> image ref for the images we ship.

    ``studio`` defaults to the locally-built ``himmy-studio:<version>`` tag but
    can be overridden (e.g. to reuse an image built by a prior CI stage). The
    ``ollama`` ref is pinned via ``OLLAMA_IMAGE`` to match deploy/compose, and
    ``busybox`` is shipped so the air-gapped installer can untar the models
    volume without a registry pull.
    """
    return {
        "studio": studio_image or f"himmy-studio:{version}",
        "pgvector": "pgvector/pgvector:pg16",
        "ollama": OLLAMA_IMAGE,
        "busybox": BUSYBOX_IMAGE,
    }


def wheel_download_argv(platform: str = "linux/amd64") -> list[str]:
    """Exact ``pip download`` argv for the offline wheelhouse.

    ``--only-binary=:all:`` forces real manylinux wheels (asyncpg / cryptography
    ship them); a source-only dep would fail loudly rather than smuggle a build
    that won't run on the target. ``platform`` is accepted for symmetry with the
    image build but only ``linux/amd64`` produces the documented wheelhouse (the
    pip ``--platform`` is the fixed manylinux tag, not the docker platform).
    """
    return [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--platform",
        PIP_PLATFORM,
        "--python-version",
        PIP_PYTHON_VERSION,
        "--only-binary=:all:",
        f".[{WHEELHOUSE_EXTRAS}]",
    ]


def _resolve_compose_var(raw: str) -> str:
    """Resolve a single ``${VAR:-default}`` / ``${VAR-default}`` to its default.

    Used only to compare compose ``image:`` refs against the bundled set, so we
    take the documented default (the bundle is built for that default) rather than
    consult the environment. Refs without a ``:-``/``-`` default are left as-is.
    """
    import re

    def repl(m: re.Match[str]) -> str:
        return m.group("default") or ""

    return re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*(?::?-(?P<default>[^}]*))?\}", repl, raw)


def compose_image_refs(compose_text: str) -> set[str]:
    """Every ``image:`` ref declared in a compose file, vars resolved to defaults.

    A tiny line-scanner (no YAML dep): grabs ``image:`` values, strips quotes and
    inline ``#`` comments, and resolves ``${VAR:-default}`` interpolation. Good
    enough to assert the air-gap bundle covers what compose will try to run.
    """
    import re

    refs: set[str] = set()
    for line in compose_text.splitlines():
        m = re.match(r"\s*image:\s*(?P<val>.+?)\s*$", line)
        if not m:
            continue
        val = m.group("val")
        # drop a trailing inline comment (only when not inside the ref)
        if " #" in val:
            val = val.split(" #", 1)[0].rstrip()
        val = val.strip().strip('"').strip("'")
        val = _resolve_compose_var(val)
        if val:
            refs.add(val)
    return refs


def format_size(num_bytes: int) -> str:
    """Human-readable size (binary units, one decimal)."""
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def size_estimates(models: Sequence[str]) -> dict[str, tuple[int, int]]:
    """Rough (low, high) byte estimates per artifact group, for the dry-run.

    These are deliberately coarse documentation-grade figures — the real bundle
    size depends on the model set. Three models land in the ~6–12 GB band.
    """
    gib = 1024**3
    # Coarse per-model band: a 3B chat model + embedders are ~1–2.5 GB each, so
    # the default 3-model set + images + wheels lands in the documented ~6–12 GB.
    per_model_low, per_model_high = gib, gib * 5 // 2
    n = max(len(models), 0)
    return {
        "images": (2 * gib, 4 * gib),
        "wheels": (200 * 1024**2, 600 * 1024**2),
        "models": (n * per_model_low, n * per_model_high),
    }


def plan_lines(
    version: str,
    images: dict[str, str],
    models: Sequence[str],
    platform: str,
) -> list[str]:
    """Human-readable build plan (printed in --dry-run and before a real build)."""
    est = size_estimates(models)
    total_low = sum(lo for lo, _ in est.values())
    total_high = sum(hi for _, hi in est.values())
    lines = [
        f"air-gap bundle plan for himmy {version} ({platform}):",
        f"  outer tar : {bundle_name(version)} (uncompressed)",
        "  images    : " + ", ".join(f"{k}={v}" for k, v in images.items()),
        f"            ~ {format_size(est['images'][0])}"
        f"–{format_size(est['images'][1])} gzipped",
        f"  wheelhouse: pip download .[{WHEELHOUSE_EXTRAS}] "
        f"({PIP_PLATFORM}/cp{PIP_PYTHON_VERSION})",
        f"            ~ {format_size(est['wheels'][0])}"
        f"–{format_size(est['wheels'][1])}",
        "  models    : " + (", ".join(models) if models else "(none)"),
        f"            ~ {format_size(est['models'][0])}"
        f"–{format_size(est['models'][1])}",
        f"  estimated total: {format_size(total_low)}–{format_size(total_high)}",
    ]
    return lines


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file (hex digest)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    version: str,
    platform: str,
    images: dict[str, str],
    image_digests: dict[str, str],
    models: Sequence[str],
    artifact_sha256: dict[str, str],
) -> dict[str, Any]:
    """Assemble the manifest.json payload.

    ``image_digests`` maps logical name -> ``docker inspect`` digest (or "" when
    unknown, e.g. a freshly-built local image with no registry digest).
    ``artifact_sha256`` maps bundle-relative path -> sha256 of the staged file.
    """
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "version": version,
        "platform": platform,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "images": [
            {
                "name": name,
                "ref": images[name],
                "digest": image_digests.get(name, ""),
                # Platform the image was pulled/saved for. The locally-built
                # studio image follows the build host's arch; the rest are
                # pulled for the target ``platform`` explicitly.
                "platform": "local" if name == "studio" else platform,
            }
            for name in images
        ],
        "models": list(models),
        "wheelhouse_extras": WHEELHOUSE_EXTRAS,
        "artifacts": [
            {"path": path, "sha256": digest}
            for path, digest in sorted(artifact_sha256.items())
        ],
    }


def sha256sums_text(artifact_sha256: dict[str, str]) -> str:
    """Render a SHA256SUMS file (``<hex>  <path>`` per line) for the installer.

    Format matches coreutils ``sha256sum -c`` so the POSIX installer needs no
    JSON parser. Sorted for stable output.
    """
    return "".join(
        f"{digest}  {path}\n" for path, digest in sorted(artifact_sha256.items())
    )


# --------------------------------------------------------------------------- #
# Impure orchestration — every external command goes through ``run``.
# --------------------------------------------------------------------------- #
def _default_run(
    argv: Sequence[str],
    *,
    capture: bool = False,
    check: bool = True,
    cwd: str | None = None,
) -> Any:
    """The real runner: a thin wrapper over subprocess.run.

    S603/S607 are only ignored for tests/** in this repo, so the script-side call
    is explicitly annotated. argv is always a list (never a shell string), so
    there is no shell=True / S602 surface.

    The keyword surface here (``capture``/``check``/``cwd``) is the contract every
    injected ``run`` callable must mirror; ``tests/airgap`` has a signature-parity
    test that asserts the fake recorder matches this signature exactly so a real
    build can never crash on a kwarg the fake silently swallowed.
    """
    return subprocess.run(  # noqa: S603
        list(argv),
        check=check,
        capture_output=capture,
        text=True,
        cwd=cwd,
    )


def _docker_image_digest(run: RunFn, ref: str) -> str:
    """Best-effort ``docker inspect`` digest for an image ref ("" if none)."""
    try:
        proc = run(
            ["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", ref],
            capture=True,
            check=True,
        )
    except Exception:
        return ""
    raw = (getattr(proc, "stdout", "") or "").strip()
    try:
        digests = json.loads(raw) if raw else []
    except (ValueError, TypeError):
        return ""
    return str(digests[0]) if digests else ""


def _pull_image(run: RunFn, ref: str, platform: str) -> None:
    """``docker pull`` an image for the TARGET ``platform``.

    Without ``--platform`` docker pulls the build host's native arch, so an
    arm64 Mac would silently bundle arm64 images for a linux/amd64 target. We
    pass it explicitly. Failures fall through to the caller (the save would fail
    anyway if the image isn't present).
    """
    run(["docker", "pull", "--platform", platform, ref], check=True)


def _save_image_gz(run: RunFn, ref: str, dest: Path) -> None:
    """``docker save`` an image to an uncompressed tar, then gzip it in stdlib.

    NOT ``docker save | gzip`` — a shell pipe would mean shell=True (S602). We
    save to a temp .tar and gzip with the stdlib so the runner stays argv-only.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    raw_tar = dest.with_suffix(".raw.tar")
    try:
        run(["docker", "save", "-o", str(raw_tar), ref], check=True)
        with raw_tar.open("rb") as src, gzip.open(dest, "wb") as gz:
            shutil.copyfileobj(src, gz)
    finally:
        raw_tar.unlink(missing_ok=True)


def _ensure_studio_image(run: RunFn, ref: str, build: bool) -> None:
    """Build the studio image if requested, else assume it exists."""
    if build:
        run(
            [
                "docker",
                "build",
                "-t",
                ref,
                "-f",
                str(_REPO_ROOT / "Dockerfile"),
                str(_REPO_ROOT),
            ],
            check=True,
        )


def _pull_models_into_volume(
    run: RunFn, volume: str, models: Sequence[str], ollama_image: str
) -> None:
    """Pull each Ollama model into ``volume`` via a throwaway container.

    A named ``docker run`` container is used (not ``--rm``) so the GUARANTEED
    try/finally can ``docker rm -f`` it even if a pull fails midway — leaking a
    multi-GB stopped container on the build host is the failure we refuse to ship.
    """
    run(["docker", "volume", "create", volume], check=True)
    container = "himmy-airgap-ollama-pull"
    try:
        run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                container,
                "-v",
                f"{volume}:/root/.ollama",
                ollama_image,
            ],
            check=True,
        )
        for model in models:
            run(["docker", "exec", container, "ollama", "pull", model], check=True)
    finally:
        # Cleanup is unconditional: stop + remove even on a failed pull.
        run(["docker", "rm", "-f", container], check=False)


def _tar_volume(run: RunFn, volume: str, dest: Path) -> None:
    """Tar a docker volume's contents to ``dest`` on the host via busybox."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{volume}:/data:ro",
            "-v",
            f"{dest.parent}:/out",
            BUSYBOX_IMAGE,
            "tar",
            "-cf",
            f"/out/{dest.name}",
            "-C",
            "/data",
            ".",
        ],
        check=True,
    )


# Substrings some pip versions emit when refusing `pip download .[...]` combined
# with --platform/--only-binary (the local project itself has no wheel to honor
# the constraint). Matched case-insensitively to surface the documented fix.
_PIP_LOCAL_PROJECT_REFUSAL = (
    "--only-binary",
    "wheels are required",
    "no binaries permitted",
    "could not find a version",
)


def _wheelhouse(run: RunFn, dest: Path, platform: str) -> None:
    """Populate the offline wheelhouse, with a clear manylinux failure message."""
    dest.mkdir(parents=True, exist_ok=True)
    argv = wheel_download_argv(platform) + ["--dest", str(dest)]
    try:
        run(argv, check=True, cwd=str(_REPO_ROOT))
    except subprocess.CalledProcessError as exc:  # pragma: no cover - real pip only
        detail = " ".join(
            str(x) for x in (getattr(exc, "stderr", ""), getattr(exc, "output", ""))
        ).lower()
        if any(s in detail for s in _PIP_LOCAL_PROJECT_REFUSAL):
            raise SystemExit(
                "error: pip refused to `pip download` the local project together "
                "with --platform/--only-binary=:all: (the project dir itself has no "
                "wheel to satisfy the binary-only constraint). Documented "
                "workaround:\n"
                "    python3 -m build --wheel              # build himmy's own wheel\n"
                "    pip download --no-deps --dest wheels/ dist/himmy-*.whl\n"
                "    pip download --platform "
                f"{PIP_PLATFORM} --python-version {PIP_PYTHON_VERSION} "
                "--only-binary=:all: --dest wheels/ \\\n"
                f'        "himmy[{WHEELHOUSE_EXTRAS}]"   # then its deps\n'
                f"See docs/enterprise/airgap.md 'Limits'. ({exc})"
            ) from exc
        raise SystemExit(
            "error: pip download failed building the wheelhouse. A dependency may "
            f"lack a {PIP_PLATFORM} / cp{PIP_PYTHON_VERSION} wheel "
            "(--only-binary=:all: forbids source builds). Re-run on a linux/amd64 "
            f"host or pre-build the missing wheel. ({exc})"
        ) from exc


def _stage_compose_and_installer(stage: Path) -> None:
    """Copy deploy/compose/ and the installer into the staging dir."""
    shutil.copytree(_COMPOSE_DIR, stage / "compose", dirs_exist_ok=True)
    shutil.copy2(_INSTALLER, stage / "airgap_install.sh")


def _tar_dir(src: Path, dest: Path) -> None:
    """Uncompressed outer tar of a directory's contents (stdlib tarfile)."""
    with tarfile.open(dest, "w") as tar:
        for child in sorted(src.iterdir()):
            tar.add(child, arcname=child.name)


def _hash_tree(root: Path) -> dict[str, str]:
    """sha256 of every file under ``root`` (keys are root-relative posix paths)."""
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[path.relative_to(root).as_posix()] = sha256_file(path)
    return out


def build_bundle(
    *,
    version: str,
    out_dir: Path,
    studio_image: str | None,
    build_studio: bool,
    models: Sequence[str],
    platform: str,
    run: RunFn,
) -> Path:
    """Orchestrate a full bundle build. Returns the outer-tar path.

    All external work goes through ``run``; the rest is filesystem + stdlib.
    """
    images = image_specs(version, studio_image)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="himmy-airgap-") as tmp:
        stage = Path(tmp) / "bundle"
        stage.mkdir()

        _ensure_studio_image(run, images["studio"], build_studio)

        image_digests: dict[str, str] = {}
        for name, ref in images.items():
            # The studio image is produced locally (built here or supplied by the
            # caller), so it is NOT pulled — its arch follows the local build.
            # Every other image is pulled for the TARGET platform so an arm64
            # host doesn't bundle arm64 images for a linux/amd64 deployment.
            if name != "studio":
                _pull_image(run, ref, platform)
            _save_image_gz(run, ref, stage / "images" / f"{name}.tar.gz")
            image_digests[name] = _docker_image_digest(run, ref)

        _wheelhouse(run, stage / "wheels", platform)

        if models:
            _pull_models_into_volume(run, OLLAMA_VOLUME, models, images["ollama"])
            _tar_volume(run, OLLAMA_VOLUME, stage / "models" / "ollama-models.tar")

        _stage_compose_and_installer(stage)

        # Hash everything staged so far, then emit manifest + SHA256SUMS (which
        # themselves are NOT in the sums — they ARE the verification anchor).
        artifact_sha256 = _hash_tree(stage)
        manifest = build_manifest(
            version, platform, images, image_digests, models, artifact_sha256
        )
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "SHA256SUMS").write_text(
            sha256sums_text(artifact_sha256), encoding="utf-8"
        )

        out_tar = out_dir / bundle_name(version)
        _tar_dir(stage, out_tar)
    return out_tar


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _himmy_version() -> str:
    """himmy version from the installed package, falling back to pyproject."""
    try:
        from importlib.metadata import version as _pkg_version

        return _pkg_version("himmy")
    except Exception:
        pass
    try:
        import tomllib

        data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text("utf-8"))
        return str(data["project"]["version"])
    except Exception:
        return "0.0.0"


def cmd_build(args: argparse.Namespace) -> int:
    version = args.version or _himmy_version()
    models = (
        [m.strip() for m in args.models.split(",") if m.strip()]
        if args.models is not None
        else list(DEFAULT_MODELS)
    )
    images = image_specs(version, args.studio_image)

    for line in plan_lines(version, images, models, args.platform):
        print(line)

    compose_missing = not _COMPOSE_DIR.is_dir() or not any(
        _COMPOSE_DIR.glob("docker-compose*.yml")
    )

    if args.dry_run:
        if compose_missing:
            _eprint(
                f"NOTE: {_COMPOSE_DIR} has no docker-compose*.yml yet "
                "(created by the compose lane); a real build will require it."
            )
        _eprint("dry-run: no images saved, no downloads, no docker calls.")
        return 0

    if compose_missing:
        _eprint(
            f"error: {_COMPOSE_DIR} has no docker-compose*.yml — the compose "
            "artifacts must exist before bundling. Aborting."
        )
        return 1

    out_dir = Path(args.out).resolve()
    out_tar = build_bundle(
        version=version,
        out_dir=out_dir,
        studio_image=args.studio_image,
        build_studio=args.studio_image is None,
        models=models,
        platform=args.platform,
        run=_default_run,
    )
    digest = sha256_file(out_tar)
    print(f"wrote {out_tar}")
    print(f"sha256  {digest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="assemble the air-gap bundle")
    p_build.add_argument(
        "--out", default="dist", help="output directory for the bundle tar"
    )
    p_build.add_argument(
        "--version", default=None, help="bundle version (default: himmy's)"
    )
    p_build.add_argument(
        "--studio-image",
        default=None,
        help="reuse an existing studio image ref instead of building it",
    )
    p_build.add_argument(
        "--models",
        default=None,
        help=(
            "comma list of Ollama models to bundle "
            f"(default: {','.join(DEFAULT_MODELS)}; empty string = none)"
        ),
    )
    p_build.add_argument(
        "--platform", default="linux/amd64", help="target container platform"
    )
    p_build.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan + size estimate; do nothing else (CI-safe)",
    )
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
