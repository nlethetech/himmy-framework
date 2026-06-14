"""Unit tests for scripts/airgap_bundle.py.

``scripts/`` is not an importable package, so the module is loaded by path via
importlib. The orchestration is driven by a fake ``run`` recorder so no real
docker / pip / network is touched; ``--dry-run`` is asserted to make zero calls.
"""

from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "airgap_bundle.py"


def _load_module() -> Any:
    spec = importlib.util.spec_from_file_location("airgap_bundle", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


airgap = _load_module()


# --------------------------------------------------------------------------- #
# Pure logic
# --------------------------------------------------------------------------- #
def test_bundle_name() -> None:
    assert airgap.bundle_name("0.1.0") == "himmy-airgap-0.1.0-linux-amd64.tar"


def test_image_specs_default_and_override() -> None:
    default = airgap.image_specs("9.9.9")
    assert default["studio"] == "himmy-studio:9.9.9"
    assert default["pgvector"] == "pgvector/pgvector:pg16"
    # ollama MUST be the compose-pinned tag, never :latest (else docker load
    # registers one tag and compose up pulls another).
    assert default["ollama"] == "ollama/ollama:0.13.5"
    assert default["ollama"] == airgap.OLLAMA_IMAGE
    # busybox is shipped so the air-gapped installer can untar models offline.
    assert default["busybox"] == "busybox:1.36"
    assert default["busybox"] == airgap.BUSYBOX_IMAGE

    override = airgap.image_specs("9.9.9", studio_image="reg/himmy:custom")
    assert override["studio"] == "reg/himmy:custom"
    # the other images are unaffected by a studio override
    assert override["pgvector"] == default["pgvector"]
    assert override["ollama"] == default["ollama"]


def test_ollama_tag_matches_compose_pin() -> None:
    """The bundled ollama tag must equal the tag pinned in compose, byte-for-byte."""
    compose = airgap._COMPOSE_DIR / "docker-compose.yml"
    if not compose.is_file():
        pytest.skip("compose file absent (produced by the compose lane)")
    refs = airgap.compose_image_refs(compose.read_text("utf-8"))
    ollama_refs = {r for r in refs if r.startswith("ollama/ollama")}
    assert ollama_refs, "compose declares no ollama/ollama image"
    assert ollama_refs == {airgap.OLLAMA_IMAGE}


def test_wheel_download_argv_is_exact() -> None:
    argv = airgap.wheel_download_argv()
    # the load-bearing flags, in order
    assert "download" in argv
    assert argv[argv.index("--platform") + 1] == "manylinux2014_x86_64"
    assert argv[argv.index("--python-version") + 1] == "312"
    assert "--only-binary=:all:" in argv
    assert argv[-1] == ".[studio,knowledge,postgres,encryption,auth]"


def test_format_size() -> None:
    assert airgap.format_size(0) == "0.0 B"
    assert airgap.format_size(1024) == "1.0 KiB"
    assert airgap.format_size(1024**3) == "1.0 GiB"


def test_size_estimates_scale_with_models() -> None:
    none = airgap.size_estimates([])
    assert none["models"] == (0, 0)
    three = airgap.size_estimates(["a", "b", "c"])
    lo, hi = three["models"]
    assert lo < hi
    # three models land in (roughly) the 6-12 GB total band
    est = airgap.size_estimates(["a", "b", "c"])
    total_low = sum(low for low, _ in est.values())
    total_high = sum(high for _, high in est.values())
    gib = 1024**3
    assert total_low >= 5 * gib
    assert total_high <= 13 * gib


def test_plan_lines_mention_key_artifacts() -> None:
    images = airgap.image_specs("0.1.0")
    lines = airgap.plan_lines("0.1.0", images, ["qwen2.5:3b-instruct"], "linux/amd64")
    text = "\n".join(lines)
    assert "himmy-airgap-0.1.0-linux-amd64.tar" in text
    assert "qwen2.5:3b-instruct" in text
    assert "estimated total" in text


def test_sha256_file(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello himmy")
    import hashlib

    assert airgap.sha256_file(p) == hashlib.sha256(b"hello himmy").hexdigest()


def test_build_manifest_shape() -> None:
    images = airgap.image_specs("0.1.0")
    manifest = airgap.build_manifest(
        version="0.1.0",
        platform="linux/amd64",
        images=images,
        image_digests={"studio": "sha256:abc"},
        models=["m1"],
        artifact_sha256={"images/studio.tar.gz": "deadbeef"},
    )
    assert manifest["schema_version"] == airgap.MANIFEST_SCHEMA_VERSION == 1
    assert manifest["version"] == "0.1.0"
    assert manifest["platform"] == "linux/amd64"
    studio = next(i for i in manifest["images"] if i["name"] == "studio")
    assert studio["digest"] == "sha256:abc"
    # an image without a known digest gets "" not a missing key
    assert next(i for i in manifest["images"] if i["name"] == "ollama")["digest"] == ""
    assert manifest["models"] == ["m1"]
    assert manifest["artifacts"] == [
        {"path": "images/studio.tar.gz", "sha256": "deadbeef"}
    ]


def test_compose_image_refs_parses_and_resolves_vars() -> None:
    text = (
        "services:\n"
        "  studio:\n    image: himmy-studio:${HIMMY_VERSION:-0.1.0}\n"
        "  pg:\n    image: pgvector/pgvector:pg16\n"
        "  ol:\n    image: ollama/ollama:0.13.5  # pinned\n"
        '  q:\n    image: "busybox:1.36"\n'
    )
    refs = airgap.compose_image_refs(text)
    assert refs == {
        "himmy-studio:0.1.0",
        "pgvector/pgvector:pg16",
        "ollama/ollama:0.13.5",
        "busybox:1.36",
    }


def test_resolve_compose_var_defaults() -> None:
    assert airgap._resolve_compose_var("${V:-0.1.0}") == "0.1.0"
    assert airgap._resolve_compose_var("${V-0.1.0}") == "0.1.0"
    assert airgap._resolve_compose_var("a:${TAG:-x}") == "a:x"
    # no default -> empty
    assert airgap._resolve_compose_var("${V}") == ""


def test_wheelhouse_local_project_refusal_prints_workaround(
    tmp_path: Path,
) -> None:
    """The pip local-project + --only-binary refusal must surface the doc'd fix."""
    import subprocess

    def fake_run(argv: Any, *, check: bool = True, cwd: str | None = None) -> Any:
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=list(argv),
            stderr="ERROR: Could not find a version ... --only-binary",
        )

    with pytest.raises(SystemExit) as ei:
        airgap._wheelhouse(fake_run, tmp_path / "wheels", "linux/amd64")
    msg = str(ei.value)
    assert "python3 -m build --wheel" in msg
    assert "--no-index" not in msg  # this is the build-side message, not install


def test_sha256sums_text_is_coreutils_format() -> None:
    text = airgap.sha256sums_text({"b/x": "11", "a/y": "22"})
    # two-space separator, sorted by path
    assert text == "22  a/y\n11  b/x\n"


# --------------------------------------------------------------------------- #
# Fake runner for orchestration tests
# --------------------------------------------------------------------------- #
class FakeRun:
    """Records argv invocations; optionally fails on a matching command."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.calls: list[list[str]] = []
        self.fail_on = fail_on
        self.digest_json = '["pgvector/pgvector@sha256:feed"]'

    def __call__(
        self,
        argv: Any,
        *,
        capture: bool = False,
        check: bool = True,
        cwd: str | None = None,
    ) -> Any:
        argv = list(argv)
        self.calls.append(argv)
        if self.fail_on is not None and self.fail_on in " ".join(argv):
            raise RuntimeError(f"boom: {self.fail_on}")
        # Emulate `docker save -o <path>` producing a tar so the stdlib gzip step
        # in _save_image_gz has a file to read.
        if argv[:2] == ["docker", "save"] and "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_bytes(b"FAKE-IMAGE-TAR")
        # Emulate busybox tarring the volume into the host dir: the LAST -v maps
        # <host_dir>:/out and tar writes /out/<name>; create that host file.
        if "tar" in argv and "-cf" in argv:
            out_arg = argv[argv.index("-cf") + 1]  # e.g. /out/ollama-models.tar
            host_dir = Path([a for a in argv if a.endswith(":/out")][0].split(":")[0])
            (host_dir / Path(out_arg).name).write_bytes(b"FAKE-VOLUME-TAR")
        if capture:
            return SimpleNamespace(stdout=self.digest_json, returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    def flat(self) -> str:
        return "\n".join(" ".join(c) for c in self.calls)


def _seed_compose(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the module's compose/installer paths at a temp fixture."""
    compose = tmp_path / "compose"
    compose.mkdir()
    (compose / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    (compose / ".env.example").write_text("X=1\n", encoding="utf-8")
    installer = tmp_path / "airgap_install.sh"
    installer.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(airgap, "_COMPOSE_DIR", compose)
    monkeypatch.setattr(airgap, "_INSTALLER", installer)
    return compose


def test_build_bundle_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_compose(tmp_path, monkeypatch)
    fake = FakeRun()
    out_dir = tmp_path / "out"

    out_tar = airgap.build_bundle(
        version="0.1.0",
        out_dir=out_dir,
        studio_image=None,
        build_studio=True,
        models=["qwen2.5:3b-instruct"],
        platform="linux/amd64",
        run=fake,
    )

    assert out_tar == out_dir / "himmy-airgap-0.1.0-linux-amd64.tar"
    assert out_tar.is_file()

    flat = fake.flat()
    # studio image built, all images saved, wheelhouse downloaded
    assert "docker build" in flat
    assert "docker save" in flat
    assert "pip download" in flat
    # model pulled into the himmy_ollama volume and the volume tarred
    assert "docker volume create himmy_ollama" in flat
    assert "ollama pull qwen2.5:3b-instruct" in flat
    assert "busybox" in flat
    # non-studio images are pulled FOR THE TARGET PLATFORM (issue 5); the
    # locally-built studio image is NOT pulled.
    assert "docker pull --platform linux/amd64 ollama/ollama:0.13.5" in flat
    assert "docker pull --platform linux/amd64 pgvector/pgvector:pg16" in flat
    assert "docker pull --platform linux/amd64 busybox:1.36" in flat
    assert "docker pull --platform linux/amd64 himmy-studio:0.1.0" not in flat

    # the outer tar carries manifest.json + SHA256SUMS + the compose copy, and
    # the saved image set + manifest cover the busybox + pinned ollama tags
    with tarfile.open(out_tar) as tar:
        names = tar.getnames()
        assert "manifest.json" in names
        assert "SHA256SUMS" in names
        assert "airgap_install.sh" in names
        assert any(n.startswith("compose") for n in names)
        assert "images/busybox.tar.gz" in names
        assert "images/ollama.tar.gz" in names
        manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]
    assert manifest["schema_version"] == 1
    assert manifest["models"] == ["qwen2.5:3b-instruct"]
    ollama = next(i for i in manifest["images"] if i["name"] == "ollama")
    assert ollama["ref"] == "ollama/ollama:0.13.5"
    assert ollama["platform"] == "linux/amd64"
    studio = next(i for i in manifest["images"] if i["name"] == "studio")
    assert studio["platform"] == "local"


def test_bundle_covers_every_compose_image() -> None:
    """Every `image:` tag compose references must be in the bundle's image set.

    Skips gracefully if the compose file isn't present yet (compose lane owns it).
    This is the regression guard for issue 1: a docker-load of one tag followed by
    a compose-up that pulls a different (absent) tag.
    """
    compose = airgap._COMPOSE_DIR / "docker-compose.yml"
    if not compose.is_file():
        pytest.skip("compose file absent (produced by the compose lane)")

    compose_refs = airgap.compose_image_refs(compose.read_text("utf-8"))
    assert compose_refs, "parsed no image: refs from compose"

    # The bundle ships these refs (studio uses the compose default version). Derive
    # the studio version from what compose actually references so this guard never
    # drifts when HIMMY_VERSION is bumped.
    studio_refs = [r for r in compose_refs if r.startswith("himmy-studio:")]
    assert len(studio_refs) == 1, f"expected one himmy-studio ref, got {studio_refs}"
    studio_version = studio_refs[0].split(":", 1)[1]
    bundled = set(airgap.image_specs(studio_version).values())

    missing = compose_refs - bundled
    assert not missing, (
        f"compose references images the bundle does not save: {sorted(missing)}; "
        f"bundle ships {sorted(bundled)}"
    )


def test_run_callable_signature_parity() -> None:
    """The fake recorder MUST mirror _default_run's signature exactly.

    Issue 4 was masked because the fake accepted **kwargs the real runner didn't,
    so a real build crashed on `cwd=` after the multi-GB saves. Lock the keyword
    surface (names + kinds + defaults) so the class of bug cannot recur.
    """
    import inspect

    real = inspect.signature(airgap._default_run)
    fake = inspect.signature(FakeRun.__call__)
    # drop `self` from the bound-method comparison
    fake_params = {n: p for n, p in fake.parameters.items() if n != "self"}

    assert set(fake_params) == set(real.parameters), (
        f"fake params {set(fake_params)} != real {set(real.parameters)}"
    )
    for name, real_p in real.parameters.items():
        fp = fake_params[name]
        assert fp.kind == real_p.kind, f"{name}: kind {fp.kind} != {real_p.kind}"
        assert fp.default == real_p.default, (
            f"{name}: default {fp.default!r} != {real_p.default!r}"
        )


def test_default_run_passes_cwd(monkeypatch: pytest.MonkeyPatch) -> None:
    """_default_run must forward cwd to subprocess.run (issue 4 real-build crash)."""
    import subprocess as _sp

    seen: dict[str, Any] = {}

    def fake_subprocess_run(argv: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(_sp, "run", fake_subprocess_run)
    airgap._default_run(["echo", "hi"], cwd="/tmp", check=True)
    assert seen["cwd"] == "/tmp"
    assert seen["check"] is True


def test_build_bundle_no_models_skips_ollama(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_compose(tmp_path, monkeypatch)
    fake = FakeRun()
    airgap.build_bundle(
        version="0.1.0",
        out_dir=tmp_path / "out",
        studio_image="reg/himmy:custom",
        build_studio=False,
        models=[],
        platform="linux/amd64",
        run=fake,
    )
    flat = fake.flat()
    assert "docker build" not in flat  # studio image reused, not built
    assert "ollama pull" not in flat
    assert "docker volume create himmy_ollama" not in flat


def test_model_pull_cleanup_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pull failure must still `docker rm -f` the throwaway container."""
    fake = FakeRun(fail_on="ollama pull")
    with pytest.raises(RuntimeError):
        airgap._pull_models_into_volume(
            fake, "himmy_ollama", ["badmodel"], "ollama/ollama:latest"
        )
    # the GUARANTEED finally fired the cleanup even though the pull blew up
    assert any(c[:3] == ["docker", "rm", "-f"] for c in fake.calls)


def test_save_image_gz_cleans_raw_tar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "images" / "studio.tar.gz"

    def fake_run(argv: Any, *, capture: bool = False, check: bool = True) -> Any:
        # emulate `docker save -o <path>` writing a tar
        argv = list(argv)
        if argv[:2] == ["docker", "save"]:
            out = Path(argv[argv.index("-o") + 1])
            out.write_bytes(b"FAKE-TAR")
        return SimpleNamespace(stdout="", returncode=0)

    airgap._save_image_gz(fake_run, "studio:0.1.0", dest)
    assert dest.is_file()
    # the intermediate .raw.tar must be gone
    assert not dest.with_suffix(".raw.tar").exists()
    import gzip

    with gzip.open(dest, "rb") as fh:
        assert fh.read() == b"FAKE-TAR"


# --------------------------------------------------------------------------- #
# CLI / dry-run
# --------------------------------------------------------------------------- #
def test_dry_run_makes_no_subprocess_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    _seed_compose(tmp_path, monkeypatch)

    called: list[Any] = []

    def boom(*args: Any, **kwargs: Any) -> Any:
        called.append(args)
        raise AssertionError("dry-run must not invoke the runner")

    monkeypatch.setattr(airgap, "_default_run", boom)
    monkeypatch.setattr(airgap, "build_bundle", boom)

    rc = airgap.main(["build", "--dry-run"])
    assert rc == 0
    assert called == []
    out = capsys.readouterr().out
    assert "air-gap bundle plan" in out


def test_build_errors_without_compose(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setattr(airgap, "_COMPOSE_DIR", missing)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("must not build when compose is missing")

    monkeypatch.setattr(airgap, "build_bundle", boom)
    rc = airgap.main(["build", "--out", str(tmp_path / "out")])
    assert rc == 1


def test_dry_run_notes_missing_compose(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(airgap, "_COMPOSE_DIR", tmp_path / "absent")
    rc = airgap.main(["build", "--dry-run"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "NOTE" in err
