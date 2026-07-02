"""`himmy init` / `himmy deploy --docker` emit a ghcr-based agent Dockerfile.

The container front door (P0 #4): a pip-install user with NO framework checkout must be able
to `docker build` from their agent folder. Both entrypoints emit the SAME 3-line recipe —
`FROM ghcr.io/nlethetech/himmy:<version>` + `COPY agent.yaml` + `CMD ["himmy","deploy",...]` —
so the published runtime image (built + pushed by .github/workflows/deploy.yml) supplies himmy
and the user's file is layered on. These tests prove, offline and with no docker daemon:

* init writes a Dockerfile pinning the ghcr image at the installed version, copying the spec,
  and running `himmy deploy`;
* the emitted Dockerfile is well-formed (parses into FROM/COPY/CMD instructions);
* init never clobbers a Dockerfile the user already has;
* `deploy --docker` emits the same recipe to stdout WITHOUT booting a server.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from himmy import __version__
from himmy.cli import commands


def _repo_root() -> Path:
    """Locate the repo root (holds the base ``Dockerfile``) from this test file's path."""
    return Path(__file__).resolve().parents[2]


def _parse_dockerfile(text: str) -> list[tuple[str, str]]:
    """Parse a Dockerfile into (INSTRUCTION, args) pairs, skipping comments/blanks.

    A deliberately tiny parser — enough to assert the emitted file is structurally a
    Dockerfile (each non-comment line starts with a real instruction) without a docker daemon.
    """
    out: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        instr, _, rest = line.partition(" ")
        out.append((instr.upper(), rest.strip()))
    return out


def _init_classic(tmp_path: Path) -> int:
    return commands.cmd_init(
        argparse.Namespace(
            directory=str(tmp_path),
            template=None,
            team=None,
            classic=True,
            force=False,
        )
    )


# ---------------------------------------------------------------- himmy init emits a Dockerfile


def test_init_emits_agent_dockerfile(tmp_path: Path) -> None:
    """`himmy init` drops a Dockerfile next to the scaffolded agent.yaml."""
    assert _init_classic(tmp_path) == 0
    dockerfile = tmp_path / "Dockerfile"
    assert dockerfile.exists()
    text = dockerfile.read_text(encoding="utf-8")
    # FROM the published runtime image, pinned to the installed version (never :latest).
    assert f"FROM {commands.GHCR_IMAGE}:{__version__}" in text
    # COPY the user's spec to the stable in-image path the CMD runs.
    assert "COPY agent.yaml /app/agent.yaml" in text
    # CMD runs `himmy deploy` on that spec.
    assert 'CMD ["himmy", "deploy", "-f", "agent.yaml"' in text


def test_emitted_port_matches_base_healthcheck() -> None:
    """The active CMD binds the same port the base image's inherited HEALTHCHECK probes.

    The base Dockerfile EXPOSEs + HEALTHCHECKs 8765; a CMD on any other port would curl a
    closed socket and leave the container permanently `(unhealthy)`. Assert the two agree by
    reading the port straight out of the repo's base Dockerfile (source of truth), not a
    hard-coded literal, so a future base-image port bump forces this to be revisited together.
    """
    base = (_repo_root() / "Dockerfile").read_text(encoding="utf-8")
    healthcheck_port = re.search(r"127\.0\.0\.1:(\d+)/readyz", base)
    expose_port = re.search(r"^EXPOSE\s+(\d+)", base, re.MULTILINE)
    assert healthcheck_port is not None and expose_port is not None
    # Base image is internally consistent (EXPOSE == HEALTHCHECK port).
    assert healthcheck_port.group(1) == expose_port.group(1)
    # And the emitted agent CMD binds exactly that port.
    assert str(commands.AGENT_IMAGE_PORT) == healthcheck_port.group(1)
    text = commands._agent_dockerfile_text("agent.yaml")
    assert f'"--port", "{commands.AGENT_IMAGE_PORT}"' in text


def test_emitted_dockerfile_is_fail_closed_by_default(tmp_path: Path) -> None:
    """The auto-emitted recipe is fail-closed: no baked unauth opt-in in the ACTIVE CMD.

    `himmy init` now drops this Dockerfile unprompted, so its default must not ship an
    open-by-default container. The active CMD binds 127.0.0.1 and the unauthenticated-proxy
    opt-in (`HIMMY_ALLOW_UNAUTHENTICATED` + `--host 0.0.0.0`) appears ONLY as commented lines
    the user must consciously uncomment.
    """
    assert _init_classic(tmp_path) == 0
    text = (tmp_path / "Dockerfile").read_text(encoding="utf-8")
    active = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    joined = "\n".join(active)
    # The live CMD binds loopback, never 0.0.0.0.
    assert '"--host", "127.0.0.1"' in joined
    assert "0.0.0.0" not in joined  # noqa: S104 - asserting absence, not binding
    # No active ENV bakes the unauthenticated opt-in — it lives only in a comment.
    assert "HIMMY_ALLOW_UNAUTHENTICATED" not in joined
    assert "# " in text and "HIMMY_ALLOW_UNAUTHENTICATED" in text


def test_init_dockerfile_parses(tmp_path: Path) -> None:
    """The emitted Dockerfile is structurally valid: FROM first, then COPY + CMD present."""
    assert _init_classic(tmp_path) == 0
    instrs = _parse_dockerfile((tmp_path / "Dockerfile").read_text(encoding="utf-8"))
    kinds = [k for k, _ in instrs]
    # The first instruction of any Dockerfile must be FROM (comments already stripped).
    assert kinds[0] == "FROM"
    assert "COPY" in kinds
    assert "CMD" in kinds
    # Every parsed line is a real Dockerfile instruction (no stray prose leaked in).
    assert set(kinds) <= {"FROM", "COPY", "CMD", "ENV", "RUN", "WORKDIR", "EXPOSE"}


def test_init_does_not_clobber_existing_dockerfile(tmp_path: Path) -> None:
    """A pre-existing Dockerfile is the user's to own — init leaves it untouched."""
    (tmp_path / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    assert _init_classic(tmp_path) == 0
    assert (tmp_path / "Dockerfile").read_text(encoding="utf-8") == "FROM scratch\n"


# -------------------------------------------------------- deploy --docker emits without booting


def test_deploy_docker_emits_same_recipe_without_booting(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`himmy deploy --docker` prints the ghcr recipe to stdout and never binds a socket.

    We fail the test if the supervised serve+worker group is ever entered — `--docker` is a
    pure text emit that must exit before any boot.
    """
    (tmp_path / "agent.yaml").write_text(
        "name: bot\ndescription: x\nprovider: stub\n", encoding="utf-8"
    )

    def _boom(*_a: object, **_k: object) -> None:  # pragma: no cover - must not run
        raise AssertionError("deploy --docker must not boot a server")

    monkeypatch.setattr(commands, "_serve_and_worker", _boom)

    args = argparse.Namespace(
        file=str(tmp_path / "agent.yaml"), agent=None, docker=True, channel="http"
    )
    assert commands.cmd_deploy(args) == 0
    out = capsys.readouterr().out
    assert f"FROM {commands.GHCR_IMAGE}:{__version__}" in out
    assert "COPY agent.yaml /app/agent.yaml" in out
    assert 'CMD ["himmy", "deploy"' in out
    # No build-time pip install — himmy is already in the base image.
    assert "pip install" not in out


def test_deploy_docker_uses_the_specs_own_filename(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """When the spec is named oddly, the COPY source matches it (dest stays /app/agent.yaml)."""
    spec = tmp_path / "my-bot.yaml"
    spec.write_text("name: bot\ndescription: x\nprovider: stub\n", encoding="utf-8")
    args = argparse.Namespace(file=str(spec), agent=None, docker=True, channel="http")
    assert commands.cmd_deploy(args) == 0
    out = capsys.readouterr().out
    assert "COPY my-bot.yaml /app/agent.yaml" in out
