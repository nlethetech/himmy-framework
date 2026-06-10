"""Record a :class:`~himmy.demovideo.models.DemoScript` into an MP4.

One Playwright-recorded clip per chapter (the player signals ``window.__sceneDone``),
then one ffmpeg concat into a social-ready H.264 MP4. Both heavy dependencies are
optional and lazily checked with actionable errors: ``playwright`` (plus a Chromium
download) and the ``ffmpeg`` binary.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

from himmy.core.errors import HimmyError
from himmy.demovideo.models import DemoScript

#: 1080p; passed as playwright's ViewportSize (a TypedDict — plain dict at runtime).
VIEWPORT: dict[str, int] = {"width": 1920, "height": 1080}


def load_script(workspace: Path) -> DemoScript:
    """Parse + validate ``<workspace>/script.json`` (fail-loud on a bad script)."""
    path = workspace / "script.json"
    if not path.exists():
        raise HimmyError(
            f"no script.json in {workspace} — run `himmy demo-video` first"
        )
    return DemoScript.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_script_js(workspace: Path, script: DemoScript) -> Path:
    """Serialize the script next to the player (file://-safe, no fetch needed)."""
    target = workspace / "script.js"
    payload = json.dumps(script.model_dump(), ensure_ascii=False)
    target.write_text(f"window.DEMO_SCRIPT = {payload};\n", encoding="utf-8")
    return target


def build_stitch_command(
    clips: list[Path], output: Path, *, fps: int = 30, crf: int = 18
) -> list[str]:
    """The ffmpeg concat command: normalize every clip and emit faststart H.264."""
    cmd: list[str] = ["ffmpeg", "-y", "-v", "warning"]
    for clip in clips:
        cmd += ["-i", str(clip)]
    chain = "".join(f"[{i}:v]" for i in range(len(clips)))
    cmd += [
        "-filter_complex",
        f"{chain}concat=n={len(clips)}:v=1:a=0,scale=1920:1080,fps={fps},format=yuv420p[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        "medium",
        "-movflags",
        "+faststart",
        str(output),
    ]
    return cmd


def record_chapters(
    workspace: Path,
    script: DemoScript,
    *,
    only: str | None = None,
    timeout_s: float = 180.0,
) -> list[Path]:
    """Record each chapter to ``<workspace>/clips/<n>_<id>.webm``; return clip order.

    With ``only``, just that chapter is re-recorded — the other clips are reused from
    disk (the iterate-on-one-scene loop). Raises :class:`HimmyError` when a reused
    clip is missing.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - import guard
        raise HimmyError(
            "demo-video rendering needs playwright: pip install playwright "
            "&& python -m playwright install chromium"
        ) from exc

    clips_dir = workspace / "clips"
    clips_dir.mkdir(exist_ok=True)
    write_script_js(workspace, script)
    player = (workspace / "player.html").resolve()
    ordered: list[Path] = []

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for i, chapter_id in enumerate(script.chapter_ids(), 1):
            clip = clips_dir / f"{i:02d}_{chapter_id}.webm"
            ordered.append(clip)
            if only is not None and chapter_id != only:
                if not clip.exists():
                    raise HimmyError(
                        f"--only {only}: missing previously recorded clip {clip.name} — "
                        "render once without --only first"
                    )
                continue
            ctx = browser.new_context(
                viewport=VIEWPORT,  # type: ignore[arg-type]  # ViewportSize TypedDict
                record_video_dir=str(clips_dir),
                record_video_size=VIEWPORT,  # type: ignore[arg-type]
            )
            page = ctx.new_page()
            page.goto(f"file://{player}?chapter={chapter_id}")
            page.wait_for_function(
                "window.__sceneDone === true", timeout=timeout_s * 1000
            )
            time.sleep(0.35)  # let the last frame breathe before the cut
            video = page.video
            ctx.close()
            if video is None:  # pragma: no cover - recording was requested above
                raise HimmyError(f"no video captured for chapter {chapter_id}")
            clip.unlink(missing_ok=True)
            Path(video.path()).rename(clip)
            print(f"  recorded {clip.name}")
        browser.close()
    return ordered


def stitch(clips: list[Path], output: Path, *, fps: int = 30, crf: int = 18) -> Path:
    """Concat the clips into ``output`` (H.264 MP4). Needs ffmpeg on PATH."""
    if shutil.which("ffmpeg") is None:
        raise HimmyError(
            "demo-video stitching needs ffmpeg on PATH (brew install ffmpeg)"
        )
    result = subprocess.run(  # noqa: S603 - argv is built from validated local paths
        build_stitch_command(clips, output, fps=fps, crf=crf),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise HimmyError(f"ffmpeg failed: {result.stderr.strip()[-500:]}")
    return output


def render(
    workspace: Path, *, only: str | None = None, output_name: str = "demo.mp4"
) -> Path:
    """The whole pipeline: load script → record chapters → stitch one MP4."""
    workspace = workspace.resolve()
    script = load_script(workspace)
    if not script.chapters:
        raise HimmyError("script.json has no chapters — nothing to record")
    clips = record_chapters(workspace, script, only=only)
    output = workspace / output_name
    stitch(clips, output)
    print(f"  wrote {output}")
    return output
