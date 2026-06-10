"""Scaffold a demo-video workspace: player + fonts + a starter script + the doctrine.

``himmy demo-video <dir>`` copies the bundled terminal player, the Fira Code fonts
(when the Studio static bundle ships them), a small working ``script.json`` the author
edits chapter-by-chapter, and a README carrying the honesty rules the ``cli_video``
skill enforces: capture real output first, replay it verbatim, never fabricate.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

_STARTER_SCRIPT: dict[str, Any] = {
    "brand": {
        "name": "my product",
        "wordmark": "my product",
        "tagline": "what it is, in one line",
        "prompt": "▲",
        "accent": "#E8294A",
        "titlebar": "my product — zsh",
        "titlebar_right": "$0.0000 / RUN",
        "art": [
            [["        ▲", "crimson"]],
            [["       ◢█◣", "snow"]],
            [["      ◢███◣", "snow"]],
            [["     ◢█████◣", "faint"]],
            [["    ◢███████◣", "faint"]],
            [["   ◢█████████◣", "faint"]],
        ],
    },
    "chapters": [
        {
            "id": "boot",
            "centered": True,
            "steps": [
                {"do": "cmd", "text": "my-product", "cps": 14},
                {"do": "art"},
                {"do": "pause", "ms": 350},
                {"do": "wordmark"},
                {"do": "pause", "ms": 800},
                {"do": "tagline"},
                {"do": "blank"},
                {
                    "do": "lines",
                    "lines": ["three selling points · as terminal lines · go here"],
                    "delay_ms": 200,
                },
                {"do": "pause", "ms": 3000},
            ],
        },
        {
            "id": "entry01",
            "steps": [
                {
                    "do": "head",
                    "no": "01",
                    "name": "THE FIRST THING",
                    "tag": "one line on why this chapter matters.",
                },
                {"do": "cmd", "text": "echo 'replace me with a REAL command you ran'"},
                {
                    "do": "lines",
                    "lines": [
                        [
                            "replace me with that command's REAL output, verbatim",
                            "out-bright",
                        ]
                    ],
                    "delay_ms": 150,
                },
                {"do": "pause", "ms": 1000},
                {
                    "do": "note",
                    "text": "# one dim selling-point line to close the chapter",
                },
                {"do": "pause", "ms": 2600},
            ],
        },
        {
            "id": "outro",
            "centered": True,
            "steps": [
                {"do": "cmd", "text": "pip install my-product", "cps": 18},
                {"do": "art", "pace_ms": 75},
                {"do": "blank"},
                {
                    "do": "lines",
                    "lines": [["github.com/you/my-product", "out-bright"]],
                    "delay_ms": 150,
                },
                {"do": "pause", "ms": 3200},
            ],
        },
    ],
}

_README = """\
# demo-video workspace

This folder renders a cinematic, all-terminal product demo (MP4) from `script.json`.

## The two rules (non-negotiable)

1. **Never fabricate terminal output.** Run the real command first, save what it
   printed, and put THAT in the script — verbatim. Long output may be truncated with
   a visible `…`, never altered.
2. **Never capture footage with live side effects.** Read-only and paper/sandbox
   surfaces only.

## Workflow

1. Run your product's real commands; keep each output.
2. Edit `script.json`: a `boot` splash (brand mark + selling points), 3–5 `ENTRY`
   chapters (one `head`, 1–2 `cmd` steps each followed by their captured output as
   `lines`, one closing `note`), and an `outro`. Keep the film 60–110 seconds.
3. Render: `himmy demo-video . --render`   (needs `playwright` + chromium + `ffmpeg`)
4. Iterate on one scene: `himmy demo-video . --render --only entry01`
5. QA frames: `ffmpeg -ss 12 -i demo.mp4 -frames:v 1 frame.png`

## script.json cheatsheet

Steps: `art` `wordmark` `tagline` `head` `cmd` `typed` `lines` `note` `pause` `blank`.
Line styles: `cmd out out-bright faint gold green crimson snow badge`.
A line is `"text"`, `["text","style"]`, or `[["seg","style"], ...]`.
Brand: `prompt` is your logo glyph (it IS the shell prompt), `accent` the one color.
"""


def scaffold(target: Path) -> list[Path]:
    """Create the workspace under ``target``; return the files written (idempotent —
    existing files are left untouched so a re-run never clobbers an edited script)."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    (target / "assets").mkdir(exist_ok=True)
    written: list[Path] = []

    def _write(path: Path, content: str) -> None:
        if path.exists():
            return
        path.write_text(content, encoding="utf-8")
        written.append(path)

    _write(
        target / "player.html",
        (Path(__file__).parent / "player.html").read_text(encoding="utf-8"),
    )
    _write(
        target / "script.json",
        json.dumps(_STARTER_SCRIPT, indent=2, ensure_ascii=False) + "\n",
    )
    _write(target / "README.md", _README)

    # Fira Code from the Studio static bundle, when this install ships it.
    fonts_src = Path(__file__).parents[1] / "api" / "_studio_static" / "fonts"
    for weight in ("Regular", "Medium", "SemiBold"):
        src = fonts_src / f"FiraCode-{weight}.woff2"
        dst = target / "assets" / src.name
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            written.append(dst)

    return written
