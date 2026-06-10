"""The ``cli_video`` skill: cinematic all-terminal product demos from real output.

Pairs with the ``himmy demo-video`` workflow (:mod:`himmy.demovideo`): scaffold a
workspace, fill ``script.json`` with chapters built from *captured* terminal output,
render an MP4. The skill is the know-how — and above all the doctrine: the film may
only replay what a real command actually printed.

Kept in its own module (rather than inline in the catalog) because it is long and
marketable on its own; ``BUILTIN_SKILLS`` imports it like any other entry.
"""

from __future__ import annotations

from himmy.skills.models import Skill, SkillExample

CLI_VIDEO = Skill(
    name="cli_video",
    description=(
        "Produce a cinematic, all-terminal product demo video (MP4) of a CLI "
        "system — from REAL captured output, never fabricated."
    ),
    when_to_use=(
        "the user wants a demo, launch, or marketing video of a CLI tool or "
        "terminal-driven system."
    ),
    tool_packs=["files", "code"],
    instructions=[
        "NEVER fabricate terminal output. First RUN the system's real commands "
        "(read-only or paper/sandbox surfaces only) and save each output "
        "verbatim; the video may only replay what actually printed. Long output "
        "may be truncated with a visible '…' — never altered or invented. If you "
        "were given captured output, use it character-for-character.",
        "Scaffold a workspace with `himmy demo-video <dir>`, then write the film "
        "into <dir>/script.json: a `boot` splash (brand mark + wordmark + 2-3 "
        "selling-point lines), 3-5 ENTRY chapters (each: one `head`, 1-2 `cmd` "
        "steps followed by their captured output as `lines`, one closing dim "
        "`note` carrying the selling point), and an `outro` (mark + install/repo "
        "lines). The scaffolded README documents every step type and style.",
        "Do NOT invent a JSON shape: read_file the scaffolded script.json FIRST "
        "and keep its exact structure — chapters[].steps[] where every step is "
        'an object with a "do" key — editing values only. Then write_file the '
        "whole edited script back to script.json (write_file needs BOTH `path` "
        "and `content`).",
        "Give the product a memorable mark: a small block-art glyph in `brand.art` "
        "with ONE accent color, and set `brand.prompt` to the same glyph family — "
        "the shell prompt IS the logo.",
        "Pace it like a film: commands type at human speed, hold 2-3 seconds "
        "after each key output so viewers can read it, total 60-110 seconds.",
        "Render with `himmy demo-video <dir> --render`; while iterating, "
        "re-record one scene with `--only <chapter-id>`. Verify before "
        "delivering: extract frames with ffmpeg and check them.",
        "Never run commands with live side effects (orders, sends, deletes, "
        "payments) to capture demo footage — read-only and paper modes only.",
    ],
    examples=[
        SkillExample(
            input="Make a launch video for our CLI tool.",
            action=(
                "run the tool's real commands and save the outputs; "
                "`himmy demo-video demo/`; read demo/script.json; write boot + "
                "3 entries + outro using ONLY the captured text; "
                "`himmy demo-video demo/ --render`; spot-check frames"
            ),
            note="capture first, script second — the film replays reality",
        ),
    ],
)
