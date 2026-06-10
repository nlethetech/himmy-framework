"""Cinematic all-terminal product demos, as data.

``himmy demo-video <dir>`` scaffolds a workspace; the author (human or the
``cli_video`` skill) fills ``script.json`` with chapters built from *captured real
output*; ``--render`` records each chapter in a headless terminal player and
stitches a social-ready MP4. See :mod:`himmy.demovideo.models` for the script
shape and the scaffolded README for the honesty rules.
"""

from himmy.demovideo.models import Brand, Chapter, DemoScript, Step
from himmy.demovideo.recorder import render
from himmy.demovideo.scaffold import scaffold

__all__ = ["Brand", "Chapter", "DemoScript", "Step", "render", "scaffold"]
