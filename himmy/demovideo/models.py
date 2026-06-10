"""The demo-video script model: a film described as data, not code.

A :class:`DemoScript` is the whole movie — brand identity plus an ordered list of
:class:`Chapter`\\ s, each a list of :class:`Step`\\ s the bundled terminal player
(``player.html``) performs: type a command, print captured output lines, draw the
brand mark, pause. The author writes ``script.json``; the recorder turns it into
clips and stitches an MP4. Nothing here generates content — every ``lines`` step
should carry output captured verbatim from real commands (the ``cli_video`` skill's
first rule).

Line shapes accepted by ``lines`` steps (matching the player's normalizer):

* ``"plain text"`` — one dim line;
* ``["text", "cls"]`` — one line, one style;
* ``[["seg a", "cls"], ["seg b", "cls"]]`` — one line, styled segments.

Styles are the player's fixed palette: ``cmd, out, out-bright, faint, gold, green,
crimson, snow, badge``.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: Step kinds the player implements. Anything else fails validation loudly.
StepKind = Literal[
    "art",  # draw the brand mark (brand.art), line by line
    "wordmark",  # the brand wordmark + accent period
    "tagline",  # the brand tagline, bright
    "head",  # a chapter rule: ── ENTRY <no> · <NAME> ── plus a dim tag line
    "cmd",  # prompt symbol + a command typed at human speed
    "typed",  # prompt symbol + text typed in a custom style (e.g. a '# beat' comment)
    "lines",  # captured output lines, revealed top-to-bottom
    "note",  # a dim '# selling point' comment line
    "pause",  # hold the frame
    "blank",  # an empty line
]


class Step(BaseModel):
    """One player action inside a chapter."""

    model_config = ConfigDict(extra="forbid")

    do: StepKind
    text: str = ""  # cmd / typed / note
    cls: str = "out"  # typed style
    cps: int = 27  # typing speed (chars/second) for cmd / typed
    lines: list[Any] = []  # lines payload (see module docstring for shapes)
    delay_ms: int = 120  # per-line reveal delay for lines
    ms: int = 1000  # pause duration
    pace_ms: int = 95  # per-line pace for art
    no: str = ""  # head: entry number ("01")
    name: str = ""  # head: entry name ("TOOLS & AUDIT")
    tag: str = ""  # head: the dim one-liner under the rule


class Chapter(BaseModel):
    """A scene recorded as one clip; chapters are stitched in order."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    centered: bool = False  # boot/outro center vertically; entries bottom-anchor
    steps: list[Step] = []


class Brand(BaseModel):
    """The film's identity: mark, wordmark, palette accent, chrome text."""

    model_config = ConfigDict(extra="forbid")

    name: str = "my product"
    wordmark: str = "my product"
    tagline: str = "what it is, in one line"
    prompt: str = "▲"  # the shell prompt symbol — make it a logo
    accent: str = "#E8294A"  # the single accent color (crimson by default)
    titlebar: str = "my product — zsh"
    titlebar_right: str = "$0.0000 / RUN"
    #: The ASCII/Unicode mark, as lines of [text, cls] segments (or plain strings).
    art: list[Any] = []


class DemoScript(BaseModel):
    """The whole film: brand + ordered chapters."""

    model_config = ConfigDict(extra="forbid")

    brand: Brand = Brand()
    chapters: list[Chapter] = []

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DemoScript:
        """Validate a parsed ``script.json`` mapping."""
        return cls.model_validate(raw)

    def chapter_ids(self) -> list[str]:
        """The clip order."""
        return [c.id for c in self.chapters]
