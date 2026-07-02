"""Contract test: docs/QUICKSTART.md section 4 must match real `himmy init` behavior.

The "Your first agent in 60 seconds" section documents a scaffold command, an
"Expected output" block, and the claim that you can edit the `role` and
`instructions` lines of the written ``agent.yaml``. On a real terminal plain
``himmy init`` is interactive (the wizard writes only ``agent.yaml`` and no
``role:`` line), so the doc must document ``--classic`` for the shown output —
otherwise an interactive reader sees something completely different.

These assertions pin the doc to what the CLI actually does, so the two can't
silently drift apart again.
"""

from __future__ import annotations

import re
from pathlib import Path

from himmy.cli.__main__ import main

_QUICKSTART = Path(__file__).resolve().parents[2] / "docs" / "QUICKSTART.md"


def _section_four() -> str:
    """Return the text of the '## 4. Your first agent ...' section."""
    text = _QUICKSTART.read_text(encoding="utf-8")
    start = text.index("## 4. ")
    end = text.index("\n## 5.", start)
    return text[start:end]


def test_doc_command_uses_classic_so_output_matches() -> None:
    """The documented scaffold command must be `himmy init my-agent --classic`.

    Plain `himmy init` is the interactive wizard on a TTY, which writes a single
    minimal agent.yaml — not the four files in the Expected-output block. Only the
    `--classic` form produces exactly what the doc shows.
    """
    section = _section_four()
    assert "himmy init my-agent --classic" in section
    # The bare command must not be presented as producing the four-file output.
    fenced = re.findall(r"```bash\n(.*?)\n```", section, re.DOTALL)
    assert fenced, "section 4 lost its command code block"
    assert fenced[0].strip() == "himmy init my-agent --classic"


def test_expected_output_block_matches_classic_scaffold(tmp_path: Path) -> None:
    """The 'Expected output' block lists exactly the files `--classic` writes."""
    section = _section_four()
    block = re.search(r"Expected output:\n\n```\n(.*?)\n```", section, re.DOTALL)
    assert block, "section 4 lost its Expected-output block"
    documented = block.group(1)

    # The documented files, in the documented order (agent spec + companions, then
    # the deploy-DX container front door written last).
    documented_files = [
        line.removeprefix("wrote my-agent/").strip()
        for line in documented.splitlines()
        if line.startswith("wrote ")
    ]
    assert documented_files == [
        "agent.yaml",
        "tools.py",
        "himmy.toml",
        "skills/my_skill.yaml",
        "Dockerfile",
    ]
    assert 'Next: himmy run -f my-agent/agent.yaml -p "hello"' in documented

    # Now run the real classic scaffold and confirm it writes exactly those files.
    target = tmp_path / "my-agent"
    assert main(["init", "--classic", str(target)]) == 0
    written = sorted(
        str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()
    )
    assert written == sorted(
        ["agent.yaml", "tools.py", "himmy.toml", "skills/my_skill.yaml", "Dockerfile"]
    )


def test_role_and_instructions_claim_holds_for_classic_yaml(tmp_path: Path) -> None:
    """The doc tells readers to edit `role`/`instructions`; classic yaml has both."""
    section = _section_four()
    assert "`role` and `instructions` lines" in section

    target = tmp_path / "my-agent"
    assert main(["init", "--classic", str(target)]) == 0
    agent_yaml = (target / "agent.yaml").read_text(encoding="utf-8")
    assert re.search(r"^role:", agent_yaml, re.MULTILINE)
    assert re.search(r"^instructions:", agent_yaml, re.MULTILINE)


def test_section_is_honest_about_interactive_wizard() -> None:
    """Section 4 must warn that plain `himmy init` is interactive on a terminal."""
    section = _section_four()
    # Mentions the interactive path and that it writes only agent.yaml.
    assert "interactive" in section.lower()
    assert "wrote my-agent/agent.yaml" in section
    # Mentions pressing Enter to accept defaults — the promised frictionless path.
    assert "Enter" in section
