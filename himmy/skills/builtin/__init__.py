"""The built-in skill catalog, ``BUILTIN_SKILLS``.

Seed skills defined against the built-in tool packs. Tier 2 adds YAML authoring + project
discovery on top of this catalog; the data shape is identical either way (each entry is a
validated :class:`~himmy.skills.models.Skill`).
"""

from __future__ import annotations

from himmy.skills.builtin.cli_video import CLI_VIDEO
from himmy.skills.models import Skill, SkillExample

BUILTIN_SKILLS: dict[str, Skill] = {
    "web_research": Skill(
        name="web_research",
        description="Find and synthesize facts from the open web, with citations.",
        when_to_use="the question needs current or external facts you cannot recall.",
        tool_packs=["web"],
        instructions=[
            "Search before answering; never guess a fact you can look up.",
            "Open the most relevant source and ground your answer in it.",
            "Cite the URL(s) you used so the answer is verifiable.",
        ],
    ),
    "data_analysis": Skill(
        name="data_analysis",
        description="Answer questions from a SQL database, accurately.",
        when_to_use="the answer lives in a connected SQLite/Postgres database.",
        tool_packs=["data"],
        instructions=[
            "Call sql_schema first if you do not already know the tables and columns.",
            "If a filter returns no rows, read the hint and re-run with a real value.",
            "Report the exact value from the result — do not estimate.",
        ],
        examples=[
            SkillExample(
                input="How many chickens are there?",
                action="call sql_schema, then sql_query "
                "SELECT quantity FROM animals WHERE name='chickens'",
                note="filter on the exact stored value, not a guess",
            ),
        ],
    ),
    "research_writer": Skill(
        name="research_writer",
        description="Research a topic on the web and write a tight, cited summary.",
        when_to_use="the user wants a short written brief on an external topic.",
        requires_skills=["web_research", "summarize"],
        instructions=[
            "First gather facts with web_research, then summarize them faithfully.",
        ],
    ),
    "file_ops": Skill(
        name="file_ops",
        description="Read and write files under the sandboxed workspace root.",
        when_to_use="the task involves reading from or writing to local files.",
        tool_packs=["files"],
        instructions=[
            "List the directory before assuming a file exists.",
            "Read a file fully before editing or summarizing it.",
        ],
    ),
    "summarize": Skill(
        name="summarize",
        description="Produce a faithful, concise summary of provided material.",
        when_to_use="the user wants a shorter, faithful version of given content.",
        instructions=[
            "Preserve the source's claims and numbers; never invent detail.",
            "Lead with the single most important point, then supporting points.",
        ],
    ),
    "knowledge_base": Skill(
        name="knowledge_base",
        description="Build and query the agent's own knowledge base (RAG).",
        when_to_use="the answer should come from previously ingested documents.",
        tool_packs=["knowledge"],
        instructions=[
            "Search the knowledge base before answering from memory.",
            "Ground the answer in retrieved passages and say when nothing matched.",
        ],
    ),
    "python_compute": Skill(
        name="python_compute",
        description="Compute exact results by running Python in a sandbox.",
        when_to_use="the task needs real computation, not an estimate.",
        tool_packs=["code"],
        instructions=[
            "For any non-trivial arithmetic or data work, run Python — do not guess.",
            "Print the final result so it is captured.",
        ],
    ),
    "nepal_brief": Skill(
        name="nepal_brief",
        description="Answer Nepal-specific questions: BS dates, NPR, NRB forex.",
        when_to_use="the question involves Nepali dates, currency, or NRB rates.",
        tool_packs=["nepal"],
        instructions=[
            "Use the Bikram Sambat calendar tool for any date conversion.",
            "Format money as NPR and report the exact figure from the tool.",
        ],
    ),
    "clarify": Skill(
        name="clarify",
        description="Ask before acting when a request is ambiguous or under-specified.",
        when_to_use=(
            "a request could be read more than one way, omits a needed detail "
            "(which item, how many, which of several matching records), or would "
            "create / change / delete data."
        ),
        instructions=[
            "Before taking ANY action — especially one that records, modifies, or "
            "deletes data — check that the request is fully specified.",
            "If something material is ambiguous or missing (e.g. which item/breed/"
            "account, how many, which of several matching records, or an unstated "
            "assumption), STOP and ask the user ONE concise clarifying question. Do "
            "NOT guess a default and do NOT proceed.",
            "When you ask, briefly state the options or what already exists (e.g. the "
            "current records) so the user can answer in one line.",
            "Make no tool call that changes anything until the user has answered. "
            "Once they answer, carry out the now-unambiguous request.",
            "Only skip the question when the request is genuinely unambiguous or the "
            "user has already supplied the missing detail.",
        ],
        examples=[
            SkillExample(
                input="Add 10 ducks to the farm.",
                action=(
                    "ask: 'Which breed, and to which flock? You currently have 12 "
                    "Khaki Campbell ducks — add to that flock, or start a new breed?' "
                    "— then wait for the answer before recording anything"
                ),
                note="quantity is clear but the breed/flock is not; never assume",
            ),
        ],
    ),
    "cli_video": CLI_VIDEO,
}

__all__ = ["BUILTIN_SKILLS"]
