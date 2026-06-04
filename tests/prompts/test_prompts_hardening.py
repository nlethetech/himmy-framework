"""Additional prompts-kernel hardening tests (renderer edges + template loading).

Complements ``tests/prompts/test_prompts.py`` by pinning the corner cases that the
``{name}``-only renderer (TP-10) and ``PromptTemplate`` validation/caching (TP-13)
must satisfy in production: adjacent placeholders, dollar-sign + brace mixes inside
both templates AND values, non-str value coercion, multi-file merge precedence,
missing-file errors, empty/whitespace YAML, and mtime cache invalidation.

Offline-only: pure stdlib + pyyaml (a core dep). A logfire-gated test is skipped
when ``logfire`` is absent so the module stays green on the offline stack.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from himmy.core.errors import HimmyError
from himmy.services.prompts import (
    PromptManager,
    PromptTemplate,
    SystemPromptVariables,
    TaskPromptVariables,
)
from himmy.services.prompts.manager import _load_validated_yaml, _render

try:  # logfire is an optional observability dep; gate its test on presence
    import logfire  # noqa: F401

    _HAS_LOGFIRE = True
except Exception:  # pragma: no cover - import guard
    _HAS_LOGFIRE = False


# ----------------------------------------------------------------- _render edges
def test_render_adjacent_placeholders() -> None:
    """Two adjacent placeholders both substitute without bleeding into each other."""
    assert _render("{a}{b}", {"a": "X", "b": "Y"}) == "XY"


def test_render_repeated_placeholder() -> None:
    """A placeholder used twice is substituted at every occurrence."""
    assert _render("{n} and {n}", {"n": "5"}) == "5 and 5"


def test_render_dollar_prefixed_brace_substitutes_only_the_brace() -> None:
    """In '${name}' the '{name}' IS a placeholder; only it substitutes, the '$' stays.

    The renderer is a real ``{name}``-only formatter: the leading ``$`` is an
    ordinary literal character left untouched, while the adjacent ``{name}``
    brace group is a valid placeholder and is substituted when present in values.
    """
    assert _render("path ${HOME}/bin", {"HOME": "usr"}) == "path $usr/bin"
    # When the key is absent the whole '${name}' survives verbatim (no rewrite).
    assert _render("path ${HOME}/bin", {}) == "path ${HOME}/bin"


def test_render_value_containing_dollar_is_inserted_literally() -> None:
    """A substituted value that contains '$' is inserted as-is (no re-parsing)."""
    assert _render("Cost: {c}", {"c": "$5 {raw}"}) == "Cost: $5 {raw}"


def test_render_non_identifier_braces_left_alone() -> None:
    """Braces around non-identifier text (digits-first / spaces) are not placeholders."""
    assert _render("{1abc} {a b}", {"1abc": "x", "a b": "y"}) == "{1abc} {a b}"


def test_render_unknown_placeholder_preserved() -> None:
    """A valid-identifier placeholder with no matching value is left as {name}."""
    assert _render("{known}-{unknown}", {"known": "K"}) == "K-{unknown}"


def test_render_empty_value_substitutes_empty_string() -> None:
    """An explicitly-empty value substitutes to the empty string (still a hit)."""
    assert _render("[{x}]", {"x": ""}) == "[]"


def test_render_non_string_value_is_stringified() -> None:
    """A non-str value is coerced via str() at substitution time."""
    assert _render("n={x}", {"x": 7}) == "n=7"


def test_full_prompt_round_trip_preserves_literal_braces() -> None:
    """Rendering through PromptManager with a brace-bearing task keeps it intact."""
    mgr = PromptManager()
    rendered = mgr.get_task_prompt(
        TaskPromptVariables(task="Return the set {a, b, c} verbatim")
    )
    assert "{a, b, c}" in rendered


def test_system_prompt_value_with_braces_not_corrupted() -> None:
    """A persona value containing braces survives the system-prompt render."""
    mgr = PromptManager()
    rendered = mgr.get_system_prompt(
        SystemPromptVariables(role="analyst", persona="uses notation {x|y}")
    )
    assert "{x|y}" in rendered


# ----------------------------------------------------------------- PromptTemplate
def test_from_paths_later_file_wins_per_key(tmp_path: Path) -> None:
    """When merging multiple files, later files override on a per-(section,key) basis."""
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text('persona:\n  role: "A role"\n  background: "A bg"\n', encoding="utf-8")
    b.write_text('persona:\n  role: "B role"\n', encoding="utf-8")
    merged = PromptTemplate.from_paths([a, b])
    persona = merged.section("persona")
    assert persona["role"] == "B role"  # later file wins
    assert persona["background"] == "A bg"  # untouched key preserved


def test_from_paths_missing_file_raises(tmp_path: Path) -> None:
    """A non-existent template path raises a clear HimmyError."""
    with pytest.raises(HimmyError):
        PromptTemplate.from_paths([tmp_path / "does_not_exist.yaml"])


def test_section_returns_empty_for_unknown(tmp_path: Path) -> None:
    """Asking for an absent section returns an empty dict, not an error."""
    p = tmp_path / "t.yaml"
    p.write_text('persona:\n  role: "x"\n', encoding="utf-8")
    tmpl = PromptTemplate.from_paths([p])
    assert tmpl.section("nope") == {}


def test_empty_yaml_loads_as_empty_mapping(tmp_path: Path) -> None:
    """An empty/whitespace YAML file loads to an empty section map (no crash)."""
    p = tmp_path / "empty.yaml"
    p.write_text("   \n", encoding="utf-8")
    assert _load_validated_yaml(p) == {}


def test_invalid_yaml_raises_himmy_error(tmp_path: Path) -> None:
    """Malformed YAML raises a clear HimmyError (not a raw yaml error)."""
    p = tmp_path / "broken.yaml"
    p.write_text("persona: [unclosed\n", encoding="utf-8")
    with pytest.raises(HimmyError):
        _load_validated_yaml(p)


def test_top_level_scalar_rejected(tmp_path: Path) -> None:
    """A template whose top level is a scalar (not a mapping) is rejected."""
    p = tmp_path / "scalar.yaml"
    p.write_text("just a string\n", encoding="utf-8")
    with pytest.raises(HimmyError):
        _load_validated_yaml(p)


def test_cache_hit_returns_same_object(tmp_path: Path) -> None:
    """Two loads of an unchanged file return the identical cached object."""
    p = tmp_path / "c.yaml"
    p.write_text('persona:\n  role: "v"\n', encoding="utf-8")
    first = _load_validated_yaml(p)
    second = _load_validated_yaml(p)
    assert first is second


def test_cache_invalidated_on_mtime_change(tmp_path: Path) -> None:
    """A rewrite with a newer mtime causes a re-parse (no stale cache hit)."""
    p = tmp_path / "c.yaml"
    p.write_text('persona:\n  role: "old"\n', encoding="utf-8")
    first = _load_validated_yaml(p)
    assert first["persona"]["role"] == "old"
    new_mtime = p.stat().st_mtime + 42
    p.write_text('persona:\n  role: "new"\n', encoding="utf-8")
    os.utime(p, (new_mtime, new_mtime))
    second = _load_validated_yaml(p)
    assert second["persona"]["role"] == "new"
    assert second is not first


def test_int_keys_coerced_to_string(tmp_path: Path) -> None:
    """Non-string section/keys are coerced to str so the map stays str->str->str."""
    p = tmp_path / "k.yaml"
    # YAML parses bare numerics as ints; the loader must coerce keys to str.
    p.write_text('persona:\n  1: "numbered"\n', encoding="utf-8")
    data = _load_validated_yaml(p)
    assert data["persona"]["1"] == "numbered"


# ----------------------------------------------------------------- end-to-end render
def test_prompt_manager_uses_custom_template(tmp_path: Path) -> None:
    """A PromptManager built from a custom path renders that file's wording."""
    p = tmp_path / "custom.yaml"
    p.write_text('persona:\n  role: "CUSTOM {role} persona"\n', encoding="utf-8")
    mgr = PromptManager(template_paths=[p])
    out = mgr.get_system_prompt(SystemPromptVariables(role="quant"))
    assert "CUSTOM quant persona" in out


# ----------------------------------------------------------------- logfire (gated)
@pytest.mark.skipif(
    not _HAS_LOGFIRE, reason="requires the observability extra (logfire)"
)
def test_logfire_importable_when_extra_present() -> (
    None
):  # pragma: no cover - extra only
    """Sanity gate: when logfire is installed it imports cleanly (no offline assumption)."""
    import logfire

    assert hasattr(logfire, "configure")
