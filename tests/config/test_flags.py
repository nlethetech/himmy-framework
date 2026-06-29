"""Tests for the canonical security-flag truthy/falsy parser (himmy.config.flags).

WHY: inconsistent truthy parsing (one site accepting ``on``, another not) is a
recurring *bypass class* — a posture kill-switch half-honored fails OPEN. These tests
(a) pin the accepted truthy/falsy vocabulary as a table, (b) prove the offline/
zero-config default is unchanged, and (c) GUARD against regression by asserting every
known security/posture flag is parsed through the canonical helper rather than via an
ad-hoc inline ``os.environ.get(...).lower() in (...)`` comparison.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from himmy.config.flags import (
    FALSY_TOKENS,
    TRUTHY_TOKENS,
    env_falsy,
    env_truthy,
    falsy,
    truthy,
)

# --------------------------------------------------------------------------- #
# Table tests: accepted truthy / falsy forms                                  #
# --------------------------------------------------------------------------- #

TRUTHY_FORMS = [
    "1",
    "true",
    "TRUE",
    "True",
    "yes",
    "YES",
    "on",
    "ON",
    "On",
    "y",
    "Y",
    "  on  ",  # surrounding whitespace trimmed
    "\tyes\n",
]

FALSY_FORMS_FOR_TRUTHY = [
    "",
    "   ",
    "0",
    "false",
    "no",
    "off",
    "n",
    "2",
    "maybe",
    "onn",
    "ye",
    "enable",
    "disabled",
]


@pytest.mark.parametrize("value", TRUTHY_FORMS)
def test_truthy_accepts_canonical_forms(value: str) -> None:
    assert truthy(value) is True


@pytest.mark.parametrize("value", FALSY_FORMS_FOR_TRUTHY)
def test_truthy_rejects_everything_else(value: str) -> None:
    assert truthy(value) is False


def test_truthy_none_and_empty_honor_default() -> None:
    assert truthy(None) is False
    assert truthy("") is False
    assert truthy(None, default=True) is True
    assert truthy("   ", default=True) is True
    # A *recognised* token always wins over the default in BOTH directions.
    assert truthy("off", default=True) is False
    assert truthy("on", default=False) is True


FALSY_FORMS = ["0", "false", "FALSE", "no", "NO", "off", "OFF", "n", "N", "  off  "]
NOT_FALSY_FORMS = ["", "   ", "1", "true", "yes", "on", "y", "garbage", "of", "nope"]


@pytest.mark.parametrize("value", FALSY_FORMS)
def test_falsy_accepts_canonical_off_forms(value: str) -> None:
    assert falsy(value) is True


@pytest.mark.parametrize("value", NOT_FALSY_FORMS)
def test_falsy_rejects_non_off_forms(value: str) -> None:
    # Unset / unrecognised must NOT count as an explicit "off" (fail-closed for a
    # default-ON switch: a typo cannot silently disable the guard).
    assert falsy(value) is False


def test_falsy_none_is_false() -> None:
    assert falsy(None) is False


def test_token_sets_are_disjoint() -> None:
    # A token can never be BOTH on and off — that would make a switch ambiguous.
    assert TRUTHY_TOKENS.isdisjoint(FALSY_TOKENS)


def test_truthy_superset_of_legacy_vocabulary() -> None:
    # Never narrower than the historical ad-hoc tuples, so no previously-honored value
    # silently becomes false after the canonicalisation.
    assert {"1", "true", "yes", "on"} <= TRUTHY_TOKENS
    assert {"0", "false", "no", "off"} <= FALSY_TOKENS


# --------------------------------------------------------------------------- #
# env_* wrappers                                                              #
# --------------------------------------------------------------------------- #


def test_env_truthy_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_TEST_FLAG", raising=False)
    assert env_truthy("HIMMY_TEST_FLAG") is False
    assert env_truthy("HIMMY_TEST_FLAG", default=True) is True
    monkeypatch.setenv("HIMMY_TEST_FLAG", "on")
    assert env_truthy("HIMMY_TEST_FLAG") is True
    monkeypatch.setenv("HIMMY_TEST_FLAG", "off")
    assert env_truthy("HIMMY_TEST_FLAG") is False


def test_env_falsy_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_TEST_FLAG", raising=False)
    assert env_falsy("HIMMY_TEST_FLAG") is False  # unset is NOT explicit-off
    monkeypatch.setenv("HIMMY_TEST_FLAG", "off")
    assert env_falsy("HIMMY_TEST_FLAG") is True
    monkeypatch.setenv("HIMMY_TEST_FLAG", "on")
    assert env_falsy("HIMMY_TEST_FLAG") is False
    monkeypatch.setenv("HIMMY_TEST_FLAG", "typo")
    assert env_falsy("HIMMY_TEST_FLAG") is False  # unrecognised stays in default state


# --------------------------------------------------------------------------- #
# Offline / zero-config default unchanged                                     #
# --------------------------------------------------------------------------- #


def test_offline_default_is_multi_tenant_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_MULTI_TENANT", raising=False)
    monkeypatch.delenv("HIMMY_AUTH_MODE", raising=False)
    from himmy.api.auth.context import is_multi_tenant

    assert is_multi_tenant() is False


def test_offline_default_studio_auth_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HIMMY_STUDIO_AUTH", raising=False)
    from himmy.api.routers.studio_common import _studio_auth_off

    # Unset => Studio guards stay ON (auth NOT off).
    assert _studio_auth_off() is False


def test_studio_auth_typo_keeps_guards_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # A mistyped value must NOT disable the operator console (fail-closed).
    monkeypatch.setenv("HIMMY_STUDIO_AUTH", "offf")
    from himmy.api.routers.studio_common import _studio_auth_off

    assert _studio_auth_off() is False
    monkeypatch.setenv("HIMMY_STUDIO_AUTH", "off")
    assert _studio_auth_off() is True


def test_multi_tenant_on_token_is_honored(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exact half-honored-posture bug this WP closes: ``on`` must engage the posture.
    monkeypatch.delenv("HIMMY_AUTH_MODE", raising=False)
    from himmy.api.auth.context import is_multi_tenant

    for tok in ("on", "yes", "y", "1", "true"):
        monkeypatch.setenv("HIMMY_MULTI_TENANT", tok)
        assert is_multi_tenant() is True, tok
    for tok in ("", "off", "no", "0", "false"):
        monkeypatch.setenv("HIMMY_MULTI_TENANT", tok)
        assert is_multi_tenant() is False, tok


# --------------------------------------------------------------------------- #
# GUARD: no security flag is parsed ad-hoc                                     #
# --------------------------------------------------------------------------- #

#: Security / posture env flags that MUST route through himmy.config.flags. If you add a
#: new one, parse it via env_truthy / env_falsy and list it here.
SECURITY_FLAGS = {
    "HIMMY_MULTI_TENANT",
    "HIMMY_STUDIO_AUTH",
    "HIMMY_ALLOW_UNAUTHENTICATED",
    "HIMMY_ALLOW_OPERATOR_SPEC_TOOLS",
    "HIMMY_SANITIZE_SPEC_STRIP",
    "HIMMY_API_KEY_REVOCATION_FAIL_CLOSED",
    "HIMMY_OIDC_SUBJECT_SCOPED",
    "HIMMY_SCHEDULER_SINGLE_NODE_ACK",
    "HIMMY_SCHEDULER_REQUIRE_ACK",
    "HIMMY_ROUTINES_SCHEDULER",
}

#: Modules whose boolean security/posture flags must be canonicalised.
_REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDED_MODULES = [
    _REPO_ROOT / "himmy" / "api" / "auth" / "context.py",
    _REPO_ROOT / "himmy" / "api" / "auth" / "apikey.py",
    _REPO_ROOT / "himmy" / "api" / "auth" / "oidc.py",
    _REPO_ROOT / "himmy" / "api" / "app.py",
    _REPO_ROOT / "himmy" / "api" / "routers" / "studio_common.py",
    _REPO_ROOT / "himmy" / "api" / "routers" / "studio_routines.py",
    _REPO_ROOT / "himmy" / "api" / "scheduler_leader.py",
    _REPO_ROOT / "himmy" / "config" / "spec_sanitizer.py",
    # The CLI worker is a DOCUMENTED parity path: it gates the routine scheduler on
    # HIMMY_ROUTINES_SCHEDULER / HIMMY_SCHEDULER_REQUIRE_ACK and is the primary surface
    # routines actually fire on, so its reads must be canonicalised too.
    _REPO_ROOT / "himmy" / "cli" / "commands.py",
]

# Matches an inline truthy/falsy comparison on a security flag, e.g.:
#   os.environ.get("HIMMY_MULTI_TENANT", "").lower() in ("1", "true", ...)
# We require the flag name to appear in the SAME os.environ.get/os.getenv call as a
# subsequent ``in (`` / ``in {`` membership test (the ad-hoc pattern), within ~200 chars.
_ADHOC = re.compile(
    r"""os\.(?:environ\.get|getenv)\(\s*["'](?P<flag>HIMMY_[A-Z_]+)["'][^)]*\)"""
    r"""[^\n]{0,200}?\.lower\(\)[^\n]{0,80}?\bin\b\s*[\(\{]""",
    re.VERBOSE,
)


def test_no_security_flag_parsed_ad_hoc() -> None:
    """Fail if any security flag is truthy-compared inline instead of via env_truthy/falsy.

    This is the regression guard: if a future edit reintroduces
    ``os.environ.get("HIMMY_MULTI_TENANT", "").lower() in ("1", ...)`` the divergent
    vocabulary it implies (the recurring half-honored-posture bug) is caught here.
    """
    offenders: list[str] = []
    for path in GUARDED_MODULES:
        text = path.read_text(encoding="utf-8")
        for m in _ADHOC.finditer(text):
            flag = m.group("flag")
            if flag in SECURITY_FLAGS:
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.name}:{line} ad-hoc parse of {flag}")
    assert not offenders, (
        "security flags must be parsed via himmy.config.flags.env_truthy/env_falsy, "
        "found ad-hoc inline comparisons:\n  " + "\n  ".join(offenders)
    )


def test_known_security_flags_have_no_raw_truthy_tuple() -> None:
    """Belt-and-suspenders: none of the guarded modules embed a raw flag-name truthy tuple.

    Catches the looser shape where a flag's value is bound to a local then compared, by
    asserting each guarded module that mentions a security flag also imports the canonical
    helper (env_truthy or env_falsy) — i.e. SOMETHING routes it canonically.
    """
    for path in GUARDED_MODULES:
        text = path.read_text(encoding="utf-8")
        mentions_flag = any(flag in text for flag in SECURITY_FLAGS)
        if not mentions_flag:
            continue
        assert ("env_truthy" in text) or ("env_falsy" in text), (
            f"{path.name} references a security flag but never imports the canonical "
            "env_truthy/env_falsy parser"
        )
