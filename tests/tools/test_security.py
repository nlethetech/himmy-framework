"""Unit tests for the tools-kernel HTTP security helpers (SSRF / traversal / redaction).

These pin the *primitives* the ToolService relies on for its security guarantees,
independently of dispatch wiring: ``safe_path_arg`` / ``build_safe_path`` reject
path traversal and injection, ``assemble_url`` pins host+scheme, and
``redact_mapping`` / ``is_sensitive_key`` strip secrets before they reach events.
Offline-only: pure stdlib, no network or optional deps.
"""

from __future__ import annotations

import pytest

from himmy.services.tools.security import (
    ALLOWED_HTTP_METHODS,
    REDACTED,
    ToolSecurityError,
    assemble_url,
    build_safe_path,
    is_sensitive_key,
    redact_mapping,
    safe_path_arg,
)


# ----------------------------------------------------------------- safe_path_arg
def test_safe_path_arg_encodes_plain_value() -> None:
    """A benign value passes through percent-encoded (nothing left raw)."""
    assert safe_path_arg("123") == "123"
    assert safe_path_arg("a b") == "a%20b"
    assert safe_path_arg("a&b=c") == "a%26b%3Dc"


def test_safe_path_arg_rejects_forward_slash() -> None:
    """A '/' in a path arg would let it traverse to a sibling path segment."""
    with pytest.raises(ToolSecurityError):
        safe_path_arg("../../admin")


def test_safe_path_arg_rejects_backslash() -> None:
    """A backslash is rejected (Windows-style separator / normalization games)."""
    with pytest.raises(ToolSecurityError):
        safe_path_arg("a\\b")


def test_safe_path_arg_rejects_query_and_fragment() -> None:
    """A '?' or '#' would inject a query string / fragment into the URL."""
    with pytest.raises(ToolSecurityError):
        safe_path_arg("123?admin=true")
    with pytest.raises(ToolSecurityError):
        safe_path_arg("123#frag")


def test_safe_path_arg_rejects_control_characters() -> None:
    """CR/LF and other control chars are rejected (header/log injection)."""
    with pytest.raises(ToolSecurityError):
        safe_path_arg("abc\r\nHost: evil")
    with pytest.raises(ToolSecurityError):
        safe_path_arg("a\x00b")


def test_safe_path_arg_rejects_bare_dot_segments() -> None:
    """A lone '.' or '..' segment is a relative-path escape and is rejected."""
    with pytest.raises(ToolSecurityError):
        safe_path_arg("..")
    with pytest.raises(ToolSecurityError):
        safe_path_arg(".")


def test_safe_path_arg_encoded_dotdot_in_longer_value_is_kept() -> None:
    """'..' embedded in a longer value is percent-encoded, not a separator escape.

    The forbidden-char set blocks the dangerous separators; a value like
    ``v..2`` has no separator, so it is safely encoded rather than rejected.
    """
    assert safe_path_arg("v..2") == "v..2"


# ----------------------------------------------------------------- build_safe_path
def test_build_safe_path_only_consumes_referenced_keys() -> None:
    """Only placeholders in the template are substituted; extras are encoded but unused."""
    out = build_safe_path("/c/{cid}/orders", {"cid": "A B", "ignored": "x"})
    assert out == "/c/A%20B/orders"


def test_build_safe_path_missing_key_raises_keyerror() -> None:
    """A referenced placeholder with no arg raises KeyError (caller -> INVALID_REQUEST)."""
    with pytest.raises(KeyError):
        build_safe_path("/c/{cid}", {})


def test_build_safe_path_propagates_security_error() -> None:
    """A traversal payload in a substituted value surfaces as ToolSecurityError."""
    with pytest.raises(ToolSecurityError):
        build_safe_path("/c/{cid}", {"cid": "../escape"})


# ----------------------------------------------------------------- assemble_url
def test_assemble_url_joins_and_keeps_host() -> None:
    """A normal join keeps the configured scheme+host."""
    url = assemble_url("https://api.example.com", "customers/1")
    assert url == "https://api.example.com/customers/1"


def test_assemble_url_normalizes_slashes() -> None:
    """Trailing/leading slashes collapse to a single separator."""
    assert assemble_url("https://h/", "/p") == "https://h/p"


def test_assemble_url_rejects_non_http_scheme() -> None:
    """A non-http(s) base scheme (file/ftp) is rejected outright (SSRF surface)."""
    with pytest.raises(ToolSecurityError):
        assemble_url("file:///etc", "passwd")


def test_assemble_url_host_pin_holds_for_path_only_input() -> None:
    """An already-encoded path can never change the netloc (host pin)."""
    # %2F is an encoded slash; it must not be decoded into a host pivot.
    url = assemble_url("https://api.example.com", "items/a%2Fb")
    assert url.startswith("https://api.example.com/")


# ----------------------------------------------------------------- method allow-list
def test_method_allow_list_contents() -> None:
    """The allow-list contains standard safe/CRUD methods and excludes TRACE/CONNECT."""
    assert "GET" in ALLOWED_HTTP_METHODS
    assert "DELETE" in ALLOWED_HTTP_METHODS
    assert "TRACE" not in ALLOWED_HTTP_METHODS
    assert "CONNECT" not in ALLOWED_HTTP_METHODS


# ----------------------------------------------------------------- redaction
def test_is_sensitive_key_hint_matching() -> None:
    """Keys whose name contains a secret-hint substring are flagged (case-insensitive)."""
    assert is_sensitive_key("password")
    assert is_sensitive_key("API_KEY")
    assert is_sensitive_key("user_session_token")
    assert is_sensitive_key("Authorization")
    assert not is_sensitive_key("customer_id")
    assert not is_sensitive_key("limit")


def test_is_sensitive_key_explicit_extra() -> None:
    """An explicit extra key is flagged even without a hint substring."""
    assert is_sensitive_key("ssn", extra=["ssn"])
    assert is_sensitive_key("SSN", extra=["ssn"])  # case-insensitive
    assert not is_sensitive_key("ssn")


def test_redact_mapping_replaces_only_sensitive_values() -> None:
    """Sensitive values are replaced; benign values pass through unchanged."""
    out = redact_mapping({"user": "bob", "password": "hunter2", "limit": 5})
    assert out == {"user": "bob", "password": REDACTED, "limit": 5}


def test_redact_mapping_honors_extra_keys() -> None:
    """A tool-declared sensitive_arg_name is redacted via extra_keys."""
    out = redact_mapping({"account": "123"}, extra_keys=["account"])
    assert out["account"] == REDACTED


def test_redact_mapping_returns_a_copy() -> None:
    """Redaction never mutates the caller's dict (the live args stay intact)."""
    original = {"password": "hunter2"}
    out = redact_mapping(original)
    assert original["password"] == "hunter2"
    assert out is not original


def test_redact_mapping_recurses_into_nested_dict() -> None:
    """A secret buried inside a benign container key is masked, not just top-level."""
    args = {
        "method": "POST",
        "json_body": {"user": "bob", "config": {"api_key": "sk-live-DEEP"}},
    }
    out = redact_mapping(args)
    # The deeply nested credential is redacted; benign siblings pass through.
    assert out["json_body"]["config"]["api_key"] == REDACTED
    assert out["json_body"]["user"] == "bob"
    # And the caller's nested dict is untouched (deep, not shallow, copy).
    assert args["json_body"]["config"]["api_key"] == "sk-live-DEEP"


def test_redact_mapping_recurses_into_lists() -> None:
    """Secrets inside list elements (and lists nested in dicts) are masked."""
    args = {"json_body": {"items": [{"password": "p1"}, {"plain": "keep"}]}}
    out = redact_mapping(args)
    assert out["json_body"]["items"][0]["password"] == REDACTED
    assert out["json_body"]["items"][1]["plain"] == "keep"


def test_redact_mapping_honors_custom_placeholder() -> None:
    """The placeholder override (e.g. the approvals UI's marker) flows through nesting."""
    out = redact_mapping(
        {"body": {"token": "abc"}}, placeholder="••••"
    )
    assert out["body"]["token"] == "••••"
