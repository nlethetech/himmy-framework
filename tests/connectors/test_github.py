"""Contract tests for the GitHub connector — fully offline (mocked transport).

GitHub is an EXTERNAL service: end-to-end verification against the real API needs a live
PAT (or App installation token) + a repository and cannot run on this machine. These tests
therefore CONTRACT-test the connector against a mocked ``httpx`` transport and a synthetic
token: they assert the exact REST calls, paths, query params, and JSON payloads it would
make; that the ``Authorization: Bearer`` header is present-on-the-wire but
redacted-in-logs/errors; that list endpoints follow the ``Link`` header (pagination); that
the repo allowlist drops unlisted repositories (default-deny) for reads AND writes; that
404 / 403 / rate-limit responses become clean, token-free errors; and that side-effecting
calls are idempotent (a repeat mutates exactly once). What still needs a real token + repo
to verify is marked ``live-pending`` in the module docstring.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from himmy.config.secrets import configure_secrets
from himmy.connectors.github import (
    GITHUB_TOKEN_SECRET,
    GitHubClient,
    GitHubError,
    GitHubOutboundConnector,
    GitHubRateLimitError,
)
from himmy.connectors.sdk import AuditSink, ConnectorContext, ConnectorError
from himmy.services.tools.registry import ToolRegistry
from himmy.services.tools.security import ToolSecurityError
from tests.conftest import run_async


class _MemSecrets:
    """In-memory secret provider for tests (installed via configure_secrets)."""

    def __init__(self, values: dict[str, str]) -> None:
        self._values = dict(values)

    def get(self, name: str) -> str | None:
        return self._values.get(name)


class _RecordingAudit:
    def __init__(self) -> None:
        self.events: list[Any] = []

    def record(self, event: Any) -> None:
        self.events.append(event)


_TOKEN = "ghp_SUPER_SECRET_token_value"  # a fake PAT (test only)


def _mock_transport(app: Any) -> Any:
    import httpx

    return httpx.MockTransport(app)


def _connector(
    app: Any,
    *,
    allowed_repos: list[str] | None = ("octo/repo",),
    default_repo: str | None = None,
    requires_approval: bool = False,
    audit: _RecordingAudit | None = None,
) -> GitHubOutboundConnector:
    ctx = ConnectorContext(audit=AuditSink(audit.record)) if audit else None
    return GitHubOutboundConnector(
        allowed_repos=list(allowed_repos) if allowed_repos is not None else None,
        default_repo=default_repo,
        requires_approval=requires_approval,
        client_factory=lambda t: GitHubClient(t, transport=_mock_transport(app)),
        context=ctx,
    )


def _registered(conn: GitHubOutboundConnector) -> ToolRegistry:
    registry = ToolRegistry()
    conn.register_tools(registry)
    return registry


# ------------------------------------------------------------- tool registration


def test_registers_full_tool_surface() -> None:
    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        conn = GitHubOutboundConnector(allowed_repos=["octo/repo"])
        registry = ToolRegistry()
        names = conn.register_tools(registry)
        assert set(names) == {
            "github_list_issues",
            "github_create_issue",
            "github_comment_issue",
            "github_list_pull_requests",
            "github_create_pull_request",
            "github_comment_pull_request",
            "github_review_pull_request",
            "github_get_repo",
            "github_get_file",
        }
    finally:
        configure_secrets(None)


def test_read_tools_are_read_only_and_writes_require_approval_by_default() -> None:
    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        conn = GitHubOutboundConnector(allowed_repos=["octo/repo"])  # default approval
        registry = ToolRegistry()
        conn.register_tools(registry)
        # Reads never require approval and are flagged read-only.
        for read_tool in ("github_list_issues", "github_get_repo", "github_get_file"):
            definition = registry.get(read_tool)
            assert definition.read_only is True
            assert definition.requires_approval is False
        # Mutations default to human-in-the-loop approval.
        for write_tool in (
            "github_create_issue",
            "github_comment_issue",
            "github_create_pull_request",
            "github_review_pull_request",
        ):
            definition = registry.get(write_tool)
            assert definition.read_only is False
            assert definition.requires_approval is True
    finally:
        configure_secrets(None)


def test_capability_gated_on_token() -> None:
    """Without the token the connector is unavailable and registers no tools."""
    configure_secrets(_MemSecrets({}))
    try:
        conn = GitHubOutboundConnector(allowed_repos=["octo/repo"])
        assert conn.available() is False
        registry = ToolRegistry()
        assert conn.register_tools(registry) == []  # skipped cleanly, no crash
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- issues: list


def test_list_issues_sends_bearer_auth_and_filters_out_prs() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("Authorization")
        seen["accept"] = request.headers.get("Accept")
        seen["version"] = request.headers.get("X-GitHub-Api-Version")
        return httpx.Response(
            200,
            json=[
                {
                    "number": 1,
                    "title": "a bug",
                    "state": "open",
                    "html_url": "u1",
                    "labels": [{"name": "bug"}],
                },
                # A PR shows up in the issues list; it must be filtered out.
                {
                    "number": 2,
                    "title": "a pr",
                    "state": "open",
                    "html_url": "u2",
                    "pull_request": {"url": "p"},
                },
            ],
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_list_issues")(
                {"repo": "octo/repo", "state": "open", "labels": "bug"}
            )
        )
        assert seen["method"] == "GET"
        assert seen["auth"] == f"Bearer {_TOKEN}"
        assert seen["accept"] == "application/vnd.github+json"
        assert seen["version"]  # the api-version header is pinned
        assert "/repos/octo/repo/issues" in seen["url"]
        assert "state=open" in seen["url"] and "labels=bug" in seen["url"]
        assert "per_page=100" in seen["url"]
        assert out["count"] == 1  # the PR was filtered out
        assert out["issues"][0]["number"] == 1
        assert out["issues"][0]["labels"] == ["bug"]
    finally:
        configure_secrets(None)


def test_list_issues_follows_link_header_pagination() -> None:
    import httpx

    calls = {"n": 0}

    def app(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if "page=2" in str(request.url) or calls["n"] == 2:
            return httpx.Response(
                200,
                json=[{"number": 3, "title": "p2", "state": "open", "html_url": "u3"}],
            )
        # First page advertises a next link.
        return httpx.Response(
            200,
            json=[{"number": 1, "title": "p1", "state": "open", "html_url": "u1"}],
            headers={
                "Link": (
                    "<https://api.github.com/repos/octo/repo/issues?per_page=100&page=2>; "
                    'rel="next", '
                    "<https://api.github.com/repos/octo/repo/issues?per_page=100&page=5>; "
                    'rel="last"'
                )
            },
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_list_issues")({"repo": "octo/repo"})
        )
        assert calls["n"] == 2  # followed the Link to the second page
        assert [i["number"] for i in out["issues"]] == [1, 3]
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- issues: create


def test_create_issue_posts_expected_body_and_returns_summary() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["method"] = request.method
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(
            201,
            json={
                "number": 7,
                "title": "new bug",
                "state": "open",
                "html_url": "https://github.com/octo/repo/issues/7",
                "labels": [{"name": "bug"}],
            },
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_create_issue")(
                {
                    "repo": "octo/repo",
                    "title": "new bug",
                    "body": "it broke",
                    "labels": ["bug"],
                }
            )
        )
        assert seen["method"] == "POST"
        assert seen["url"].endswith("/repos/octo/repo/issues")
        assert seen["body"] == {
            "title": "new bug",
            "body": "it broke",
            "labels": ["bug"],
        }
        assert out["number"] == 7
        assert out["repo"] == "octo/repo"
    finally:
        configure_secrets(None)


def test_create_issue_is_idempotent_across_repeat() -> None:
    """The same (repo, title, body) opens exactly one issue across a repeated tool call."""
    import httpx

    hits = {"n": 0}

    def app(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        return httpx.Response(
            201, json={"number": hits["n"], "title": "dup", "state": "open"}
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        args = {"repo": "octo/repo", "title": "dup", "body": "same"}
        first = run_async(registry.handler_for("github_create_issue")(dict(args)))
        second = run_async(registry.handler_for("github_create_issue")(dict(args)))
        assert first == second
        assert hits["n"] == 1  # the effect fired exactly once
    finally:
        configure_secrets(None)


def test_create_issue_requires_title() -> None:
    import httpx

    def app(
        request: httpx.Request,
    ) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(201, json={})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        with pytest.raises(ToolSecurityError, match="title"):
            run_async(
                registry.handler_for("github_create_issue")({"repo": "octo/repo"})
            )
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- comments


def test_comment_issue_posts_to_comments_endpoint() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(
            201, json={"id": 555, "html_url": "https://github.com/octo/repo/issues/4#c"}
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_comment_issue")(
                {"repo": "octo/repo", "number": 4, "body": "thanks!"}
            )
        )
        assert seen["url"].endswith("/repos/octo/repo/issues/4/comments")
        assert seen["body"] == {"body": "thanks!"}
        assert out["comment_id"] == 555
        assert out["number"] == 4
    finally:
        configure_secrets(None)


def test_comment_pr_uses_same_issue_comment_endpoint_and_is_idempotent() -> None:
    import httpx

    hits = {"n": 0}

    def app(request: httpx.Request) -> httpx.Response:
        hits["n"] += 1
        assert request.url.path.endswith("/issues/9/comments")  # PR comment == issue
        return httpx.Response(201, json={"id": hits["n"]})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        args = {"repo": "octo/repo", "number": 9, "body": "LGTM"}
        run_async(registry.handler_for("github_comment_pull_request")(dict(args)))
        run_async(registry.handler_for("github_comment_pull_request")(dict(args)))
        assert hits["n"] == 1  # idempotent
    finally:
        configure_secrets(None)


def test_comment_requires_positive_integer_number() -> None:
    import httpx

    def app(
        request: httpx.Request,
    ) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(201, json={})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        with pytest.raises(ToolSecurityError, match="integer `number`"):
            run_async(
                registry.handler_for("github_comment_issue")(
                    {"repo": "octo/repo", "number": "not-a-number", "body": "x"}
                )
            )
        with pytest.raises(ToolSecurityError, match="must be positive"):
            run_async(
                registry.handler_for("github_comment_issue")(
                    {"repo": "octo/repo", "number": 0, "body": "x"}
                )
            )
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- pull requests


def test_list_pull_requests_hits_pulls_endpoint_with_base_filter() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json=[
                {
                    "number": 11,
                    "title": "feat",
                    "state": "open",
                    "html_url": "u",
                    "head": {"ref": "feature"},
                    "base": {"ref": "main"},
                    "draft": False,
                }
            ],
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_list_pull_requests")(
                {"repo": "octo/repo", "state": "open", "base": "main"}
            )
        )
        assert "/repos/octo/repo/pulls" in seen["url"]
        assert "base=main" in seen["url"]
        assert out["pull_requests"][0]["head"] == "feature"
        assert out["pull_requests"][0]["base"] == "main"
    finally:
        configure_secrets(None)


def test_create_pull_request_posts_expected_body() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(
            201,
            json={
                "number": 21,
                "title": "feat: thing",
                "state": "open",
                "html_url": "u",
                "head": {"ref": "feature"},
                "base": {"ref": "main"},
                "draft": True,
            },
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_create_pull_request")(
                {
                    "repo": "octo/repo",
                    "title": "feat: thing",
                    "head": "feature",
                    "base": "main",
                    "body": "does the thing",
                    "draft": True,
                }
            )
        )
        assert seen["url"].endswith("/repos/octo/repo/pulls")
        assert seen["body"] == {
            "title": "feat: thing",
            "head": "feature",
            "base": "main",
            "body": "does the thing",
            "draft": True,
        }
        assert out["number"] == 21
        assert out["draft"] is True
    finally:
        configure_secrets(None)


def test_create_pull_request_requires_head_and_base() -> None:
    import httpx

    def app(
        request: httpx.Request,
    ) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(201, json={})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        with pytest.raises(ToolSecurityError, match="`head`"):
            run_async(
                registry.handler_for("github_create_pull_request")(
                    {"repo": "octo/repo", "title": "x", "base": "main"}
                )
            )
    finally:
        configure_secrets(None)


def test_review_pull_request_posts_event_and_body() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"id": 99, "state": "CHANGES_REQUESTED"})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_review_pull_request")(
                {
                    "repo": "octo/repo",
                    "number": 12,
                    "event": "request_changes",  # lower-case accepted (normalized)
                    "body": "please fix the lint",
                }
            )
        )
        assert seen["url"].endswith("/repos/octo/repo/pulls/12/reviews")
        assert seen["body"] == {
            "event": "REQUEST_CHANGES",
            "body": "please fix the lint",
        }
        assert out["event"] == "REQUEST_CHANGES"
        assert out["review_id"] == 99
    finally:
        configure_secrets(None)


def test_review_request_changes_requires_body() -> None:
    import httpx

    def app(
        request: httpx.Request,
    ) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(200, json={})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        with pytest.raises(ToolSecurityError, match="body` is required"):
            run_async(
                registry.handler_for("github_review_pull_request")(
                    {"repo": "octo/repo", "number": 12, "event": "REQUEST_CHANGES"}
                )
            )
    finally:
        configure_secrets(None)


def test_review_approve_does_not_require_body() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.read().decode())
        return httpx.Response(200, json={"id": 1, "state": "APPROVED"})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_review_pull_request")(
                {"repo": "octo/repo", "number": 3, "event": "APPROVE"}
            )
        )
        # No body field is sent for an approval (None dropped from the payload).
        assert seen["body"] == {"event": "APPROVE"}
        assert out["event"] == "APPROVE"
    finally:
        configure_secrets(None)


# --------------------------------------------------------------- repo reads


def test_get_repo_returns_metadata_summary() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo"
        return httpx.Response(
            200,
            json={
                "full_name": "octo/repo",
                "description": "a repo",
                "default_branch": "main",
                "private": False,
                "open_issues_count": 3,
                "stargazers_count": 42,
                "html_url": "https://github.com/octo/repo",
            },
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(registry.handler_for("github_get_repo")({"repo": "octo/repo"}))
        assert out["default_branch"] == "main"
        assert out["stars"] == 42
        assert out["open_issues"] == 3
    finally:
        configure_secrets(None)


def test_get_file_decodes_base64_content_at_ref() -> None:
    import base64

    import httpx

    seen: dict[str, Any] = {}
    content = b"hello himmy\n"

    def app(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "type": "file",
                "encoding": "base64",
                "path": "docs/readme.md",
                "sha": "abc123",
                "size": len(content),
                "content": base64.b64encode(content).decode(),
            },
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        out = run_async(
            registry.handler_for("github_get_file")(
                {"repo": "octo/repo", "path": "docs/readme.md", "ref": "main"}
            )
        )
        assert "/repos/octo/repo/contents/docs/readme.md" in seen["url"]
        assert "ref=main" in seen["url"]
        assert out["content"] == "hello himmy\n"
        assert out["sha"] == "abc123"
        assert out["truncated"] is False
    finally:
        configure_secrets(None)


def test_get_file_rejects_a_directory_payload() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        # The Contents API returns a list for a directory.
        return httpx.Response(200, json=[{"type": "file", "name": "a"}])

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        with pytest.raises(ConnectorError, match="not a readable file"):
            run_async(
                registry.handler_for("github_get_file")(
                    {"repo": "octo/repo", "path": "docs"}
                )
            )
    finally:
        configure_secrets(None)


# --------------------------------------------------------- allowlist (default-deny)


def test_unlisted_repo_denied_before_any_network_for_reads_and_writes() -> None:
    calls = {"n": 0}

    import httpx

    def app(
        request: httpx.Request,
    ) -> httpx.Response:  # pragma: no cover - never called
        calls["n"] += 1
        return httpx.Response(200, json=[])

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        audit = _RecordingAudit()
        conn = _connector(app, allowed_repos=["octo/repo"], audit=audit)
        registry = _registered(conn)
        # A read of an unlisted repo is denied (default-deny gates reads too).
        with pytest.raises(ToolSecurityError, match="not on the GitHub allowlist"):
            run_async(registry.handler_for("github_list_issues")({"repo": "evil/repo"}))
        # And a write.
        with pytest.raises(ToolSecurityError, match="not on the GitHub allowlist"):
            run_async(
                registry.handler_for("github_create_issue")(
                    {"repo": "evil/repo", "title": "x"}
                )
            )
        assert calls["n"] == 0  # never hit the network for a denied repo
        assert audit.events[-1].outcome == "deny"
    finally:
        configure_secrets(None)


def test_empty_allowlist_and_no_default_denies_everything() -> None:
    import httpx

    def app(
        request: httpx.Request,
    ) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(200, json=[])

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        conn = _connector(app, allowed_repos=None)  # no allowlist, no default
        registry = _registered(conn)
        with pytest.raises(ToolSecurityError):
            run_async(registry.handler_for("github_get_repo")({"repo": "octo/repo"}))
    finally:
        configure_secrets(None)


def test_default_repo_used_when_repo_omitted() -> None:
    import httpx

    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"full_name": "octo/default"})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        conn = _connector(app, allowed_repos=None, default_repo="octo/default")
        registry = _registered(conn)
        out = run_async(registry.handler_for("github_get_repo")({}))
        assert "/repos/octo/default" in seen["url"]
        assert out["full_name"] == "octo/default"
    finally:
        configure_secrets(None)


def test_malformed_repo_rejected() -> None:
    import httpx

    def app(
        request: httpx.Request,
    ) -> httpx.Response:  # pragma: no cover - never called
        return httpx.Response(200, json={})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        conn = _connector(app, allowed_repos=["octo/repo"])
        registry = _registered(conn)
        for bad in ("not-a-repo", "a/b/c", "../etc/passwd", "octo/"):
            with pytest.raises(ToolSecurityError):
                run_async(registry.handler_for("github_get_repo")({"repo": bad}))
    finally:
        configure_secrets(None)


# ----------------------------------------------------- error handling (404/403/rate)


def test_404_surfaces_clean_error_with_status() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        with pytest.raises(ConnectorError) as exc:
            run_async(registry.handler_for("github_get_repo")({"repo": "octo/repo"}))
        assert _TOKEN not in str(exc.value)  # the token never rides out
    finally:
        configure_secrets(None)


def test_403_rate_limit_maps_to_rate_limit_error_with_retry_after() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={"message": "API rate limit exceeded"},
            headers={"X-RateLimit-Remaining": "0", "Retry-After": "42"},
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        # Drive the client directly to assert the typed rate-limit error + retry hint.
        client = GitHubClient(_TOKEN, transport=_mock_transport(app))
        with pytest.raises(GitHubRateLimitError) as exc:
            client.get("/repos/octo/repo")
        assert exc.value.retry_after == 42.0
        assert exc.value.status == 429
    finally:
        configure_secrets(None)


def test_secondary_rate_limit_retry_after_without_remaining_header() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        # Secondary rate limit: Retry-After present, no X-RateLimit-Remaining: 0.
        return httpx.Response(
            429, json={"message": "secondary"}, headers={"Retry-After": "7"}
        )

    client = GitHubClient(_TOKEN, transport=_mock_transport(app))
    with pytest.raises(GitHubRateLimitError) as exc:
        client.get("/repos/octo/repo")
    assert exc.value.retry_after == 7.0


def test_403_authorization_failure_is_not_a_rate_limit() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        # A plain 403 (no rate-limit headers) is an auth/permission failure, not a limit.
        return httpx.Response(403, json={"message": "Resource not accessible by token"})

    client = GitHubClient(_TOKEN, transport=_mock_transport(app))
    with pytest.raises(GitHubError) as exc:
        client.get("/repos/octo/repo")
    assert not isinstance(exc.value, GitHubRateLimitError)
    assert exc.value.status == 403


def test_rate_limit_error_through_tool_is_secret_safe() -> None:
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            json={
                "message": f"limited; token {_TOKEN}"
            },  # maliciously echoes the token
            headers={"X-RateLimit-Remaining": "0", "Retry-After": "5"},
        )

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        registry = _registered(_connector(app))
        with pytest.raises(ConnectorError) as exc:
            run_async(
                registry.handler_for("github_create_issue")(
                    {"repo": "octo/repo", "title": "x"}
                )
            )
        assert _TOKEN not in str(exc.value)  # token scrubbed from the surfaced error
    finally:
        configure_secrets(None)


# ------------------------------------------------------------- token sourcing / SSRF


def test_token_read_from_secrets_layer_not_env(monkeypatch: Any) -> None:
    """The token comes from the secrets layer, never straight from os.environ."""
    import httpx

    monkeypatch.setenv(GITHUB_TOKEN_SECRET, "from-env-MUST-NOT-USE")
    seen: dict[str, Any] = {}

    def app(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"full_name": "octo/repo"})

    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: "from-secrets-layer"}))
    try:
        registry = _registered(_connector(app))
        run_async(registry.handler_for("github_get_repo")({"repo": "octo/repo"}))
        assert seen["auth"] == "Bearer from-secrets-layer"
    finally:
        configure_secrets(None)


def test_default_fetcher_carries_ssrf_guard_and_egress_allowlist() -> None:
    """The connector's guard refuses an SSRF target and pins egress to the GitHub host."""
    configure_secrets(_MemSecrets({GITHUB_TOKEN_SECRET: _TOKEN}))
    try:
        conn = GitHubOutboundConnector(allowed_repos=["octo/repo"])
        with pytest.raises(ToolSecurityError):
            conn.guard("http://169.254.169.254/latest/meta-data/")
        with pytest.raises(ToolSecurityError, match="egress allow-list"):
            conn.guard("https://evil.example.test/webhook")
    finally:
        configure_secrets(None)


def test_pagination_link_leaving_api_host_is_rejected() -> None:
    """A tampered Link header pointing off the API host ends the crawl with an error."""
    import httpx

    def app(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=[{"number": 1, "title": "p1", "state": "open"}],
            headers={
                "Link": "<https://evil.example.test/repos/octo/repo/issues?page=2>; "
                'rel="next"'
            },
        )

    client = GitHubClient(_TOKEN, transport=_mock_transport(app))
    with pytest.raises(GitHubError, match="left the API host"):
        client.get_paginated("/repos/octo/repo/issues")
