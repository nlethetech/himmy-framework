"""GitHub connector on the SDK: outbound tools for issues, pull requests, and repos.

A single **outbound** surface over GitHub's public REST API (v3), inheriting the connector
SDK's security posture (default-deny allowlist, secrets-layer credentials, SSRF-guarded
egress, idempotent side effects, secret-safe errors, structured audit events):

* :class:`GitHubOutboundConnector` exposes a family of typed tools an agent can call —

  - **issues**: ``github_list_issues``, ``github_create_issue``, ``github_comment_issue``;
  - **pull requests**: ``github_list_pull_requests``, ``github_create_pull_request``,
    ``github_comment_pull_request`` (a PR is an issue for the comment API),
    ``github_review_pull_request`` (approve / request-changes / comment);
  - **repo reads**: ``github_get_repo`` (metadata), ``github_get_file`` (a file's decoded
    text content at a ref).

  Every call carries the token from the **secrets layer** (:data:`GITHUB_TOKEN_SECRET`),
  never the model; goes through the connector's **SSRF-guarded pinned transport** pinned to
  the GitHub API host; targets a repository on the **repo allowlist** (default-deny, an
  ``owner/repo`` set so an agent can only touch repositories an operator named); respects
  **pagination** (follows the ``Link`` header up to a page cap) and **rate limits**
  (honours ``Retry-After`` / ``X-RateLimit-Reset`` on a 403/429 and surfaces a clean,
  token-free error instead of hammering the API); and every **side-effecting** call
  (create / comment / review) is **idempotent** — deduped on a per-action key so a retry or
  a re-issued tool call mutates exactly once.

Authentication accepts either a **personal access token (PAT)** or a **GitHub App
installation token** — both are bearer tokens on the REST API, so the connector reads one
opaque token from the secrets layer and sends ``Authorization: Bearer <token>``; minting an
installation token from an App's private key is the operator's job (outside this connector,
which never sees a private key). The token is never logged and is scrubbed from any error.

Nothing here imports ``httpx`` at module load — capability detection probes lazily — so
importing this module is free in an air-gapped install.

LIVE-PENDING: the request shapes, auth header, pagination, and rate-limit handling are
contract-tested against a mocked transport (see ``tests/connectors/test_github.py``);
verifying against the REAL GitHub API needs a live PAT/App token + a repository and cannot
run on this machine.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Callable, Collection, Mapping
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from himmy.connectors.sdk import (
    Capability,
    ConnectorContext,
    ConnectorError,
    OutboundToolConnector,
    RateLimiter,
    capability,
    redact_args,
    safe_error,
)
from himmy.services.tools.registry import ToolRegistry, register_local_tool
from himmy.services.tools.security import ToolSecurityError
from himmy.toolkit._net import build_pinned_transport

if TYPE_CHECKING:  # pragma: no cover - typing only (httpx stays a lazy import)
    import httpx

#: GitHub REST API root. The connector pins its egress to this host.
GITHUB_API_ROOT = "https://api.github.com"
#: The single API host the connector is allowed to reach (egress allow-list).
GITHUB_API_HOST = "api.github.com"
#: The REST media type pinning a stable response schema (GitHub's recommended default).
GITHUB_API_VERSION = "2022-11-28"

#: Secret name for the API token — a PAT *or* a GitHub App installation token, resolved via
#: the secrets layer, NEVER the model. This is the NAME of the env/secret var, not a
#: credential value (S105 false positive).
GITHUB_TOKEN_SECRET = "HIMMY_GITHUB_TOKEN"  # noqa: S105

#: Hard cap on pages followed for a paginated list (defence against an unbounded crawl that
#: would burn the rate-limit budget). 100 items/page * this is the most a list returns.
_MAX_PAGES = 10
#: GitHub's max page size for list endpoints.
_PER_PAGE = 100
#: Cap on a fetched file's decoded size surfaced to the model (a DoS / context guard).
_MAX_FILE_BYTES = 1024 * 1024

#: Valid PR review verdicts (the GitHub ``event`` field on a review).
_REVIEW_EVENTS = ("APPROVE", "REQUEST_CHANGES", "COMMENT")


class GitHubError(ConnectorError):
    """A GitHub API call failed. Carries the HTTP status so a caller can branch.

    The message is constructed to be secret-safe (only the status, the API's own
    token-free error message, and the request path — never a request header or the raw
    body that could echo the token). ``status`` is the HTTP status code (0 for a
    transport-level failure), so a handler can distinguish 404 / 403 / rate-limit from a
    generic error without parsing the string.
    """

    def __init__(self, message: str, *, status: int = 0) -> None:
        super().__init__(message)
        self.status = status


class GitHubRateLimitError(GitHubError):
    """The GitHub API refused the request for rate limiting (primary or secondary).

    ``retry_after`` is the number of seconds the API asked the caller to wait (parsed from
    ``Retry-After`` or computed from ``X-RateLimit-Reset``), or None when the API gave no
    hint. The connector NEVER auto-retries a side-effecting call on this — it surfaces the
    wait so the agent (or a human) backs off, rather than blindly re-POSTing.
    """

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message, status=429)
        self.retry_after = retry_after


def _parse_repo(repo: str) -> tuple[str, str]:
    """Split ``"owner/name"`` into ``(owner, name)``; raise on a malformed value.

    Rejects anything that is not exactly ``owner/name`` (no extra slashes, no empty half,
    no path traversal) so a model-supplied repo can never be coerced into a different API
    path. The owner/name are later percent-encoded into the URL too.
    """
    parts = repo.strip().strip("/").split("/")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ToolSecurityError(f"repo must be in 'owner/name' form, got {repo!r}")
    owner, name = parts
    if any(c in owner + name for c in ("..", "\\", " ")):
        raise ToolSecurityError(f"repo {repo!r} contains an illegal character")
    return owner, name


class GitHubClient:
    """A small REST client for the GitHub API: GET (paged) / POST / PATCH.

    All HTTP rides on the SSRF-guarded **pinned transport** (the DNS-rebinding-safe one the
    rest of the connector layer uses) with redirects OFF, so a 3xx can never bounce an
    authenticated request to an internal host. The token is passed by the caller (resolved
    from the secrets layer) and sent as ``Authorization: Bearer <token>``; it is never
    logged. ``transport`` is a test seam for injecting an ``httpx`` mock.

    Pagination and rate limits are handled here so the tool handlers stay declarative:
    :meth:`get_paginated` walks the ``Link`` header up to ``max_pages``; every method maps a
    403/429 + ``X-RateLimit-Remaining: 0`` (or a secondary-rate ``Retry-After``) to a
    :class:`GitHubRateLimitError` carrying the wait, and any other non-2xx to a secret-safe
    :class:`GitHubError` carrying the status.
    """

    def __init__(
        self,
        token: str,
        *,
        base_url: str = GITHUB_API_ROOT,
        timeout: float = 20.0,
        allow_private_hosts: bool = False,
        egress_allow_hosts: Collection[str] | None = (GITHUB_API_HOST,),
        transport: Any = None,
        max_pages: int = _MAX_PAGES,
    ) -> None:
        self._token = token
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._allow_private = allow_private_hosts
        self._allow_hosts = set(egress_allow_hosts) if egress_allow_hosts else None
        self._transport = transport
        self._max_pages = max(1, max_pages)

    def _auth_headers(self) -> dict[str, str]:
        # GitHub REST auth uses the standard Bearer scheme for BOTH a PAT and an App
        # installation token; the version header pins a stable response schema.
        return {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
            "Content-Type": "application/json",
        }

    def _client(self) -> httpx.Client:
        import httpx

        transport = self._transport
        if transport is None:
            transport = build_pinned_transport(
                allow_private=self._allow_private, allow_hosts=self._allow_hosts
            )
        return httpx.Client(
            timeout=self._timeout,
            follow_redirects=False,  # never auto-follow a 3xx with the token attached
            transport=transport,
        )

    # -- error normalization ------------------------------------------------
    def _raise_for_status(self, resp: httpx.Response, *, path: str) -> None:
        """Map a non-2xx response to a secret-safe :class:`GitHubError` (or rate-limit).

        Surfaces ONLY the status, the request *path*, and the API's own (token-free)
        ``message`` field — never a request header or the raw body verbatim, so a token a
        proxy might echo cannot ride out in an error. A 403/429 that is actually a rate
        limit (``X-RateLimit-Remaining: 0`` or a secondary-rate ``Retry-After``) becomes a
        :class:`GitHubRateLimitError` carrying the wait.
        """
        if resp.status_code < 400:
            return
        retry_after = _rate_limit_retry_after(resp)
        if retry_after is not None or (
            resp.status_code in (403, 429)
            and resp.headers.get("X-RateLimit-Remaining") == "0"
        ):
            raise GitHubRateLimitError(
                f"github rate limit hit on {path} (HTTP {resp.status_code})",
                retry_after=retry_after,
            )
        # The API's `message` is documentation-quality and token-free; safe to surface.
        api_message = ""
        try:
            api_message = str((resp.json() or {}).get("message", ""))
        except Exception:  # noqa: BLE001 - non-JSON error body
            api_message = ""
        suffix = f": {api_message}" if api_message else ""
        raise GitHubError(
            f"github {path} failed: HTTP {resp.status_code}{suffix}",
            status=resp.status_code,
        )

    def _json_body(self, resp: httpx.Response, *, path: str) -> Any:
        try:
            return resp.json()
        except Exception:  # noqa: BLE001 - non-JSON body
            raise GitHubError(
                f"github {path} returned a non-JSON body", status=resp.status_code
            ) from None

    # -- verbs --------------------------------------------------------------
    def get(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        """GET ``path`` (a single object/array) and return the parsed JSON."""
        url = self._url(path, params)
        with self._client() as client:
            resp = client.get(url, headers=self._auth_headers())
            self._raise_for_status(resp, path=path)
            return self._json_body(resp, path=path)

    def get_paginated(
        self, path: str, *, params: Mapping[str, Any] | None = None
    ) -> list[Any]:
        """GET a list endpoint, following ``Link: rel="next"`` up to ``max_pages``.

        Each page is fetched with ``per_page=100``; the ``Link`` header's ``next`` URL
        (already absolute, on the API host) is re-guarded by the pinned transport before
        the hop. The page cap bounds the crawl so a huge list cannot exhaust the
        rate-limit budget; the result is the concatenation of every page's array.
        """
        merged = dict(params or {})
        merged.setdefault("per_page", _PER_PAGE)
        url: str | None = self._url(path, merged)
        out: list[Any] = []
        pages = 0
        with self._client() as client:
            while url is not None and pages < self._max_pages:
                resp = client.get(url, headers=self._auth_headers())
                self._raise_for_status(resp, path=path)
                page = self._json_body(resp, path=path)
                if isinstance(page, list):
                    out.extend(page)
                else:  # a non-list payload ends pagination (defensive)
                    out.append(page)
                    break
                url = _next_link(resp.headers.get("Link"))
                # Pin every follow-up page to the API host (the Link header is GitHub's,
                # but re-validate so a tampered/proxied Link cannot redirect egress).
                if url is not None and not url.startswith(self._base):
                    raise GitHubError(
                        f"github {path} pagination link left the API host",
                        status=0,
                    )
                pages += 1
        return out

    def post(self, path: str, *, body: Mapping[str, Any]) -> Any:
        """POST ``body`` to ``path`` and return the created object."""
        url = self._url(path, None)
        with self._client() as client:
            resp = client.post(url, content=_dump(body), headers=self._auth_headers())
            self._raise_for_status(resp, path=path)
            return self._json_body(resp, path=path)

    def patch(self, path: str, *, body: Mapping[str, Any]) -> Any:
        """PATCH ``body`` onto ``path`` and return the updated object."""
        url = self._url(path, None)
        with self._client() as client:
            resp = client.request(
                "PATCH", url, content=_dump(body), headers=self._auth_headers()
            )
            self._raise_for_status(resp, path=path)
            return self._json_body(resp, path=path)

    def _url(self, path: str, params: Mapping[str, Any] | None) -> str:
        # ``path`` is already absolute when it is a Link-header follow-up.
        if path.startswith("http://") or path.startswith("https://"):
            base = path
        else:
            base = f"{self._base}/{path.lstrip('/')}"
        if params:
            query = urlencode(
                {k: v for k, v in params.items() if v is not None}, doseq=True
            )
            if query:
                base = f"{base}{'&' if '?' in base else '?'}{query}"
        return base


def _dump(body: Mapping[str, Any]) -> bytes:
    """Serialize a JSON body, dropping ``None`` values (GitHub rejects nulls on some fields)."""
    return json.dumps({k: v for k, v in body.items() if v is not None}).encode("utf-8")


def _rate_limit_retry_after(resp: httpx.Response) -> float | None:
    """Seconds to wait per a rate-limit response, or None when it is not a rate limit.

    Honours the secondary-rate ``Retry-After`` header first (delta seconds), then the
    primary-rate ``X-RateLimit-Reset`` (an epoch second) when the remaining budget is 0.
    Returns None for a 403 that is an authorization failure rather than a rate limit.
    """
    retry_after = resp.headers.get("Retry-After")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            return None
    if (
        resp.status_code in (403, 429)
        and resp.headers.get("X-RateLimit-Remaining") == "0"
    ):
        reset = resp.headers.get("X-RateLimit-Reset")
        if reset:
            try:
                import time

                return max(0.0, float(reset) - time.time())
            except ValueError:
                return None
        return None
    return None


def _next_link(link_header: str | None) -> str | None:
    """Extract the ``rel="next"`` URL from a GitHub ``Link`` header (None when absent)."""
    if not link_header:
        return None
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        url_seg = segments[0].strip()
        if not (url_seg.startswith("<") and url_seg.endswith(">")):
            continue
        rels = [s.strip() for s in segments[1:]]
        if any(rel == 'rel="next"' for rel in rels):
            return url_seg[1:-1]
    return None


# --------------------------------------------------------------- tool schemas
_REPO_PROP = {
    "type": "string",
    "description": (
        "Target repository in 'owner/name' form. Must be on the connector's repo "
        "allowlist; defaults to the configured default repository."
    ),
}

_LIST_ISSUES_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": _REPO_PROP,
        "state": {
            "type": "string",
            "enum": ["open", "closed", "all"],
            "description": "Filter by issue state (default: open).",
        },
        "labels": {
            "type": "string",
            "description": "Comma-separated label names to filter by (optional).",
        },
    },
    "required": [],
    "additionalProperties": False,
}

_CREATE_ISSUE_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": _REPO_PROP,
        "title": {"type": "string", "description": "The issue title (required)."},
        "body": {"type": "string", "description": "The issue body (Markdown)."},
        "labels": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Label names to apply (optional).",
        },
    },
    "required": ["title"],
    "additionalProperties": False,
}

_COMMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": _REPO_PROP,
        "number": {
            "type": "integer",
            "description": "The issue or pull-request number to comment on (required).",
        },
        "body": {
            "type": "string",
            "description": "The comment body (Markdown, required).",
        },
    },
    "required": ["number", "body"],
    "additionalProperties": False,
}

_LIST_PRS_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": _REPO_PROP,
        "state": {
            "type": "string",
            "enum": ["open", "closed", "all"],
            "description": "Filter by pull-request state (default: open).",
        },
        "base": {
            "type": "string",
            "description": "Filter by base branch name (optional).",
        },
    },
    "required": [],
    "additionalProperties": False,
}

_CREATE_PR_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": _REPO_PROP,
        "title": {
            "type": "string",
            "description": "The pull-request title (required).",
        },
        "head": {
            "type": "string",
            "description": "The branch (or 'owner:branch') the changes are on (required).",
        },
        "base": {
            "type": "string",
            "description": "The branch to merge into, e.g. 'main' (required).",
        },
        "body": {"type": "string", "description": "The PR description (Markdown)."},
        "draft": {
            "type": "boolean",
            "description": "Open as a draft PR (default false).",
        },
    },
    "required": ["title", "head", "base"],
    "additionalProperties": False,
}

_REVIEW_PR_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": _REPO_PROP,
        "number": {
            "type": "integer",
            "description": "The pull-request number to review (required).",
        },
        "event": {
            "type": "string",
            "enum": list(_REVIEW_EVENTS),
            "description": "The review verdict (required).",
        },
        "body": {
            "type": "string",
            "description": (
                "The review body. Required by GitHub for REQUEST_CHANGES and COMMENT."
            ),
        },
    },
    "required": ["number", "event"],
    "additionalProperties": False,
}

_GET_REPO_SCHEMA = {
    "type": "object",
    "properties": {"repo": _REPO_PROP},
    "required": [],
    "additionalProperties": False,
}

_GET_FILE_SCHEMA = {
    "type": "object",
    "properties": {
        "repo": _REPO_PROP,
        "path": {
            "type": "string",
            "description": "The file path within the repository (required).",
        },
        "ref": {
            "type": "string",
            "description": "A branch, tag, or commit SHA (optional; defaults to the "
            "repository's default branch).",
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


class GitHubOutboundConnector(OutboundToolConnector):
    """Exposes the GitHub issue / pull-request / repo tools — repo-scoped, idempotent.

    Security contract (all inherited from :class:`OutboundToolConnector` + enforced here):

    * the API token (PAT or App installation token) is read from the **secrets layer**
      (:data:`GITHUB_TOKEN_SECRET`) at call time, never from a constructor arg the model
      could set, never logged;
    * the connector is **unavailable** (skipped cleanly by discovery / ``register_tools``)
      when the token is absent — capability-gated;
    * every request goes through the SSRF-guarded **pinned transport**, pinned to the GitHub
      API host (egress allow-list);
    * the target repository must be on the **repo allowlist** (default-deny: an empty
      allowlist with no configured default rejects every call, read or write);
    * list endpoints **paginate** (follow the ``Link`` header up to a page cap) and every
      call **respects rate limits** (a 403/429 rate-limit surfaces a clean wait, never an
      auto-retried POST);
    * every **side-effecting** call (create issue/PR, comment, review) is **idempotent** —
      deduped per ``(repo, action, key)`` so a retry or a re-issued tool call mutates once.
    """

    name = "github"
    required_secrets = (GITHUB_TOKEN_SECRET,)

    def __init__(
        self,
        *,
        allowed_repos: Collection[str] | None = None,
        default_repo: str | None = None,
        requires_approval: bool = True,
        client_factory: Callable[[str | None], GitHubClient] | None = None,
        context: ConnectorContext | None = None,
        **kwargs: Any,
    ) -> None:
        """Wire the repo allowlist + default repo.

        ``allowed_repos`` is the allowlist of ``owner/name`` strings (compared
        case-insensitively, as GitHub treats them). With it empty and no ``default_repo``,
        EVERY tool denies — the default-deny posture (a read is gated too, so an agent
        cannot enumerate a repository the operator did not name). ``requires_approval``
        gates the *side-effecting* tools behind human-in-the-loop approval by default (the
        read tools are registered as read-only and never require approval).
        ``client_factory`` is the injectable seam tests use to supply a
        :class:`GitHubClient` over a mock transport.
        """
        # Default the egress allow-list to the GitHub API host for defence in depth.
        kwargs.setdefault("egress_allow_hosts", (GITHUB_API_HOST,))
        # A sane per-connector self-throttle so a runaway agent can't burn the rate limit.
        if context is None:
            context = ConnectorContext(rate_limiter=RateLimiter(rate=10, window=1.0))
        super().__init__(context=context, **kwargs)
        self._allowed_repos = {str(r).lower() for r in (allowed_repos or [])}
        self._default_repo = str(default_repo) if default_repo else None
        self._requires_approval = requires_approval
        self._client_factory = client_factory

    def capability(self) -> Capability:
        """Available only when the API token resolves through the secrets layer."""
        return capability(requires_secrets=self.required_secrets)

    # -- repo allowlist -----------------------------------------------------
    def _repo_allowed(self, repo: str) -> bool:
        """Whether ``repo`` (``owner/name``) may be accessed (default-deny when unconfigured)."""
        target = repo.strip().lower()
        if not self._allowed_repos:
            # No explicit allowlist: only the single configured default repo is permitted
            # (and only if one was configured). Never "any repo".
            return (
                self._default_repo is not None and target == self._default_repo.lower()
            )
        return target in self._allowed_repos

    def _resolve_repo(self, args: Mapping[str, Any]) -> str:
        """Resolve + allowlist-gate the target repo for a tool call (raises on deny).

        Uses the ``repo`` arg or the configured default; validates its ``owner/name`` shape;
        then applies the repo allowlist (auditing a deny). Raises :class:`ToolSecurityError`
        pre-flight (no network) so a denied repo is distinct from a downstream API failure.
        """
        repo = str(args.get("repo") or self._default_repo or "")
        if not repo:
            raise ToolSecurityError("a repo is required (or configure a default repo)")
        _parse_repo(repo)  # shape check (raises a ToolSecurityError on malformed)
        if not self._repo_allowed(repo):
            self._audit(
                "github",
                "deny",
                actor={"subject": repo},
                detail="repo not on allowlist",
            )
            raise ToolSecurityError(f"repo {repo!r} is not on the GitHub allowlist")
        return repo

    def _build_client(self) -> GitHubClient:
        token = self.secret(GITHUB_TOKEN_SECRET)  # raises (named) if absent
        if self._client_factory is not None:
            return self._client_factory(token)
        return GitHubClient(
            token or "",
            allow_private_hosts=self._allow_private,
            egress_allow_hosts=self._allow_hosts or (GITHUB_API_HOST,),
        )

    # -- secret-safe error wrapper ------------------------------------------
    def _fail(self, tool: str, repo: str, exc: BaseException) -> ConnectorError:
        """Audit + normalize ``exc`` into a secret-safe :class:`ConnectorError`.

        The token is scrubbed from the surfaced text belt-and-braces, on top of
        :func:`safe_error` only trusting a connector-authored message — so even if a
        third-party library echoed the token into an exception, it never rides out.
        """
        token = self.secret(GITHUB_TOKEN_SECRET, required=False) or ""
        detail = safe_error(exc, secrets=[token])
        self._audit(tool, "deny", actor={"subject": repo}, detail=detail)
        return ConnectorError(f"{tool} failed: {detail}")

    # -- side-effect helper -------------------------------------------------
    def _mutate(
        self,
        tool: str,
        repo: str,
        args: Mapping[str, Any],
        idem_key: str,
        call: Callable[[GitHubClient], Any],
    ) -> Any:
        """Run an idempotent, throttled, secret-safe side-effecting GitHub call.

        Dedupes on ``(repo, tool, idem_key)`` so a retried or re-issued mutation fires the
        upstream effect exactly once; throttles INSIDE the idempotent body so a deduplicated
        repeat consumes no rate-limit token; audits the allow; and converts any failure into
        a secret-safe error.
        """
        client = self._build_client()
        digest = hashlib.sha256(f"{repo}:{tool}:{idem_key}".encode()).hexdigest()

        def _send() -> Any:
            self._throttle()
            return call(client)

        try:
            result = self.call_idempotent(f"{tool}:{digest}", _send)
        except Exception as exc:  # noqa: BLE001 - normalize to a secret-safe error
            raise self._fail(tool, repo, exc) from None
        self._audit(
            tool, "allow", actor={"subject": repo}, detail=f"{dict(redact_args(args))}"
        )
        return result

    # -- read helper --------------------------------------------------------
    def _read(
        self,
        tool: str,
        repo: str,
        call: Callable[[GitHubClient], Any],
    ) -> Any:
        """Run a throttled, secret-safe read against GitHub (no idempotency needed)."""
        client = self._build_client()
        self._throttle()
        try:
            result = call(client)
        except Exception as exc:  # noqa: BLE001 - normalize to a secret-safe error
            raise self._fail(tool, repo, exc) from None
        self._audit(tool, "allow", actor={"subject": repo})
        return result

    def register(self, registry: ToolRegistry) -> list[str]:
        import asyncio

        # ---- issues: list (read) ----------------------------------------
        async def github_list_issues(args: dict[str, Any]) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)
            params: dict[str, Any] = {"state": args.get("state") or "open"}
            if args.get("labels"):
                params["labels"] = str(args["labels"])

            def _call(client: GitHubClient) -> list[Any]:
                return client.get_paginated(
                    f"/repos/{quote(owner)}/{quote(name)}/issues", params=params
                )

            items = await asyncio.to_thread(
                self._read, "github_list_issues", repo, _call
            )
            # GitHub returns PRs in the issues list; filter to true issues so the tool's
            # name matches its output (a PR has a `pull_request` key).
            issues = [
                _summarize_issue(it)
                for it in items
                if isinstance(it, dict) and "pull_request" not in it
            ]
            return {"repo": repo, "count": len(issues), "issues": issues}

        # ---- issues: create (mutation) ----------------------------------
        async def github_create_issue(args: dict[str, Any]) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)
            title = str(args.get("title") or "")
            if not title:
                raise ToolSecurityError("github_create_issue requires `title`")
            body: dict[str, Any] = {"title": title, "body": args.get("body")}
            labels = args.get("labels")
            if labels:
                body["labels"] = [str(label) for label in labels]
            # Idempotency over (repo, title, body): a retried create of the SAME issue
            # returns the first result rather than opening a duplicate.
            idem = hashlib.sha256(
                f"{title}\x00{args.get('body') or ''}".encode()
            ).hexdigest()

            def _call(client: GitHubClient) -> Any:
                return client.post(
                    f"/repos/{quote(owner)}/{quote(name)}/issues", body=body
                )

            created = await asyncio.to_thread(
                self._mutate, "github_create_issue", repo, args, idem, _call
            )
            return {"repo": repo, **_summarize_issue(created)}

        # ---- issues: comment (mutation) ---------------------------------
        async def github_comment_issue(args: dict[str, Any]) -> dict[str, Any]:
            return await _comment(args, "github_comment_issue")

        # ---- shared comment path (issues + PRs use the same endpoint) ----
        async def _comment(args: dict[str, Any], tool: str) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)
            number = _require_number(args, tool)
            comment_body = str(args.get("body") or "")
            if not comment_body:
                raise ToolSecurityError(f"{tool} requires `body`")
            idem = hashlib.sha256(f"{number}\x00{comment_body}".encode()).hexdigest()

            def _call(client: GitHubClient) -> Any:
                return client.post(
                    f"/repos/{quote(owner)}/{quote(name)}/issues/{number}/comments",
                    body={"body": comment_body},
                )

            created = await asyncio.to_thread(
                self._mutate, tool, repo, args, idem, _call
            )
            return {
                "repo": repo,
                "number": number,
                "comment_id": (
                    int(created.get("id", 0)) if isinstance(created, dict) else 0
                ),
                "url": str(created.get("html_url", ""))
                if isinstance(created, dict)
                else "",
            }

        # ---- pull requests: list (read) ---------------------------------
        async def github_list_pull_requests(args: dict[str, Any]) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)
            params: dict[str, Any] = {"state": args.get("state") or "open"}
            if args.get("base"):
                params["base"] = str(args["base"])

            def _call(client: GitHubClient) -> list[Any]:
                return client.get_paginated(
                    f"/repos/{quote(owner)}/{quote(name)}/pulls", params=params
                )

            items = await asyncio.to_thread(
                self._read, "github_list_pull_requests", repo, _call
            )
            prs = [_summarize_pr(it) for it in items if isinstance(it, dict)]
            return {"repo": repo, "count": len(prs), "pull_requests": prs}

        # ---- pull requests: create (mutation) ---------------------------
        async def github_create_pull_request(args: dict[str, Any]) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)
            title = str(args.get("title") or "")
            head = str(args.get("head") or "")
            base = str(args.get("base") or "")
            if not (title and head and base):
                raise ToolSecurityError(
                    "github_create_pull_request requires `title`, `head`, and `base`"
                )
            body: dict[str, Any] = {
                "title": title,
                "head": head,
                "base": base,
                "body": args.get("body"),
                "draft": bool(args.get("draft")) if args.get("draft") else None,
            }
            idem = hashlib.sha256(f"{head}\x00{base}\x00{title}".encode()).hexdigest()

            def _call(client: GitHubClient) -> Any:
                return client.post(
                    f"/repos/{quote(owner)}/{quote(name)}/pulls", body=body
                )

            created = await asyncio.to_thread(
                self._mutate, "github_create_pull_request", repo, args, idem, _call
            )
            return {"repo": repo, **_summarize_pr(created)}

        # ---- pull requests: comment (mutation) --------------------------
        async def github_comment_pull_request(args: dict[str, Any]) -> dict[str, Any]:
            # A PR is an issue for the comment API; reuse the same idempotent path.
            return await _comment(args, "github_comment_pull_request")

        # ---- pull requests: review (mutation) ---------------------------
        async def github_review_pull_request(args: dict[str, Any]) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)
            number = _require_number(args, "github_review_pull_request")
            event = str(args.get("event") or "").upper()
            if event not in _REVIEW_EVENTS:
                raise ToolSecurityError(
                    f"github_review_pull_request `event` must be one of {_REVIEW_EVENTS}"
                )
            review_body = str(args.get("body") or "")
            if event in ("REQUEST_CHANGES", "COMMENT") and not review_body:
                # GitHub rejects these two without a body; fail fast (no wasted call).
                raise ToolSecurityError(
                    f"github_review_pull_request `body` is required for {event}"
                )
            body: dict[str, Any] = {"event": event, "body": review_body or None}
            idem = hashlib.sha256(
                f"{number}\x00{event}\x00{review_body}".encode()
            ).hexdigest()

            def _call(client: GitHubClient) -> Any:
                return client.post(
                    f"/repos/{quote(owner)}/{quote(name)}/pulls/{number}/reviews",
                    body=body,
                )

            created = await asyncio.to_thread(
                self._mutate, "github_review_pull_request", repo, args, idem, _call
            )
            return {
                "repo": repo,
                "number": number,
                "event": event,
                "review_id": (
                    int(created.get("id", 0)) if isinstance(created, dict) else 0
                ),
                "state": str(created.get("state", ""))
                if isinstance(created, dict)
                else "",
            }

        # ---- repo: metadata (read) --------------------------------------
        async def github_get_repo(args: dict[str, Any]) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)

            def _call(client: GitHubClient) -> Any:
                return client.get(f"/repos/{quote(owner)}/{quote(name)}")

            meta = await asyncio.to_thread(self._read, "github_get_repo", repo, _call)
            return _summarize_repo(meta if isinstance(meta, dict) else {})

        # ---- repo: file contents (read) ---------------------------------
        async def github_get_file(args: dict[str, Any]) -> dict[str, Any]:
            repo = self._resolve_repo(args)
            owner, name = _parse_repo(repo)
            path = str(args.get("path") or "")
            if not path:
                raise ToolSecurityError("github_get_file requires `path`")
            # Encode the path segment-by-segment so slashes stay structural but each
            # component is escaped (no traversal, no query injection).
            enc_path = "/".join(quote(seg) for seg in path.strip("/").split("/"))
            params: dict[str, Any] = {}
            if args.get("ref"):
                params["ref"] = str(args["ref"])

            def _call(client: GitHubClient) -> Any:
                return client.get(
                    f"/repos/{quote(owner)}/{quote(name)}/contents/{enc_path}",
                    params=params,
                )

            meta = await asyncio.to_thread(self._read, "github_get_file", repo, _call)
            return _decode_file(repo, path, meta if isinstance(meta, dict) else {})

        read_only_tools = {
            "github_list_issues": (
                github_list_issues,
                _LIST_ISSUES_SCHEMA,
                "List a repository's open/closed issues (paginated).",
            ),
            "github_list_pull_requests": (
                github_list_pull_requests,
                _LIST_PRS_SCHEMA,
                "List a repository's pull requests (paginated).",
            ),
            "github_get_repo": (
                github_get_repo,
                _GET_REPO_SCHEMA,
                "Read an allow-listed repository's metadata.",
            ),
            "github_get_file": (
                github_get_file,
                _GET_FILE_SCHEMA,
                "Read a file's text content from an allow-listed repository at a ref.",
            ),
        }
        write_tools = {
            "github_create_issue": (
                github_create_issue,
                _CREATE_ISSUE_SCHEMA,
                "Open an issue in an allow-listed repository (idempotent).",
            ),
            "github_comment_issue": (
                github_comment_issue,
                _COMMENT_SCHEMA,
                "Comment on an issue in an allow-listed repository (idempotent).",
            ),
            "github_create_pull_request": (
                github_create_pull_request,
                _CREATE_PR_SCHEMA,
                "Open a pull request in an allow-listed repository (idempotent).",
            ),
            "github_comment_pull_request": (
                github_comment_pull_request,
                _COMMENT_SCHEMA,
                "Comment on a pull request in an allow-listed repository (idempotent).",
            ),
            "github_review_pull_request": (
                github_review_pull_request,
                _REVIEW_PR_SCHEMA,
                "Submit a review (approve/request-changes/comment) on a pull request "
                "(idempotent).",
            ),
        }

        names: list[str] = []
        for tool_name, (handler, schema, description) in read_only_tools.items():
            register_local_tool(
                registry,
                name=tool_name,
                read_only=True,
                handler=handler,
                description=description,
                args_json_schema=schema,
                requires_approval=False,  # reads never need approval
                sensitive_arg_names=[],
                metadata={"connector": "github"},
            )
            names.append(tool_name)
        for tool_name, (handler, schema, description) in write_tools.items():
            register_local_tool(
                registry,
                name=tool_name,
                read_only=False,
                handler=handler,
                description=description,
                args_json_schema=schema,
                requires_approval=self._requires_approval,
                sensitive_arg_names=[],
                metadata={"connector": "github"},
            )
            names.append(tool_name)
        return names


# --------------------------------------------------------------- arg helpers
def _require_number(args: Mapping[str, Any], tool: str) -> int:
    """Coerce + validate the ``number`` arg into a positive int (raises otherwise)."""
    raw = args.get("number")
    try:
        number = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ToolSecurityError(f"{tool} requires an integer `number`") from None
    if number <= 0:
        raise ToolSecurityError(f"{tool} `number` must be positive")
    return number


# --------------------------------------------------------------- summarizers
def _summarize_issue(issue: Mapping[str, Any]) -> dict[str, Any]:
    """Project a GitHub issue object down to the fields a model needs (token-free)."""
    return {
        "number": int(issue.get("number", 0)),
        "title": str(issue.get("title", "")),
        "state": str(issue.get("state", "")),
        "url": str(issue.get("html_url", "")),
        "labels": [
            str(label.get("name", ""))
            for label in (issue.get("labels") or [])
            if isinstance(label, dict)
        ],
    }


def _summarize_pr(pr: Mapping[str, Any]) -> dict[str, Any]:
    """Project a GitHub pull-request object down to the fields a model needs."""
    return {
        "number": int(pr.get("number", 0)),
        "title": str(pr.get("title", "")),
        "state": str(pr.get("state", "")),
        "url": str(pr.get("html_url", "")),
        "head": str((pr.get("head") or {}).get("ref", "")),
        "base": str((pr.get("base") or {}).get("ref", "")),
        "draft": bool(pr.get("draft", False)),
    }


def _summarize_repo(repo: Mapping[str, Any]) -> dict[str, Any]:
    """Project a GitHub repository object down to its useful metadata."""
    return {
        "full_name": str(repo.get("full_name", "")),
        "description": str(repo.get("description") or ""),
        "default_branch": str(repo.get("default_branch", "")),
        "private": bool(repo.get("private", False)),
        "open_issues": int(repo.get("open_issues_count", 0)),
        "stars": int(repo.get("stargazers_count", 0)),
        "url": str(repo.get("html_url", "")),
    }


def _decode_file(repo: str, path: str, meta: Mapping[str, Any]) -> dict[str, Any]:
    """Decode a GitHub Contents API file object into UTF-8 text (capped).

    The Contents API returns base64-encoded content for a file. A directory (a list
    payload) or a non-file type is rejected with a clear error. The decoded body is capped
    at :data:`_MAX_FILE_BYTES` so an enormous file cannot blow up the model context.
    """
    if meta.get("type") != "file" or "content" not in meta:
        raise GitHubError(
            f"github path {path!r} in {repo} is not a readable file", status=0
        )
    encoding = str(meta.get("encoding", "base64"))
    raw = str(meta.get("content", ""))
    if encoding != "base64":
        raise GitHubError(
            f"github file {path!r} has unexpected encoding {encoding!r}", status=0
        )
    try:
        decoded = base64.b64decode(raw)
    except (ValueError, TypeError):
        raise GitHubError(
            f"github file {path!r} could not be base64-decoded", status=0
        ) from None
    truncated = len(decoded) > _MAX_FILE_BYTES
    text = decoded[:_MAX_FILE_BYTES].decode("utf-8", "replace")
    return {
        "repo": repo,
        "path": str(meta.get("path", path)),
        "sha": str(meta.get("sha", "")),
        "size": int(meta.get("size", len(decoded))),
        "truncated": truncated,
        "content": text,
    }


# ----------------------------------------------------- registry self-registration
def _make_outbound() -> GitHubOutboundConnector:
    """Build the default outbound connector from the secrets/allowlist env.

    Reads the repo allowlist + default repo from the environment so a deployment that only
    sets ``HIMMY_GITHUB_TOKEN`` gets a connector that is capability-OK but refuses every
    call (default-deny) until a repository is allow-listed.
    """
    import os

    allow = [
        r.strip()
        for r in (os.environ.get("HIMMY_GITHUB_ALLOWED_REPOS") or "").split(",")
        if r.strip()
    ]
    return GitHubOutboundConnector(
        allowed_repos=allow or None,
        default_repo=os.environ.get("HIMMY_GITHUB_DEFAULT_REPO"),
    )


def register_github_connectors() -> None:
    """Register the GitHub outbound connector on the process-wide registry.

    Safe to call more than once.
    """
    from himmy.connectors.sdk import register_connector

    register_connector("github", _make_outbound)


__all__ = [
    "GITHUB_API_ROOT",
    "GITHUB_API_HOST",
    "GITHUB_API_VERSION",
    "GITHUB_TOKEN_SECRET",
    "GitHubClient",
    "GitHubError",
    "GitHubRateLimitError",
    "GitHubOutboundConnector",
    "register_github_connectors",
]
