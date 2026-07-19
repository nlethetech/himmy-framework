"""API-key authentication: a header key → a Principal (hashed, expiring, revocable).

Two modes, both timing-safe and both stored as a salted HASH in memory (never the
plaintext key — a heap/core dump or a logged repr leaks no usable credential):

* **Shared trusted-boundary key** (back-compat): one or more keys in
  ``HIMMY_INTERNAL_API_KEY`` (comma-separated for rotation) each map to an
  unrestricted, all-tenants principal — the existing "internal trusted boundary"
  behavior. It does NOT bind a caller to a tenant.
* **Mapped keys** (closes the cross-tenant hole): a JSON map of key → tenants/roles
  (``HIMMY_API_KEYS_FILE``) yields a Principal *bound* to specific tenants, so a key
  scoped to tenant A cannot read tenant B.

Lifecycle hardening (P1):

* Every configured key is reduced at construction to a :class:`KeyRecord` carrying a
  non-secret ``key_id`` (a short fingerprint prefix for O(1) lookup + logs) and an
  ``hmac-sha256`` of the secret under a server *pepper* (``HIMMY_API_KEY_PEPPER``,
  resolved through the secrets provider). The raw key is never retained — only its
  hash, compared with :func:`hmac.compare_digest` so the match stays constant-time.
* Per-key metadata (``expires_at`` / ``disabled``) is enforced at
  :meth:`ApiKeyAuthenticator.authenticate`: an expired or disabled key is a 401, not
  a silent pass.
* A **live-consulted revocation file** (``HIMMY_API_KEY_REVOCATION_FILE``) lets a
  leaked key die WITHOUT a restart: it is a JSON list of revoked ``key_id`` strings
  re-read (mtime-cached) on each authenticate, so dropping a key_id into it kills the
  key on the next request. To keep steady-state key auth a cheap in-memory check (no
  synchronous ``stat()`` on the async event loop per request), the freshness ``stat()``
  can be THROTTLED to at most once per ``revocation_stat_interval`` seconds. This is
  OPT-IN and OFF by default (``_REVOCATION_STAT_INTERVAL_S = 0.0``) so revoke-on-next-
  authenticate stays immediate/synchronous — the safe default for an auth surface. A
  high-QPS deployment that profiles the per-request ``stat()`` as a bottleneck (e.g. on a
  slow/NFS revocation path) can pass ``revocation_stat_interval=1.0`` to re-stat at most
  once/sec, accepting a bounded ~1s revocation-propagation staleness in exchange.
  If a CONFIGURED file later becomes unreadable (deleted,
  truncated, corrupted, or ``0600`` and written by a different user than the server —
  e.g. ``himmy apikey revoke`` run as another account), the control does NOT silently
  disarm: the LAST-KNOWN revoked set is retained (a transient blip can never
  un-revoke a key) and a loud ``WARNING`` is logged. Set
  ``HIMMY_API_KEY_REVOCATION_FAIL_CLOSED=1`` to instead 401 all key auth while the
  configured file is unreadable. NOTE: the revocation file must be readable by the
  server process — if the CLI writes it ``0600`` as another user, either run the CLI
  as the server user or relax the file mode.

INVARIANT — offline path unchanged: when no key is configured ``build_authenticator``
returns ``None`` and every request is ANONYMOUS (all-tenants); none of this engages.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from himmy.api.auth.base import AuthError, client_ip
from himmy.api.auth.principal import Principal
from himmy.config.flags import env_truthy

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import Request

logger = logging.getLogger("himmy.api.auth.apikey")

#: Default header carrying the API key.
DEFAULT_HEADER = "x-himmy-internal-key"

#: Secret name resolved through the secrets provider for the HMAC pepper. A pepper is
#: a server-side secret mixed into every key hash so a stolen hash store is useless
#: without ALSO stealing the pepper (which lives in env/keychain/vault, not the key
#: file). When unset the hash falls back to an unkeyed digest — still non-reversible,
#: just without the extra defense — so the offline/zero-config path needs no setup.
PEPPER_SECRET = "HIMMY_API_KEY_PEPPER"  # noqa: S105 - secret NAME, not a value

#: Env var naming the live-consulted revocation file (a JSON list of revoked key_ids).
REVOCATION_FILE_ENV = "HIMMY_API_KEY_REVOCATION_FILE"

#: Opt-in env flag: when truthy AND a revocation file IS configured but cannot be
#: read/parsed, key authentication FAILS CLOSED (every key 401s) instead of falling
#: back to the last-known revoked set. Off by default so the historical behavior
#: (transient blips do not lock everyone out) is preserved unless an operator opts in.
REVOCATION_FAIL_CLOSED_ENV = "HIMMY_API_KEY_REVOCATION_FAIL_CLOSED"

#: Minimum wall-time (seconds, monotonic) between successive ``stat()`` probes of the
#: revocation file on the async auth hot path. When SET (> 0) a healthy, already-loaded
#: list is trusted in-memory and the file is re-stat'd (and re-read only if its fingerprint
#: changed) at most once per interval — bounding 'revoke-on-next-authenticate' staleness to
#: ~this interval. DEFAULT 0.0 = OFF: revocation stays immediate/synchronous, the safe
#: default for an auth surface. Opt in (e.g. ``revocation_stat_interval=1.0``) only where a
#: per-request ``stat()`` is a profiled bottleneck (high QPS on a slow/NFS revocation path).
_REVOCATION_STAT_INTERVAL_S = 0.0


def _env_truthy(name: str) -> bool:
    """Whether env var ``name`` is set to a truthy value (1/true/yes/on/y).

    Thin alias over the canonical :func:`himmy.config.flags.env_truthy` so this
    module's fail-closed switch (``HIMMY_API_KEY_REVOCATION_FAIL_CLOSED``) shares the
    one accepted truthy vocabulary with every other posture flag.
    """
    return env_truthy(name)


#: Default roles a DEMOTED shared key receives under the multi-tenant posture (G1).
#: ``operator`` reads + writes the operational surface (run/agent/knowledge/model/…)
#: but holds NO tenant_ids and is ``all_tenants=False``, so ``resolve_workspace``
#: 403s it off any tenant's data — it is useful for ops/diagnostics, not a tenant
#: super-user. (Verified against ``DEFAULT_RBAC``: ``operator`` grants run/agent/
#: knowledge/model/diagnostics read+write, so the demoted key is NOT 403-everywhere.)
DEMOTED_SHARED_KEY_ROLES: tuple[str, ...] = ("operator",)


#: Process-lifetime memo of the resolved HMAC pepper. ``None`` = not yet resolved
#: (distinct from a resolved-but-empty ``b""`` pepper — the valid unkeyed-fallback
#: case), so the sentinel never collapses with "no pepper configured". Cleared by
#: :func:`reset_pepper_cache` (fired by ``configure_secrets`` on a provider swap).
_PEPPER_CACHE: bytes | None = None


def _pepper() -> bytes:
    """The server HMAC pepper (``HIMMY_API_KEY_PEPPER``) as bytes, or empty if unset.

    Resolved lazily (not at import) through the secrets provider so keychain/vault
    backends and test monkeypatching both work, then MEMOIZED for the process lifetime:
    the pepper is fixed once the provider is chosen, so resolving it per authenticate
    (via :func:`hash_key`) would pay a fresh secrets-provider round-trip — a keychain /
    AWS / GCP / Azure call — on every request. :func:`reset_pepper_cache` (fired by
    ``configure_secrets``) drops the memo so a provider swap re-resolves. Empty ⇒
    unkeyed digest fallback.
    """
    global _PEPPER_CACHE
    cached = _PEPPER_CACHE
    if cached is not None:
        return cached
    from himmy.config.secrets import get_secret

    resolved = (get_secret(PEPPER_SECRET) or "").encode("utf-8")
    _PEPPER_CACHE = resolved
    return resolved


def reset_pepper_cache() -> None:
    """Clear the memoized HMAC pepper so the next hash re-resolves it through the provider.

    Fired by :func:`himmy.config.secrets.configure_secrets` on a provider swap (so the
    new backend's pepper takes effect) and available as a test hook. Mirrors
    :func:`himmy.licensing.reset_license_cache`.
    """
    global _PEPPER_CACHE
    _PEPPER_CACHE = None


def hash_key(secret: str) -> str:
    """Return the salted, non-reversible hash stored for ``secret`` (hex).

    ``hmac-sha256(pepper, secret)`` when a pepper is configured, else a plain
    ``sha256(secret)`` digest. Either way the result is one-way: the stored value can
    verify a presented key (constant-time) but cannot reconstruct it.
    """
    pepper = _pepper()
    if pepper:
        return hmac.new(pepper, secret.encode("utf-8"), hashlib.sha256).hexdigest()
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()


def _fingerprint(key: str) -> str:
    """A short, non-reversible tag for a key (for the subject id / key_id / logs)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _parse_expiry(value: object) -> float | None:
    """Coerce an ``expires_at`` spec (ISO-8601 string or epoch number) to epoch seconds.

    Returns ``None`` for a missing/empty value (no expiry). Accepts a trailing ``Z``
    and naive ISO strings (treated as UTC) so hand-written key files are forgiving.
    """
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.isdigit():
        return float(text)
    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.timestamp()


@dataclass(frozen=True)
class KeyRecord:
    """A configured key reduced to non-secret metadata + a one-way hash.

    The raw key is never stored: ``key_hash`` is the salted ``hmac-sha256`` digest
    (see :func:`hash_key`) and ``key_id`` is a short non-secret fingerprint used for
    O(1) revocation lookups and logs. ``principal`` is the verified identity minted on
    a match; ``expires_at`` (epoch seconds, ``None`` = never) and ``disabled`` are the
    lifecycle gates enforced at authenticate time.
    """

    key_id: str
    key_hash: str
    principal: Principal
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    disabled: bool = False

    def is_expired(self, *, now: float | None = None) -> bool:
        """Whether the key's ``expires_at`` is in the past (``None`` ⇒ never)."""
        if self.expires_at is None:
            return False
        return (now if now is not None else time.time()) >= self.expires_at

    @classmethod
    def from_secret(
        cls,
        secret: str,
        principal: Principal,
        *,
        expires_at: float | None = None,
        disabled: bool = False,
    ) -> KeyRecord:
        """Build a record from a raw ``secret`` (hashed here; the secret is not kept)."""
        return cls(
            key_id=_fingerprint(secret),
            key_hash=hash_key(secret),
            principal=principal,
            expires_at=expires_at,
            disabled=disabled,
        )


class _RevocationList:
    """A live-consulted set of revoked ``key_id`` strings backed by a JSON file.

    Re-reads the file only when its mtime changes (so the hot path is a stat, not a
    parse), giving "kill a key without a restart": appending a key_id to the file
    revokes it on the next authenticate. When no file is configured this is inert
    (``is_revoked`` is always ``False``).

    Fail behavior when a file IS configured but a refresh fails (deleted, truncated,
    corrupted, or unreadable to the server process — e.g. ``himmy apikey revoke`` ran
    as another user and wrote it ``0600``):

    * The control NO LONGER silently disarms. The LAST-KNOWN revoked set is RETAINED
      across the failure, so a transient blip (or a cross-user read error) can never
      un-revoke an already-revoked key. (A file that has NEVER been read successfully
      has an empty last-known set, so a configured-but-absent file is still inert —
      a key still has to be otherwise valid to authenticate.)
    * The failure is logged loudly ONCE per distinct error (deduplicated so the hot
      path does not spam the log every request), so operators notice that the
      configured revocation file is unavailable instead of it being swallowed.
    * If :data:`REVOCATION_FAIL_CLOSED_ENV` is opted in, :meth:`is_revoked` RAISES
      :class:`AuthError` while the configured file is unreadable, so key auth fails
      CLOSED (every key 401s) rather than relying on the last-known set.
    """

    def __init__(
        self, path: str | Path | None, *, stat_interval: float = _REVOCATION_STAT_INTERVAL_S
    ) -> None:
        self._path = Path(path).expanduser() if path else None
        #: Min seconds between ``stat()`` probes (see ``_REVOCATION_STAT_INTERVAL_S``).
        #: Pass ``0.0`` to re-stat on EVERY consult (synchronous-freshness semantics —
        #: e.g. unit tests that revoke and immediately re-check within the same instant).
        self._stat_interval = stat_interval
        #: The composite freshness fingerprint of the last successful read. Keyed on
        #: ``(st_mtime_ns, st_size, st_ino)`` rather than ``st_mtime`` alone so a content
        #: rewrite that PRESERVES the mtime — an atomic replace that copies the source
        #: mtime, an ``rsync --times`` / ``cp -p`` / ``git checkout``, or two writes within
        #: the filesystem's coarse mtime granularity — still changes the size or inode and
        #: is therefore observed, instead of silently serving the stale revoked set (which
        #: would let an already-revoked, leaked key keep authenticating).
        self._fingerprint: tuple[int, int, int] | None = None
        self._revoked: frozenset[str] = frozenset()
        #: True once the file has been read+parsed successfully at least once.
        self._ever_loaded = False
        #: Last error string surfaced, so we log a given failure loudly only once.
        self._last_error: str | None = None
        #: Monotonic timestamp of the last ``stat()`` probe; throttles the synchronous
        #: FS syscall off the async auth hot path (see ``_REVOCATION_STAT_INTERVAL_S``).
        self._last_stat_at: float = 0.0

    def is_revoked(self, key_id: str) -> bool:
        """Whether ``key_id`` is in the current (freshly re-read) revocation set.

        Raises :class:`AuthError` (fail-closed) when the configured file is currently
        unreadable AND :data:`REVOCATION_FAIL_CLOSED_ENV` is opted in; otherwise the
        last-known revoked set is consulted (fail-static, never fail-open).
        """
        if self._path is None:
            return False
        ok = self._refresh()
        if not ok and _env_truthy(REVOCATION_FAIL_CLOSED_ENV):
            raise AuthError("revocation list unavailable")
        return key_id in self._revoked

    def _refresh(self) -> bool:
        """Re-read the file if its stat fingerprint changed. Returns whether trusted.

        ``True`` ⇒ the current ``_revoked`` reflects a successful read (or an unchanged
        ``(st_mtime_ns, st_size, st_ino)`` fingerprint since the last success). ``False`` ⇒
        a stat/read/parse failure on a CONFIGURED file; ``_revoked`` is left at its
        last-known value (NOT cleared) and the failure is logged once.

        The fingerprint is composite (not bare ``st_mtime``) so a same-mtime content rewrite
        — an mtime-preserving sync/restore, or two edits within the filesystem's mtime
        granularity — still flips ``st_size``/``st_ino`` and is re-read, instead of serving
        a stale revoked set that would keep an already-revoked key authenticating.
        """
        assert self._path is not None
        # Throttle the synchronous ``stat()`` off the async auth hot path: once the list
        # has loaded cleanly, trust the in-memory revoked set and skip the FS syscall
        # until the interval elapses, so steady-state key auth is a cheap memory check.
        # (Skipped only while healthy — a pending error keeps re-probing, bounded by the
        # same interval, so recovery is still prompt without spamming the event loop.)
        now = time.monotonic()
        if (
            self._stat_interval > 0
            and self._ever_loaded
            and self._last_error is None
            and (now - self._last_stat_at) < self._stat_interval
        ):
            return True
        self._last_stat_at = now
        try:
            st = self._path.stat()
            fingerprint = (st.st_mtime_ns, st.st_size, st.st_ino)
        except OSError as exc:
            return self._on_failure(f"cannot stat revocation file {self._path}: {exc}")
        if fingerprint == self._fingerprint and self._ever_loaded:
            return True
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            return self._on_failure(
                f"cannot read/parse revocation file {self._path}: {exc}"
            )
        if isinstance(raw, dict):  # tolerate {"revoked": [...]} as well as a bare list
            raw = raw.get("revoked", [])
        if isinstance(raw, list):
            self._revoked = frozenset(str(item) for item in raw)
        else:
            self._revoked = frozenset()
        self._fingerprint = fingerprint
        self._ever_loaded = True
        if self._last_error is not None:
            logger.warning("revocation file recovered: %s", self._path)
            self._last_error = None
        return True

    def _on_failure(self, message: str) -> bool:
        """Retain the last-known set, log the failure once, and report 'not trusted'.

        Emits a loud ``WARNING`` the first time a given error message is seen (and on
        every distinct new error) so an operator notices the security control is
        degraded, without spamming the per-request hot path. The last-known revoked
        set is deliberately NOT cleared — a transient failure must not un-revoke a key.
        """
        if message != self._last_error:
            logger.warning(
                "API-key revocation list unavailable (retaining last-known %d "
                "revoked id(s); set %s=1 to fail closed): %s",
                len(self._revoked),
                REVOCATION_FAIL_CLOSED_ENV,
                message,
            )
            self._last_error = message
        return False


class ApiKeyAuthenticator:
    """Authenticate a request by a header API key (shared or tenant-mapped).

    Keys are held only as :class:`KeyRecord` hashes; the match is constant-time and
    each authenticate consults per-key expiry/disabled state plus a live revocation
    file, so a key can be aged out or killed without restarting the process.
    """

    def __init__(
        self,
        *,
        shared_keys: set[str] | None = None,
        key_principals: dict[str, Principal] | None = None,
        header_name: str = DEFAULT_HEADER,
        shared_key_roles: tuple[str, ...] | None = None,
        records: list[KeyRecord] | None = None,
        revocation_path: str | Path | None = None,
        revocation_stat_interval: float = _REVOCATION_STAT_INTERVAL_S,
    ) -> None:
        """Wire shared keys (→ all-tenants) and/or mapped keys (→ bound principals).

        ``shared_key_roles`` controls the posture of a *shared* (unmapped) key match:

        * ``None`` (default) keeps the historical single-box behavior — a shared key
          maps to an unrestricted ``all_tenants`` ``admin`` principal (the "internal
          trusted boundary").
        * A concrete role tuple (e.g. :data:`DEMOTED_SHARED_KEY_ROLES`) DEMOTES the
          shared key to a tenant-bound-by-absence principal: ``all_tenants=False``,
          no ``tenant_ids``, and exactly those roles. The multi-tenant fail-closed
          posture (G2) passes this so a shared-key match can no longer act as a
          cross-tenant admin, while remaining useful for ops/diagnostics routes.

        Mapped keys (``key_principals``) are unaffected by this — they always bind to
        their declared tenants.

        ``records`` lets a caller supply pre-built :class:`KeyRecord`\\ s carrying
        lifecycle metadata (expiry/disabled) — the mapped-key file loader uses this to
        thread per-key ``expires_at`` / ``disabled`` through. ``revocation_path`` names
        the live-consulted revocation file (defaults to ``HIMMY_API_KEY_REVOCATION_FILE``
        when omitted).
        """
        self._header = header_name
        self._shared_key_roles = shared_key_roles
        self._records: list[KeyRecord] = list(records or [])
        # ``binds_tenants`` is True iff any non-shared (mapped/record) key is present —
        # matching the historical ``bool(self._mapped)``: it does NOT inspect tenant_ids
        # (an all-tenants admin mapped key still counts as a binding authenticator).
        self._mapped_present = bool(key_principals) or bool(records)

        # Mapped keys first (a key bound to tenants is more specific than a shared one),
        # mirroring the historical lookup order. Each is reduced to a hashed record.
        for key, principal in (key_principals or {}).items():
            self._records.append(KeyRecord.from_secret(key, principal))

        # Shared keys mint a posture-appropriate principal on match (built once here so
        # the per-request path only does a hash compare + lifecycle check).
        roles = self._shared_key_roles
        demoted = roles is not None
        for key in shared_keys or set():
            shared_principal = Principal.build(
                subject=f"apikey:{_fingerprint(key)}",
                all_tenants=not demoted,
                roles=roles if roles is not None else ("admin",),
                auth_method="apikey",
            )
            self._records.append(KeyRecord.from_secret(key, shared_principal))

        if revocation_path is None:
            revocation_path = os.environ.get(REVOCATION_FILE_ENV)
        # ``revocation_stat_interval`` throttles the synchronous ``stat()`` off the async
        # auth hot path (steady-state auth is then a cheap in-memory check); pass ``0.0``
        # for synchronous-freshness semantics (revoke observed on the very next consult).
        self._revocations = _RevocationList(
            revocation_path, stat_interval=revocation_stat_interval
        )

        if not self._records:
            raise AuthError("ApiKeyAuthenticator needs at least one configured key")

    @property
    def binds_tenants(self) -> bool:
        """Whether this authenticator binds callers to concrete tenants (G1).

        True iff at least one tenant-mapped key is configured — those principals
        carry ``tenant_ids`` and close the cross-tenant hole. A shared-key-ONLY
        authenticator binds nobody (every shared match is all-tenants or, when
        demoted, tenant-LESS), so it returns ``False`` and the multi-tenant posture
        (G2) refuses to start on it.
        """
        return self._mapped_present

    @property
    def binds_concrete_tenants(self) -> bool:
        """Whether ANY configured key is scoped to a concrete tenant (stricter than G1).

        ``binds_tenants`` counts a mapped/record key as tenant-binding even when it is an
        all-tenants admin — so a keys FILE holding ONLY all-tenants records satisfies it,
        giving false assurance under a DECLARED multi-tenant posture (every caller would still
        be an all-tenants admin, with no tenant to isolate from). This is the honest stricter
        test the ``HIMMY_MULTI_TENANT`` guard uses: True only when at least one record carries
        a non-empty ``tenant_ids`` (``all_tenants=False`` with concrete tenants). A file of
        purely all-tenants records returns ``False`` here even though ``binds_tenants`` is True.
        """
        return any(
            not record.principal.all_tenants and bool(record.principal.tenant_ids)
            for record in self._records
        )

    def openapi_security_scheme(self) -> dict[str, dict[str, object]]:
        """Advertise the API-key header as an OpenAPI security scheme (for docs)."""
        return {
            "himmyApiKey": {
                "type": "apiKey",
                "in": "header",
                "name": self._header,
            }
        }

    async def authenticate(self, request: Request) -> Principal:
        """Resolve the key header to a Principal (raises AuthError if invalid).

        Beyond the constant-time hash match, each candidate is gated on its lifecycle:
        a disabled, expired, or live-revoked key is rejected with a 401 even though the
        secret itself is correct — so a leaked key dies the moment it is revoked/ages
        out, with no process restart.
        """
        provided = request.headers.get(self._header)
        if not provided:
            raise AuthError("missing api key")
        ip = client_ip(request)
        provided_hash = hash_key(provided)
        now = time.time()
        for record in self._records:
            if not hmac.compare_digest(provided_hash, record.key_hash):
                continue
            if record.disabled:
                raise AuthError("api key disabled")
            if record.is_expired(now=now):
                raise AuthError("api key expired")
            if self._revocations.is_revoked(record.key_id):
                raise AuthError("api key revoked")
            return _with_ip(record.principal, ip)
        raise AuthError("invalid api key")


def _with_ip(principal: Principal, ip: str | None) -> Principal:
    """Return ``principal`` stamped with the request source IP."""
    if ip is None or principal.source_ip == ip:
        return principal
    return Principal(
        subject=principal.subject,
        tenant_ids=principal.tenant_ids,
        roles=principal.roles,
        scopes=principal.scopes,
        all_tenants=principal.all_tenants,
        auth_method=principal.auth_method,
        source_ip=ip,
        claims=principal.claims,
        subject_scoped=principal.subject_scoped,
    )


def load_key_principals(path: str | Path) -> dict[str, Principal]:
    """Load a key→Principal map from a JSON file (back-compat surface).

    Schema: ``{"<key>": {"subject": str, "tenant_ids": [..], "roles": [..]}}``. Each
    entry yields a Principal bound to those tenants (auth_method ``apikey``). This
    drops any lifecycle metadata; :func:`load_key_records` is the lifecycle-aware
    loader the authenticator uses. Kept for callers/tests that only want the mapping.
    """
    out: dict[str, Principal] = {}
    for key, record in _load_records(path).items():
        out[key] = record[0]
    return out


def load_key_records(path: str | Path) -> list[KeyRecord]:
    """Load lifecycle-aware :class:`KeyRecord`\\ s from a JSON key file.

    Extends the back-compat schema with optional per-key ``expires_at`` (ISO-8601 or
    epoch seconds) and ``disabled`` (bool). The plaintext key in the file is hashed
    here and never retained in the returned records — only its hash + non-secret
    ``key_id``. The file itself should be ``chmod 0600`` and, ideally, sourced from a
    secrets manager; the hashing protects the IN-MEMORY copy regardless.
    """
    records: list[KeyRecord] = []
    for key, (principal, expires_at, disabled) in _load_records(path).items():
        records.append(
            KeyRecord.from_secret(
                key, principal, expires_at=expires_at, disabled=disabled
            )
        )
    return records


def _load_records(
    path: str | Path,
) -> dict[str, tuple[Principal, float | None, bool]]:
    """Parse the JSON key file into ``{key: (principal, expires_at, disabled)}``."""
    raw = json.loads(Path(path).expanduser().read_text())
    if not isinstance(raw, dict):
        raise AuthError(f"api-keys file {path} must be a JSON object")
    out: dict[str, tuple[Principal, float | None, bool]] = {}
    for key, spec in raw.items():
        spec = spec or {}
        principal = Principal.build(
            subject=str(spec.get("subject") or f"apikey:{_fingerprint(str(key))}"),
            tenant_ids=[str(t) for t in (spec.get("tenant_ids") or [])],
            roles=[str(r) for r in (spec.get("roles") or [])],
            scopes=[str(s) for s in (spec.get("scopes") or [])],
            all_tenants=bool(spec.get("all_tenants", False)),
            auth_method="apikey",
            subject_scoped=bool(spec.get("subject_scoped", False)),
        )
        out[str(key)] = (
            principal,
            _parse_expiry(spec.get("expires_at")),
            bool(spec.get("disabled", False)),
        )
    return out


__all__ = [
    "ApiKeyAuthenticator",
    "KeyRecord",
    "load_key_principals",
    "load_key_records",
    "hash_key",
    "reset_pepper_cache",
    "DEFAULT_HEADER",
    "DEMOTED_SHARED_KEY_ROLES",
    "PEPPER_SECRET",
    "REVOCATION_FILE_ENV",
    "REVOCATION_FAIL_CLOSED_ENV",
]
