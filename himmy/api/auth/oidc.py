"""OIDC / JWT authentication: verify a Bearer token and map its claims to a Principal.

Validates a JWT against the provider's JWKS — RSA/EC signature, ``iss``, ``aud``, and a
required ``exp`` — then maps claims to a :class:`Principal` (subject, tenants, roles,
scopes). Works with any standards-compliant IdP (Entra ID, Keycloak, Okta, Auth0,
Google). The JWKS is cached with a TTL and refreshed when an unknown ``kid`` appears
(key rotation). Requires the ``auth`` extra (``pyjwt[crypto]``).

The JWKS source is injectable (:class:`StaticJwks` for a fixed key set, :class:`JwksCache`
for a live URL), so the whole verification path is exercised **offline** in tests with a
self-signed key — no IdP or network required.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from himmy.api.auth.base import AuthError, client_ip
from himmy.api.auth.principal import Principal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from collections.abc import Iterable

    from fastapi import Request


@runtime_checkable
class JwksProvider(Protocol):
    """Supplies the current JSON Web Key Set (refreshable for key rotation)."""

    async def get(self, *, force: bool = False) -> dict[str, Any]:
        """Return the JWKS; ``force`` bypasses any cache."""
        ...


class StaticJwks:
    """A fixed JWKS (for tests / a pinned key set)."""

    def __init__(self, jwks: dict[str, Any]) -> None:
        self._jwks = jwks

    async def get(self, *, force: bool = False) -> dict[str, Any]:
        return self._jwks


class JwksCache:
    """Fetch + cache a JWKS from a URL with a TTL (refreshable on key rotation)."""

    def __init__(self, url: str, *, ttl: float = 3600.0, timeout: float = 10.0) -> None:
        self._url = url
        self._ttl = ttl
        self._timeout = timeout
        self._jwks: dict[str, Any] | None = None
        self._fetched_at = float("-inf")

    async def get(self, *, force: bool = False) -> dict[str, Any]:
        fresh = (
            self._jwks is not None and (time.monotonic() - self._fetched_at) < self._ttl
        )
        if not force and fresh:
            assert self._jwks is not None
            return self._jwks
        import httpx

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._url)
            resp.raise_for_status()
            self._jwks = resp.json()
        self._fetched_at = time.monotonic()
        return self._jwks


def _claim_path(claims: dict[str, Any], path: str) -> Any:
    """Read a possibly dotted claim path (e.g. ``realm_access.roles``)."""
    cur: Any = claims
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _as_tokens(value: Any) -> list[str]:
    """Coerce a roles/scopes claim (list, or space/comma string) to a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [t for t in value.replace(",", " ").split() if t]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _as_ids(value: Any) -> list[str]:
    """Coerce a tenant claim (single id or list) to a list (strings not split)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


class OidcAuthenticator:
    """Authenticate a request by a verified OIDC Bearer JWT."""

    #: OIDC always projects a verified subject (+ tenants/roles) from the token, so it
    #: binds callers to an identity the multi-tenant posture (G2) can trust (G1).
    binds_tenants: bool = True

    def __init__(
        self,
        *,
        issuer: str,
        audience: str | list[str],
        jwks: dict[str, Any] | JwksProvider,
        algorithms: Iterable[str] = ("RS256",),
        subject_claim: str = "sub",
        tenant_claim: str | None = None,
        roles_claim: str = "roles",
        scopes_claim: str = "scope",
        all_tenants_roles: Iterable[str] = (),
        leeway: float = 0.0,
    ) -> None:
        """Configure issuer/audience/JWKS + how claims map to the principal."""
        self._issuer = issuer
        self._audience = audience
        self._provider: JwksProvider = (
            StaticJwks(jwks) if isinstance(jwks, dict) else jwks
        )
        self._algorithms = list(algorithms)
        self._subject_claim = subject_claim
        self._tenant_claim = tenant_claim
        self._roles_claim = roles_claim
        self._scopes_claim = scopes_claim
        self._all_tenants_roles = set(all_tenants_roles)
        self._leeway = leeway

    @classmethod
    def from_env(cls) -> OidcAuthenticator:
        """Build from ``HIMMY_OIDC_*`` env vars (JWKS fetched + cached from the IdP)."""
        import os

        issuer = os.environ.get("HIMMY_OIDC_ISSUER")
        audience = os.environ.get("HIMMY_OIDC_AUDIENCE")
        if not issuer or not audience:
            raise AuthError("OIDC auth needs HIMMY_OIDC_ISSUER and HIMMY_OIDC_AUDIENCE")
        jwks_url = os.environ.get("HIMMY_OIDC_JWKS_URL") or (
            f"{issuer.rstrip('/')}/.well-known/jwks.json"
        )
        algorithms = [
            a.strip()
            for a in (os.environ.get("HIMMY_OIDC_ALGORITHMS") or "RS256").split(",")
            if a.strip()
        ]
        admin_roles = [
            r.strip()
            for r in (os.environ.get("HIMMY_OIDC_ADMIN_ROLES") or "").split(",")
            if r.strip()
        ]
        aud: str | list[str] = (
            [a.strip() for a in audience.split(",") if a.strip()]
            if "," in audience
            else audience
        )
        return cls(
            issuer=issuer,
            audience=aud,
            jwks=JwksCache(jwks_url),
            algorithms=algorithms,
            subject_claim=os.environ.get("HIMMY_OIDC_SUBJECT_CLAIM", "sub"),
            tenant_claim=os.environ.get("HIMMY_OIDC_TENANT_CLAIM"),
            roles_claim=os.environ.get("HIMMY_OIDC_ROLES_CLAIM", "roles"),
            scopes_claim=os.environ.get("HIMMY_OIDC_SCOPES_CLAIM", "scope"),
            all_tenants_roles=admin_roles,
        )

    def openapi_security_scheme(self) -> dict[str, dict[str, object]]:
        """Advertise the bearer-JWT scheme in the OpenAPI doc."""
        return {
            "himmyOidc": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
            }
        }

    async def authenticate(self, request: Request) -> Principal:
        """Verify the Bearer token and map its claims to a Principal."""
        import jwt

        token = self._bearer_token(request)
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError as exc:
            raise AuthError(f"malformed token: {exc}") from exc
        key = await self._signing_key(header.get("kid"))
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=self._algorithms,
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={"require": ["exp"]},
            )
        except jwt.PyJWTError as exc:
            raise AuthError(f"token rejected: {exc}") from exc
        return self._principal_from_claims(claims, request)

    def _bearer_token(self, request: Request) -> str:
        """Extract the Bearer token from the Authorization header."""
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            raise AuthError("missing bearer token")
        return token.strip()

    async def _signing_key(self, kid: str | None) -> Any:
        """Find the signing key for ``kid``, refreshing the JWKS on a miss (rotation)."""
        import jwt

        jwks = await self._provider.get()
        key = self._match(jwks, kid)
        if key is None:
            jwks = await self._provider.get(force=True)
            key = self._match(jwks, kid)
        if key is None:
            raise AuthError("no matching signing key for token")
        try:
            return jwt.PyJWK.from_dict(key).key
        except jwt.PyJWTError as exc:  # pragma: no cover - defensive
            raise AuthError(f"invalid signing key: {exc}") from exc

    @staticmethod
    def _match(jwks: dict[str, Any], kid: str | None) -> dict[str, Any] | None:
        """Return the JWK whose ``kid`` matches (or the sole key when no kid)."""
        keys = jwks.get("keys") or []
        for jwk in keys:
            if kid is None or jwk.get("kid") == kid:
                return cast(dict[str, Any], jwk)
        return None

    def _principal_from_claims(
        self, claims: dict[str, Any], request: Request
    ) -> Principal:
        """Project verified claims into a Principal (subject/tenants/roles/scopes)."""
        subject = str(claims.get(self._subject_claim) or claims.get("sub") or "")
        if not subject:
            raise AuthError("token has no subject claim")
        tenants = (
            _as_ids(_claim_path(claims, self._tenant_claim))
            if self._tenant_claim
            else []
        )
        roles = _as_tokens(_claim_path(claims, self._roles_claim))
        scopes = _as_tokens(claims.get(self._scopes_claim) or claims.get("scp"))
        all_tenants = bool(self._all_tenants_roles & set(roles))
        return Principal.build(
            subject=subject,
            tenant_ids=tenants,
            roles=roles,
            scopes=scopes,
            all_tenants=all_tenants,
            auth_method="oidc",
            source_ip=client_ip(request),
            claims=claims,
        )


__all__ = [
    "OidcAuthenticator",
    "JwksProvider",
    "StaticJwks",
    "JwksCache",
]
