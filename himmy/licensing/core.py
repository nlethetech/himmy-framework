"""Offline Enterprise Edition licensing: the monetization seam.

Himmy's OSS / offline / single-tenant CORE is always FREE and byte-unchanged. This
module gates only the ENTERPRISE-specific surfaces (OIDC SSO, signed audit export, the
enterprise console, …) behind a signed license.

A license is a compact string ``base64url(json_payload)"."base64url(ed25519_sig)`` where
the signature is over the RAW payload bytes. It is verified FULLY OFFLINE against a
baked-in Ed25519 PUBLIC key (:data:`LICENSE_PUBLIC_KEY_HEX`) — there is NO network call,
NO phone-home, so an air-gapped install verifies exactly the same as a connected one. The
vendor's PRIVATE signing key never ships in the repo (see ``scripts/mint_license.py``).

Everything FAILS CLOSED to :data:`Edition.COMMUNITY`: a missing, malformed, tampered, or
expired license — or any verification fault — resolves to community, never to enterprise.
Using a gated feature without the matching entitlement raises a clear
:class:`EnterpriseFeatureError` telling the operator how to install a license; it never
crashes and is never silently allowed.
"""

from __future__ import annotations

import base64
import binascii
import enum
import json
import os
import time
from dataclasses import dataclass

from himmy.core.errors import HimmyError

#: Baked-in vendor Ed25519 PUBLIC key (hex, 32-byte raw) that every license signature is
#: verified against. This is the PUBLIC half of the vendor keypair — safe to commit. The
#: PRIVATE half lives only on the vendor's admin machine (``~/.himmy-admin/license_ed25519``)
#: and is NEVER in the repo. Rotating it means shipping a new build with a new public key.
LICENSE_PUBLIC_KEY_HEX = "e14eb685c0f46f9fb629e975b9212c03e82d0744b26e39088214dd8ead378a2e"

#: Env var carrying an inline license string (takes precedence over the license file).
HIMMY_LICENSE_KEY_ENV = "HIMMY_LICENSE_KEY"

#: Secret/config name the license file resolves under (``~/.himmy/license`` by default).
LICENSE_SECRET_NAME = "HIMMY_LICENSE"  # noqa: S105 - a secret NAME, not a credential


class Edition(str, enum.Enum):
    """The two Himmy editions. Anything but a valid EE license is COMMUNITY."""

    COMMUNITY = "community"
    ENTERPRISE = "enterprise"


# -- the gated-feature registry (extensible constant) -------------------------------------
#: OIDC / SSO enterprise authenticator build path.
FEATURE_OIDC_SSO = "oidc_sso"
#: Tamper-evident signed audit-bundle export (CLI + route).
FEATURE_AUDIT_EXPORT = "audit_export"
#: Custom RBAC policy authoring beyond the built-in roles.
FEATURE_CUSTOM_RBAC_POLICY = "custom_rbac_policy"
#: Reproducible air-gap deployment bundle builder.
FEATURE_AIRGAP_BUNDLE = "airgap_bundle"
#: The governance / cost enterprise console (WP2).
FEATURE_ENTERPRISE_CONSOLE = "enterprise_console"

#: The starter set of Enterprise-gated features. Extend this as EE surfaces land; a
#: feature string not in here is not gate-able (``require_entitlement`` on it always
#: raises, since community never carries unknown features either).
FEATURE_REGISTRY: frozenset[str] = frozenset(
    {
        FEATURE_OIDC_SSO,
        FEATURE_AUDIT_EXPORT,
        FEATURE_CUSTOM_RBAC_POLICY,
        FEATURE_AIRGAP_BUNDLE,
        FEATURE_ENTERPRISE_CONSOLE,
    }
)


class EnterpriseFeatureError(HimmyError):
    """Raised when a gated Enterprise feature is used without the entitlement.

    Carries the offending ``feature`` and a message that tells the operator exactly how to
    unlock it, so a community user gets an actionable upgrade prompt rather than a crash.
    """

    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(
            f"{feature} is an Enterprise Edition feature — install a license: "
            f"himmy license install <key>"
        )


@dataclass(frozen=True)
class License:
    """A verified license payload (see the module docstring for the wire format)."""

    license_id: str
    edition: Edition
    features: frozenset[str]
    customer: str
    issued_at: int
    expires_at: int

    def is_expired(self, *, now: float | None = None) -> bool:
        """True when ``expires_at`` is in the past (0 means never expires)."""
        if not self.expires_at:
            return False
        return (now if now is not None else time.time()) >= self.expires_at


def _b64url_decode(segment: str) -> bytes:
    """URL-safe base64 decode tolerant of stripped ``=`` padding."""
    pad = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + pad)


def _b64url_encode(raw: bytes) -> str:
    """URL-safe base64 encode WITHOUT padding (matches the mint tool)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def verify_license(
    token: str | None, *, public_key_hex: str | None = None
) -> License | None:
    """Verify a license token OFFLINE; return the :class:`License` or ``None``.

    Splits ``payload"."signature``, checks the Ed25519 signature over the raw payload
    bytes with the baked-in public key, then decodes the JSON. Returns ``None`` — never
    raises — for a missing, malformed, wrongly-signed, or structurally invalid token, so
    every fault path lands on community (fail-closed). ``cryptography`` is imported lazily
    so this module still loads in an install without the extra (then verification denies).
    """
    # Read the baked-in key at call time (not def time) so a rebuild — or a test — that
    # swaps LICENSE_PUBLIC_KEY_HEX is honoured.
    if public_key_hex is None:
        public_key_hex = LICENSE_PUBLIC_KEY_HEX
    if not token or not public_key_hex:
        return None
    token = token.strip()
    if token.count(".") != 1:
        return None
    payload_b64, sig_b64 = token.split(".", 1)
    try:
        payload_raw = _b64url_decode(payload_b64)
        signature = _b64url_decode(sig_b64)
    except (binascii.Error, ValueError):
        return None

    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError:  # pragma: no cover - only without the crypto extra
        # No backend: we cannot verify, so we MUST deny (fail closed to community).
        return None
    try:
        key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
    except (ValueError, TypeError):
        return None
    try:
        key.verify(signature, payload_raw)
    except InvalidSignature:
        return None
    except Exception:  # noqa: BLE001 - any verification fault is a deny, never a crash
        return None

    try:
        data = json.loads(payload_raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        edition = Edition(str(data.get("edition")))
    except ValueError:
        return None
    features = data.get("features") or []
    if not isinstance(features, list):
        return None
    try:
        return License(
            license_id=str(data["license_id"]),
            edition=edition,
            features=frozenset(str(f) for f in features),
            customer=str(data.get("customer", "")),
            issued_at=int(data.get("issued_at", 0)),
            expires_at=int(data.get("expires_at", 0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _load_license_token() -> str | None:
    """Resolve the raw license token: env first, then the license file / secrets layer.

    ``HIMMY_LICENSE_KEY`` wins when set. Otherwise we read ``~/.himmy/license`` directly
    (its contents may itself be a path to a license file, supporting ``install <file>``),
    falling back to the secrets provider under :data:`LICENSE_SECRET_NAME` so an operator
    can stash it in a vault/keychain like any other secret.
    """
    inline = os.environ.get(HIMMY_LICENSE_KEY_ENV)
    if inline and inline.strip():
        return inline.strip()

    path = license_file_path()
    if path and os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as handle:
                text = handle.read().strip()
        except OSError:
            text = ""
        if text:
            # Allow the file to contain a token OR a path to a token file.
            if "." not in text and os.path.isfile(os.path.expanduser(text)):
                try:
                    with open(os.path.expanduser(text), encoding="utf-8") as handle:
                        return handle.read().strip() or None
                except OSError:
                    return None
            return text

    try:
        from himmy.config.secrets import get_secret

        secret = get_secret(LICENSE_SECRET_NAME)
    except Exception:  # noqa: BLE001 - secrets backend faults must not escalate to enterprise
        secret = None
    return secret.strip() if secret and secret.strip() else None


def license_file_path() -> str:
    """The default on-disk license path (``~/.himmy/license``)."""
    from pathlib import Path

    return str(Path.home() / ".himmy" / "license")


@dataclass(frozen=True)
class _Resolution:
    """The resolution of the active license posture at a point in time."""

    edition: Edition
    entitlements: frozenset[str]
    license: License | None
    reason: str


@dataclass(frozen=True)
class _Verified:
    """The cached, token-keyed result of the (relatively costly) SIGNATURE verification.

    We cache only what is invariant for a given raw token — the Ed25519-verified
    :class:`License` (or ``None`` for a bad/absent token). We do NOT cache the final
    edition/entitlement verdict, because that depends on wall-clock time (expiry) and must
    be re-evaluated live on every read. Keying on the raw token also makes install / remove
    / rotation live: a changed token source misses the cache and re-verifies, so a
    running server honours a removed or swapped license without a restart.
    """

    token: str | None
    license: License | None


_CACHE: _Verified | None = None


def _verified_license() -> tuple[str | None, License | None]:
    """Return the current ``(token, verified License|None)``, caching only the verify step.

    The raw token source is read on EVERY call (cheap: env/file/secret) so install, removal
    and rotation are picked up live. The expensive Ed25519 verification is memoized, but
    ONLY while the raw token is byte-identical to the cached one — a different (or absent)
    token invalidates the cache and re-verifies. Crucially this returns the parsed License
    regardless of expiry; expiry is evaluated by :func:`_resolve` at read time.
    """
    global _CACHE
    token = _load_license_token()
    cached = _CACHE
    if cached is not None and cached.token == token:
        return cached.token, cached.license
    lic = verify_license(token) if token else None
    _CACHE = _Verified(token=token, license=lic)
    return token, lic


def _resolve() -> _Resolution:
    """Resolve the active edition + entitlements, fail-closed, with LIVE expiry.

    Every failure mode — no token, bad signature, wrong edition, expired, or a decode
    fault — resolves to community with empty entitlements. Only a validly-signed,
    unexpired, enterprise-edition license grants entitlements (intersected with the
    registry so a forged feature name a signer never should mint still can't leak).

    Expiry is re-checked on every call against the current wall clock, so a license that
    expires mid-process stops granting Enterprise the instant it lapses — the cache holds
    only the token-keyed signature verification, never the time-sensitive verdict. This is
    still fully offline: no network, no phone-home (Ed25519 verify is microseconds and the
    token source read is a local env/file/secret lookup).
    """
    token, lic = _verified_license()
    if not token:
        return _Resolution(Edition.COMMUNITY, frozenset(), None, "no-license")
    if lic is None:
        return _Resolution(Edition.COMMUNITY, frozenset(), None, "invalid-signature")
    if lic.edition is not Edition.ENTERPRISE:
        return _Resolution(Edition.COMMUNITY, frozenset(), lic, "not-enterprise")
    if lic.is_expired():
        return _Resolution(Edition.COMMUNITY, frozenset(), lic, "expired")

    entitlements = frozenset(lic.features) & FEATURE_REGISTRY
    return _Resolution(Edition.ENTERPRISE, entitlements, lic, "ok")


def reset_license_cache() -> None:
    """Clear the verification cache so the next lookup re-reads + re-verifies the token.

    A test hook / post-install nicety. Note that correctness no longer DEPENDS on this
    being called: the token source is re-read on every resolution and expiry is evaluated
    live, so install / removal / rotation / expiry are all honoured without an explicit
    reset (this just drops the memoized signature check).
    """
    global _CACHE
    _CACHE = None


def current_edition() -> Edition:
    """The active edition (community unless a valid, unexpired EE license is present)."""
    return _resolve().edition


def current_entitlements() -> frozenset[str]:
    """The active entitlement feature set (empty on community)."""
    return _resolve().entitlements


def current_license() -> License | None:
    """The resolved license payload, if any was parsed (even an expired one)."""
    return _resolve().license


def resolution_reason() -> str:
    """A short machine reason for the current posture (``ok``/``expired``/… ; for `verify`)."""
    return _resolve().reason


def has_entitlement(feature: str) -> bool:
    """True when the active license grants ``feature``."""
    return feature in _resolve().entitlements


def require_entitlement(feature: str) -> None:
    """Gate an Enterprise surface; raise :class:`EnterpriseFeatureError` on community.

    Call this at the entry of a gated code path. On an entitled enterprise install it
    returns quietly; otherwise it raises with an actionable install message.
    """
    if feature not in _resolve().entitlements:
        raise EnterpriseFeatureError(feature)


__all__ = [
    "Edition",
    "EnterpriseFeatureError",
    "FEATURE_AIRGAP_BUNDLE",
    "FEATURE_AUDIT_EXPORT",
    "FEATURE_CUSTOM_RBAC_POLICY",
    "FEATURE_ENTERPRISE_CONSOLE",
    "FEATURE_OIDC_SSO",
    "FEATURE_REGISTRY",
    "HIMMY_LICENSE_KEY_ENV",
    "LICENSE_PUBLIC_KEY_HEX",
    "LICENSE_SECRET_NAME",
    "License",
    "current_edition",
    "current_entitlements",
    "current_license",
    "has_entitlement",
    "license_file_path",
    "require_entitlement",
    "reset_license_cache",
    "resolution_reason",
    "verify_license",
    "_b64url_encode",
]
