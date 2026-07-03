#!/usr/bin/env python3
"""Vendor admin tool: mint a signed Himmy Enterprise Edition license.

This is a VENDOR-ONLY tool. It signs a license payload with the vendor's Ed25519 PRIVATE
key, which lives OUTSIDE the repo (default ``~/.himmy-admin/license_ed25519``, a 32-byte
raw private key, ``chmod 600``). The PUBLIC half is baked into
``himmy.licensing.core.LICENSE_PUBLIC_KEY_HEX`` and is the only key ever committed.

The private key must NEVER be committed. In production the vendor keeps it in an offline
HSM / secrets vault; this script only reads a local path so a dev keypair can drive tests.

Bootstrap a dev keypair (one time)::

    python scripts/mint_license.py --gen-key
    # prints the PUBLIC key hex to paste into himmy/licensing/core.py, writes the private
    # key to ~/.himmy-admin/license_ed25519 (never checked in)

Mint a license::

    python scripts/mint_license.py \
        --customer "Acme Corp" \
        --features oidc_sso,audit_export,enterprise_console \
        --days 365 > acme.license

Install it on the customer side::

    himmy license install acme.license
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

DEFAULT_KEY_PATH = os.path.expanduser("~/.himmy-admin/license_ed25519")
ALL_FEATURES = [
    "oidc_sso",
    "audit_export",
    "custom_rbac_policy",
    "airgap_bundle",
    "enterprise_console",
]


def _b64url(raw: bytes) -> str:
    """URL-safe base64 without padding (matches ``himmy.licensing.core``)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _gen_key(path: str) -> int:
    """Generate a fresh Ed25519 keypair; write the private key, print the public hex."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    if os.path.exists(path):
        print(f"refusing to overwrite existing key at {path}", file=sys.stderr)
        return 1
    priv = Ed25519PrivateKey.generate()
    raw_priv = priv.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    raw_pub = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(raw_priv)
    os.chmod(path, 0o600)
    print(f"private key written to {path} (chmod 600 — NEVER commit)", file=sys.stderr)
    print(f"public key (bake into himmy/licensing/core.py):\n{raw_pub.hex()}")
    return 0


def _mint(args: argparse.Namespace) -> int:
    """Sign a license payload with the private key at ``--key`` and print the token."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    try:
        with open(args.key, "rb") as handle:
            priv = Ed25519PrivateKey.from_private_bytes(handle.read())
    except (OSError, ValueError) as exc:
        print(f"cannot load signing key {args.key}: {exc}", file=sys.stderr)
        return 1

    features = [f.strip() for f in args.features.split(",") if f.strip()]
    now = int(time.time())
    payload = {
        "license_id": args.license_id or f"lic-{now}",
        "edition": "enterprise",
        "features": features,
        "customer": args.customer,
        "issued_at": now,
        "expires_at": 0 if args.days <= 0 else now + args.days * 86400,
    }
    payload_raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    signature = priv.sign(payload_raw)
    token = f"{_b64url(payload_raw)}.{_b64url(signature)}"
    print(token)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry: ``--gen-key`` bootstraps a keypair; otherwise mint a license."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gen-key", action="store_true", help="generate a fresh keypair and exit"
    )
    parser.add_argument("--key", default=DEFAULT_KEY_PATH, help="private signing key path")
    parser.add_argument("--customer", default="", help="customer / org name")
    parser.add_argument(
        "--features",
        default=",".join(ALL_FEATURES),
        help="comma-separated entitlements (default: all)",
    )
    parser.add_argument("--license-id", default="", help="explicit license id")
    parser.add_argument(
        "--days", type=int, default=365, help="validity in days (<=0 = never expires)"
    )
    args = parser.parse_args(argv)

    if args.gen_key:
        return _gen_key(args.key)
    return _mint(args)


if __name__ == "__main__":
    raise SystemExit(main())
