"""``himmy license`` — inspect, install, and verify the Enterprise Edition license.

No subcommand shows the active edition, entitlements, customer, and expiry. ``install``
persists a license (a raw token or a path to a token file) to ``~/.himmy/license`` so it
loads on the next run. ``verify`` re-resolves and reports whether the current license is
valid, expired, or absent — always OFFLINE, no network.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any


def _fmt_expiry(expires_at: int) -> str:
    """Render an ``expires_at`` epoch as a readable date, or 'never'."""
    if not expires_at:
        return "never"
    stamp = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(expires_at))
    remaining = expires_at - int(time.time())
    if remaining <= 0:
        return f"{stamp} (EXPIRED)"
    return f"{stamp} ({remaining // 86400}d left)"


def cmd_license(args: argparse.Namespace) -> int:
    """Dispatch the ``license`` subcommands (default: show)."""
    action = getattr(args, "action", None)
    if action == "install":
        return _install(args.value)
    if action == "verify":
        return _verify()
    return _show()


def _show() -> int:
    """Print the current edition, entitlements, customer, and expiry."""
    from himmy.licensing import (
        current_edition,
        current_entitlements,
        current_license,
        reset_license_cache,
    )

    reset_license_cache()
    edition = current_edition()
    lic = current_license()
    print(f"edition:      {edition.value}")
    if lic is not None:
        print(f"license_id:   {lic.license_id}")
        print(f"customer:     {lic.customer or '(unspecified)'}")
        print(f"expires:      {_fmt_expiry(lic.expires_at)}")
    ents = sorted(current_entitlements())
    print(f"entitlements: {', '.join(ents) if ents else '(none — community)'}")
    if edition.value == "community":
        print(
            "\nCommunity Edition. All core / offline / single-tenant features are free.\n"
            "Enterprise features (SSO, signed audit export, console) need a license:\n"
            "  himmy license install <key-or-file>"
        )
    return 0


def _install(value: str) -> int:
    """Persist ``value`` (a token or a path to a token file) to ``~/.himmy/license``."""
    from himmy.licensing import reset_license_cache, verify_license

    candidate = Path(value).expanduser()
    if candidate.is_file():
        token = candidate.read_text(encoding="utf-8").strip()
    else:
        token = value.strip()

    if not verify_license(token):
        print(
            "error: that license is invalid or tampered (signature did not verify). "
            "Nothing was written."
        )
        return 1

    dest = Path.home() / ".himmy" / "license"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(token + "\n", encoding="utf-8")
    dest.chmod(0o600)
    reset_license_cache()
    print(f"license installed to {dest}")
    return _verify()


def _verify() -> int:
    """Re-resolve and report the current license posture; exit non-zero if not enterprise."""
    from himmy.licensing import (
        Edition,
        current_edition,
        reset_license_cache,
        resolution_reason,
    )

    reset_license_cache()
    edition = current_edition()
    reason = resolution_reason()
    if edition is Edition.ENTERPRISE:
        print("license OK: Enterprise Edition active.")
        return 0
    messages = {
        "no-license": "no license found (Community Edition).",
        "invalid-signature": "license signature is invalid or tampered (Community).",
        "expired": "license has EXPIRED (Community).",
        "not-enterprise": "license is not an Enterprise edition (Community).",
    }
    print(f"license: {messages.get(reason, 'Community Edition.')}")
    return 1


def add_license_parser(sub: Any) -> None:
    """Register the ``license`` subcommand tree on the CLI's subparsers."""
    p = sub.add_parser(
        "license", help="show / install / verify the Enterprise Edition license"
    )
    lsub = p.add_subparsers(dest="action")
    p_install = lsub.add_parser("install", help="install a license (token or file path)")
    p_install.add_argument("value", help="the license token or a path to a license file")
    p_install.set_defaults(func=cmd_license)
    p_verify = lsub.add_parser("verify", help="verify the currently-installed license")
    p_verify.set_defaults(func=cmd_license)
    p.set_defaults(func=cmd_license, action=None)


__all__ = ["add_license_parser", "cmd_license"]
