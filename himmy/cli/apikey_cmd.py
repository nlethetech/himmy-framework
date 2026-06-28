"""``himmy apikey`` — mint, rotate, revoke, and list lifecycle-managed API keys.

Operates on the same two artifacts the server reads at startup
(:func:`himmy.api.auth.context.build_authenticator`):

* the **keys file** (``HIMMY_API_KEYS_FILE``) — a JSON map of
  ``{secret: {subject, tenant_ids, roles, expires_at, disabled}}`` whose plaintext
  secrets the server hashes in-memory at load. ``mint`` adds an entry and prints the
  generated secret ONCE (it is never recoverable afterwards — only its hash is kept by
  the server); ``rotate`` mints a replacement and disables the old entry; ``list``
  shows non-secret metadata only.
* the **revocation file** (``HIMMY_API_KEY_REVOCATION_FILE``) — a JSON list of revoked
  ``key_id`` fingerprints the server re-reads live, so ``revoke`` kills a key WITHOUT
  a restart.

SECURITY: write the keys file as ``chmod 0600`` (mint does so for files it creates)
and prefer sourcing it from a secrets manager in production; the plaintext secret only
needs to survive long enough to hand to the caller — the server stores a salted hash.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from himmy.api.auth.apikey import REVOCATION_FILE_ENV, _fingerprint
from himmy.cli.ui import styles

#: Default keys-file path when neither ``--file`` nor ``HIMMY_API_KEYS_FILE`` is set.
_DEFAULT_KEYS_FILE = ".himmy/api_keys.json"


def _keys_path(args: argparse.Namespace) -> Path:
    """Resolve the keys file: ``--file`` ▸ ``HIMMY_API_KEYS_FILE`` ▸ default."""
    raw = (
        getattr(args, "file", None)
        or os.environ.get("HIMMY_API_KEYS_FILE")
        or _DEFAULT_KEYS_FILE
    )
    return Path(raw).expanduser()


def _revocation_path(args: argparse.Namespace) -> Path:
    """Resolve the revocation file: ``--revocation-file`` ▸ env ▸ keys-file sibling."""
    raw = getattr(args, "revocation_file", None) or os.environ.get(REVOCATION_FILE_ENV)
    if raw:
        return Path(raw).expanduser()
    return _keys_path(args).with_name("api_keys.revoked.json")


def _load_keys(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"keys file {path} must be a JSON object")
    return raw


def _write_json_0600(path: Path, data: Any) -> None:
    """Write JSON atomically with ``0600`` perms (owner read/write only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def _err(msg: str) -> int:
    s = styles(sys.stdout)
    print(f"{s['crimson']}{msg}{s['reset']}", file=sys.stderr)
    return 2


def cmd_apikey_mint(args: argparse.Namespace) -> int:
    """Generate a new key, append it to the keys file, and print the secret ONCE."""
    s = styles(sys.stdout)
    path = _keys_path(args)
    keys = _load_keys(path)

    secret = f"himmy_{secrets.token_urlsafe(32)}"
    expires_at: str | None = None
    if getattr(args, "ttl_days", None):
        expires_at = (
            datetime.now(UTC) + timedelta(days=args.ttl_days)
        ).isoformat()

    entry: dict[str, Any] = {
        "subject": args.subject or f"apikey:{_fingerprint(secret)}",
        "tenant_ids": list(args.tenant or []),
        "roles": list(args.role or []),
        "all_tenants": bool(args.all_tenants),
        "disabled": False,
    }
    if expires_at:
        entry["expires_at"] = expires_at
    keys[secret] = entry
    _write_json_0600(path, keys)

    key_id = _fingerprint(secret)
    print(f"{s['gold']}minted api key{s['reset']}  key_id={key_id}")
    print(f"  subject    : {entry['subject']}")
    print(f"  tenant_ids : {entry['tenant_ids'] or '(none)'}")
    print(f"  roles      : {entry['roles'] or '(none)'}")
    if expires_at:
        print(f"  expires_at : {expires_at}")
    print(f"  file       : {path}")
    print()
    print(f"{s['gold']}SECRET (shown once — store it now):{s['reset']}")
    print(f"  {secret}")
    return 0


def cmd_apikey_rotate(args: argparse.Namespace) -> int:
    """Mint a replacement for an existing key (by key_id) and disable the old one."""
    s = styles(sys.stdout)
    path = _keys_path(args)
    keys = _load_keys(path)

    target = None
    for secret, spec in keys.items():
        if _fingerprint(secret) == args.key_id:
            target = (secret, spec or {})
            break
    if target is None:
        return _err(f"no key with key_id={args.key_id} in {path}")

    old_secret, old_spec = target
    new_secret = f"himmy_{secrets.token_urlsafe(32)}"
    new_spec = dict(old_spec)
    new_spec["disabled"] = False
    keys[new_secret] = new_spec
    # Disable the old entry in place so an unrotated client fails closed (it is NOT
    # deleted, so audit/`list` still shows the retired key_id).
    old = dict(old_spec)
    old["disabled"] = True
    keys[old_secret] = old
    _write_json_0600(path, keys)

    print(
        f"{s['gold']}rotated{s['reset']}  old key_id={args.key_id} (disabled) "
        f"→ new key_id={_fingerprint(new_secret)}"
    )
    print()
    print(f"{s['gold']}NEW SECRET (shown once — store it now):{s['reset']}")
    print(f"  {new_secret}")
    return 0


def cmd_apikey_revoke(args: argparse.Namespace) -> int:
    """Add a key_id to the live revocation file — kills the key without a restart."""
    s = styles(sys.stdout)
    path = _revocation_path(args)
    revoked: list[str] = []
    if path.is_file():
        try:
            raw = json.loads(path.read_text())
            if isinstance(raw, dict):
                raw = raw.get("revoked", [])
            if isinstance(raw, list):
                revoked = [str(x) for x in raw]
        except (OSError, json.JSONDecodeError):
            revoked = []
    if args.key_id in revoked:
        print(f"key_id={args.key_id} already revoked ({path})")
        return 0
    revoked.append(args.key_id)
    _write_json_0600(path, revoked)
    print(
        f"{s['gold']}revoked{s['reset']}  key_id={args.key_id} → {path} "
        f"(takes effect on the next request; no restart needed)"
    )
    return 0


def cmd_apikey_list(args: argparse.Namespace) -> int:
    """List configured keys as NON-SECRET metadata (key_id + subject/tenants/state)."""
    s = styles(sys.stdout)
    path = _keys_path(args)
    keys = _load_keys(path)
    if not keys:
        print(f"no keys in {path}")
        return 0
    rev_path = _revocation_path(args)
    revoked: set[str] = set()
    if rev_path.is_file():
        try:
            raw = json.loads(rev_path.read_text())
            if isinstance(raw, dict):
                raw = raw.get("revoked", [])
            if isinstance(raw, list):
                revoked = {str(x) for x in raw}
        except (OSError, json.JSONDecodeError):
            revoked = set()

    print(f"{s['dim']}api keys in {path}{s['reset']}")
    for secret, spec in keys.items():
        spec = spec or {}
        key_id = _fingerprint(secret)
        state = []
        if spec.get("disabled"):
            state.append("disabled")
        if key_id in revoked:
            state.append("revoked")
        if spec.get("expires_at"):
            state.append(f"expires={spec['expires_at']}")
        flags = f"  [{', '.join(state)}]" if state else ""
        print(
            f"  {key_id}  subject={spec.get('subject', '?')}  "
            f"tenants={spec.get('tenant_ids') or '-'}  "
            f"roles={spec.get('roles') or '-'}{flags}"
        )
    return 0


def cmd_apikey(args: argparse.Namespace) -> int:
    """``himmy apikey <action>`` dispatcher."""
    action = getattr(args, "action", None)
    dispatch = {
        "mint": cmd_apikey_mint,
        "rotate": cmd_apikey_rotate,
        "revoke": cmd_apikey_revoke,
        "list": cmd_apikey_list,
    }
    fn = dispatch.get(action)
    if fn is None:
        return _err(
            f"unknown apikey action: {action!r} (expected: mint/rotate/revoke/list)"
        )
    return fn(args)


__all__ = [
    "cmd_apikey",
    "cmd_apikey_mint",
    "cmd_apikey_rotate",
    "cmd_apikey_revoke",
    "cmd_apikey_list",
]
