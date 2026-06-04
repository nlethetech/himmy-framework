"""Entities kernel: tamper-evident integrity over the lineage graph.

The registry captures a rich provenance graph, but ``record_id`` is derived only
from ``(kind, stable_id, version)`` — the payload is NOT in the identity, so a row
edited in place in a backend store is, on its own, undetectable. This module adds
the missing integrity layer WITHOUT changing the record identity:

* :func:`content_hash` — a deterministic SHA-256 fingerprint of a record's full
  content (kind/stable_id/version/payload/metadata). Recompute it and compare to
  catch any in-place mutation.
* :func:`export_audit_bundle` — freeze the graph into a signed manifest: per-record
  and per-link content hashes, a Merkle root over them, and an HMAC signature with a
  caller-held secret. This is the trusted "as-of" snapshot.
* :func:`verify_audit_bundle` — re-derive hashes from a (possibly tampered) graph
  and check them against a bundle, reporting exactly which records/links were
  altered, added, or dropped, and whether the signature itself is intact.

Everything is stdlib (``hashlib``/``hmac``/``json``) and fully offline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from pydantic import BaseModel, Field

from himmy.entities.records import EntityLink, EntityRecord

#: Bumped if the canonical-serialization or Merkle scheme ever changes.
AUDIT_BUNDLE_VERSION = 1


def _canonical(value: Any) -> bytes:
    """Deterministically serialize a value (sorted keys, no whitespace)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def content_hash(record: EntityRecord) -> str:
    """Return the SHA-256 content fingerprint of a record's full content.

    Covers identity (kind/stable_id/version) AND content (payload/metadata), so any
    mutation to a stored row changes the hash. ``created_at`` is excluded so the
    fingerprint is stable across re-projections of identical content.
    """
    return hashlib.sha256(
        _canonical(
            {
                "kind": record.kind,
                "stable_id": record.stable_id,
                "version": record.version,
                "payload": record.payload,
                "metadata": record.metadata,
            }
        )
    ).hexdigest()


def link_hash(link: EntityLink) -> str:
    """Return the SHA-256 fingerprint of a link's endpoints + relation + metadata."""
    return hashlib.sha256(
        _canonical(
            {
                "from": link.from_record_id,
                "to": link.to_record_id,
                "relation": link.relation,
                "metadata": link.metadata,
            }
        )
    ).hexdigest()


def _merkle_root(leaves: list[str]) -> str:
    """A simple binary Merkle root over hex leaf hashes (order-independent: sorted).

    Sorting the leaves makes the root invariant to record/link ordering, so two
    exports of the same graph produce the same root.
    """
    level = sorted(leaves)
    if not level:
        return hashlib.sha256(b"").hexdigest()
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            left = level[i]
            right = level[i + 1] if i + 1 < len(level) else left
            nxt.append(hashlib.sha256((left + right).encode("utf-8")).hexdigest())
        level = nxt
    return level[0]


class AuditBundle(BaseModel):
    """A signed, tamper-evident snapshot of a lineage graph's integrity.

    ``records``/``links`` map id -> content hash; ``merkle_root`` commits to all of
    them; ``signature`` is an HMAC-SHA256 over the root with the caller's secret.
    """

    bundle_version: int = AUDIT_BUNDLE_VERSION
    records: dict[str, str] = Field(default_factory=dict)
    links: dict[str, str] = Field(default_factory=dict)
    merkle_root: str = ""
    signature: str = ""
    algorithm: str = "HMAC-SHA256"


class VerificationResult(BaseModel):
    """The outcome of verifying a live graph against an :class:`AuditBundle`."""

    ok: bool
    signature_valid: bool
    tampered_record_ids: list[str] = Field(default_factory=list)
    missing_record_ids: list[str] = Field(default_factory=list)
    added_record_ids: list[str] = Field(default_factory=list)
    tampered_link_ids: list[str] = Field(default_factory=list)
    missing_link_ids: list[str] = Field(default_factory=list)
    added_link_ids: list[str] = Field(default_factory=list)


def _sign(merkle_root: str, secret: str | bytes) -> str:
    """HMAC-SHA256 the Merkle root with the caller's secret."""
    key = secret.encode("utf-8") if isinstance(secret, str) else secret
    return hmac.new(key, merkle_root.encode("utf-8"), hashlib.sha256).hexdigest()


def export_audit_bundle(
    records: list[EntityRecord],
    links: list[EntityLink],
    *,
    secret: str | bytes,
) -> AuditBundle:
    """Freeze + sign the integrity of a graph into an :class:`AuditBundle`."""
    record_hashes = {r.record_id: content_hash(r) for r in records}
    link_hashes = {link.link_id: link_hash(link) for link in links}
    root = _merkle_root(list(record_hashes.values()) + list(link_hashes.values()))
    return AuditBundle(
        records=record_hashes,
        links=link_hashes,
        merkle_root=root,
        signature=_sign(root, secret),
    )


def verify_audit_bundle(
    bundle: AuditBundle,
    records: list[EntityRecord],
    links: list[EntityLink],
    *,
    secret: str | bytes,
) -> VerificationResult:
    """Verify a live graph against a signed bundle, pinpointing any divergence.

    Checks, in order: (1) the bundle's signature is intact for its Merkle root
    (catches a forged/edited bundle); (2) each record/link hash matches — surfacing
    altered (``tampered``), removed (``missing``), and newly-introduced (``added``)
    ids. ``ok`` is True only when the signature is valid AND nothing diverged.
    """
    signature_valid = hmac.compare_digest(
        bundle.signature, _sign(bundle.merkle_root, secret)
    )

    live_records = {r.record_id: content_hash(r) for r in records}
    live_links = {link.link_id: link_hash(link) for link in links}

    tampered_records = [
        rid
        for rid, h in bundle.records.items()
        if rid in live_records and live_records[rid] != h
    ]
    missing_records = [rid for rid in bundle.records if rid not in live_records]
    added_records = [rid for rid in live_records if rid not in bundle.records]

    tampered_links = [
        lid
        for lid, h in bundle.links.items()
        if lid in live_links and live_links[lid] != h
    ]
    missing_links = [lid for lid in bundle.links if lid not in live_links]
    added_links = [lid for lid in live_links if lid not in bundle.links]

    ok = signature_valid and not any(
        [
            tampered_records,
            missing_records,
            added_records,
            tampered_links,
            missing_links,
            added_links,
        ]
    )
    return VerificationResult(
        ok=ok,
        signature_valid=signature_valid,
        tampered_record_ids=tampered_records,
        missing_record_ids=missing_records,
        added_record_ids=added_records,
        tampered_link_ids=tampered_links,
        missing_link_ids=missing_links,
        added_link_ids=added_links,
    )


__all__ = [
    "AUDIT_BUNDLE_VERSION",
    "AuditBundle",
    "VerificationResult",
    "content_hash",
    "link_hash",
    "export_audit_bundle",
    "verify_audit_bundle",
]
