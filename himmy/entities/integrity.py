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
* :func:`export_audit_chain` / :func:`verify_audit_chain` — an OPT-IN hash-chain
  mode for ordered audit trails. The Merkle root above is order-INdependent (leaves
  are sorted), so record *reordering* is invisible to it; the chain binds each
  record to its predecessor via ``prev_hash`` and signs the chain tip, so
  verification walks the chain and pinpoints exactly where it first breaks
  (tamper, reorder, deletion, or insertion). The Merkle bundle stays the default
  for unordered graph snapshots.

Everything is stdlib (``hashlib``/``hmac``/``json``) and fully offline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

from pydantic import BaseModel, Field

from himmy.core.errors import HimmyError
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


# ------------------------------------------------------------ hash chain (opt-in)
# The Merkle bundle above is deliberately order-independent (sorted leaves), which is
# right for unordered graph snapshots but means a reordered audit TRAIL still produces
# the same root. The chain mode below is the opt-in alternative for append-only,
# ordered histories: each entry carries the hash of its predecessor, the export signs
# only the chain TIP, and verification walks the chain so the FIRST divergent position
# is pinpointed rather than just "something changed".

#: The ``prev_hash`` of the first chain entry (a fixed, content-free genesis value).
CHAIN_GENESIS_HASH = hashlib.sha256(b"himmy-audit-chain:genesis").hexdigest()


def _chain_step(prev_hash: str, leaf: str) -> str:
    """Advance the chain: hash the predecessor's chain hash with the next leaf."""
    return hashlib.sha256(f"{prev_hash}:{leaf}".encode()).hexdigest()


class ChainEntry(BaseModel):
    """One link of an audit hash chain.

    ``content_hash`` fingerprints the record itself (:func:`content_hash`);
    ``prev_hash`` is the predecessor entry's ``chain_hash`` (``CHAIN_GENESIS_HASH``
    for the first entry); ``chain_hash`` commits to the entire prefix up to and
    including this record, so any earlier edit/reorder/deletion changes it.
    """

    record_id: str
    content_hash: str
    prev_hash: str
    chain_hash: str


class AuditChainBundle(BaseModel):
    """A signed, ORDER-SENSITIVE snapshot of an append-only record sequence.

    ``entries`` mirror the records in their stored order; ``tip_hash`` is the last
    entry's ``chain_hash`` (the genesis hash for an empty sequence) and commits to
    the full ordered history; ``signature`` is an HMAC-SHA256 over the tip with the
    caller's secret. Unlike :class:`AuditBundle`, reordering records changes the tip.
    """

    bundle_version: int = AUDIT_BUNDLE_VERSION
    entries: list[ChainEntry] = Field(default_factory=list)
    tip_hash: str = CHAIN_GENESIS_HASH
    signature: str = ""
    algorithm: str = "HMAC-SHA256"


class ChainVerificationResult(BaseModel):
    """The outcome of verifying a live record sequence against a chain bundle.

    ``chain_intact`` is True when the chain tip recomputed from the live sequence
    equals the signed tip (i.e. the ordered history is byte-for-byte what was
    signed). ``first_break_index`` is the earliest position where the live sequence
    diverges from the bundle, and the ``*_record_ids`` lists classify every
    divergent id exactly once: content edited in place (``tampered``), present on
    both sides but at a different position (``reordered``), in the bundle but gone
    from the live sequence (``missing``), or live but never signed (``added``).
    """

    ok: bool
    signature_valid: bool
    chain_intact: bool
    first_break_index: int | None = None
    tampered_record_ids: list[str] = Field(default_factory=list)
    reordered_record_ids: list[str] = Field(default_factory=list)
    missing_record_ids: list[str] = Field(default_factory=list)
    added_record_ids: list[str] = Field(default_factory=list)


def export_audit_chain(
    records: list[EntityRecord],
    *,
    secret: str | bytes,
) -> AuditChainBundle:
    """Freeze + sign an ORDERED record sequence into an :class:`AuditChainBundle`.

    ``records`` must be in their canonical stored order (e.g. registration order /
    ascending version order) — the chain commits to that order, which is the whole
    point: pass the same order at verification time.
    """
    entries: list[ChainEntry] = []
    prev = CHAIN_GENESIS_HASH
    for record in records:
        leaf = content_hash(record)
        chained = _chain_step(prev, leaf)
        entries.append(
            ChainEntry(
                record_id=record.record_id,
                content_hash=leaf,
                prev_hash=prev,
                chain_hash=chained,
            )
        )
        prev = chained
    return AuditChainBundle(
        entries=entries,
        tip_hash=prev,
        signature=_sign(prev, secret),
    )


def verify_audit_chain(
    bundle: AuditChainBundle,
    records: list[EntityRecord],
    *,
    secret: str | bytes,
) -> ChainVerificationResult:
    """Walk a live record sequence against a signed chain, pinpointing any break.

    Checks, in order: (1) the bundle's signature is intact for its chain tip
    (catches a forged/edited bundle); (2) the chain tip recomputed from the live
    sequence matches the signed tip — because each link binds its predecessor, ANY
    in-place edit, reorder, mid-history deletion, or insertion changes the live
    tip; (3) a positional walk classifies each divergent id (tampered / reordered /
    missing / added) and reports the first break index. ``ok`` is True only when
    the signature is valid AND the live tip equals the signed tip.
    """
    signature_valid = hmac.compare_digest(
        bundle.signature, _sign(bundle.tip_hash, secret)
    )

    live = [(r.record_id, content_hash(r)) for r in records]
    live_tip = CHAIN_GENESIS_HASH
    for _, leaf in live:
        live_tip = _chain_step(live_tip, leaf)
    chain_intact = live_tip == bundle.tip_hash

    bundle_ids = {entry.record_id for entry in bundle.entries}
    live_ids = {rid for rid, _ in live}
    tampered: list[str] = []
    reordered: list[str] = []
    missing: list[str] = []
    added: list[str] = []
    flagged: set[str] = set()
    first_break: int | None = None

    def _flag(bucket: list[str], rid: str) -> None:
        if rid not in flagged:
            flagged.add(rid)
            bucket.append(rid)

    for i in range(max(len(bundle.entries), len(live))):
        entry = bundle.entries[i] if i < len(bundle.entries) else None
        pair = live[i] if i < len(live) else None
        diverged = False
        if entry is not None and pair is not None:
            rid, leaf = pair
            if rid == entry.record_id:
                if leaf != entry.content_hash:
                    _flag(tampered, rid)
                    diverged = True
            else:
                diverged = True
                _flag(reordered if rid in bundle_ids else added, rid)
                _flag(
                    reordered if entry.record_id in live_ids else missing,
                    entry.record_id,
                )
        elif pair is not None:  # live sequence is longer than the signed one
            rid = pair[0]
            diverged = True
            _flag(reordered if rid in bundle_ids else added, rid)
        elif entry is not None:  # signed sequence is longer than the live one
            diverged = True
            _flag(
                reordered if entry.record_id in live_ids else missing,
                entry.record_id,
            )
        if diverged and first_break is None:
            first_break = i

    return ChainVerificationResult(
        ok=signature_valid and chain_intact,
        signature_valid=signature_valid,
        chain_intact=chain_intact,
        first_break_index=first_break,
        tampered_record_ids=tampered,
        reordered_record_ids=reordered,
        missing_record_ids=missing,
        added_record_ids=added,
    )


# --------------------------------------------------------- Ed25519 (asymmetric)
# An asymmetric option (WS4.5): the auditor verifies with only the PUBLIC key, so the
# signing secret never has to be shared with verifiers — stronger non-repudiation than
# the shared-secret HMAC. Keys are PEM strings (the private key ideally lives in an HSM
# / KMS); ``cryptography`` is required.


def generate_ed25519_keypair() -> tuple[str, str]:
    """Generate an Ed25519 keypair; return ``(private_pem, public_pem)``."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    private = Ed25519PrivateKey.generate()
    private_pem = private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("ascii")
    public_pem = (
        private.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode("ascii")
    )
    return private_pem, public_pem


def _sign_ed25519(merkle_root: str, private_pem: str) -> str:
    """Ed25519-sign the Merkle root; return a hex signature."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = serialization.load_pem_private_key(private_pem.encode("ascii"), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise HimmyError("audit signing key must be an Ed25519 private key")
    return key.sign(merkle_root.encode("utf-8")).hex()


def _verify_ed25519(merkle_root: str, signature_hex: str, public_pem: str) -> bool:
    """Verify an Ed25519 signature over the Merkle root with the public key."""
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    key = serialization.load_pem_public_key(public_pem.encode("ascii"))
    if not isinstance(key, Ed25519PublicKey):
        return False
    try:
        key.verify(bytes.fromhex(signature_hex), merkle_root.encode("utf-8"))
        return True
    except (InvalidSignature, ValueError):
        return False


def export_audit_bundle_ed25519(
    records: list[EntityRecord],
    links: list[EntityLink],
    *,
    private_pem: str,
) -> AuditBundle:
    """Freeze + Ed25519-sign a graph's integrity (verifiable with the public key)."""
    record_hashes = {r.record_id: content_hash(r) for r in records}
    link_hashes = {link.link_id: link_hash(link) for link in links}
    root = _merkle_root(list(record_hashes.values()) + list(link_hashes.values()))
    return AuditBundle(
        records=record_hashes,
        links=link_hashes,
        merkle_root=root,
        signature=_sign_ed25519(root, private_pem),
        algorithm="Ed25519",
    )


def verify_audit_bundle_ed25519(
    bundle: AuditBundle,
    records: list[EntityRecord],
    links: list[EntityLink],
    *,
    public_pem: str,
) -> VerificationResult:
    """Verify an Ed25519-signed bundle against a live graph (public-key only)."""
    base = verify_audit_bundle(bundle, records, links, secret=b"")
    signature_valid = _verify_ed25519(bundle.merkle_root, bundle.signature, public_pem)
    return base.model_copy(
        update={
            "signature_valid": signature_valid,
            "ok": signature_valid
            and not any(
                [
                    base.tampered_record_ids,
                    base.missing_record_ids,
                    base.added_record_ids,
                    base.tampered_link_ids,
                    base.missing_link_ids,
                    base.added_link_ids,
                ]
            ),
        }
    )


__all__ = [
    "AUDIT_BUNDLE_VERSION",
    "CHAIN_GENESIS_HASH",
    "AuditBundle",
    "AuditChainBundle",
    "ChainEntry",
    "ChainVerificationResult",
    "VerificationResult",
    "content_hash",
    "link_hash",
    "export_audit_bundle",
    "verify_audit_bundle",
    "export_audit_chain",
    "verify_audit_chain",
    "generate_ed25519_keypair",
    "export_audit_bundle_ed25519",
    "verify_audit_bundle_ed25519",
]
