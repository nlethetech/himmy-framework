"""Supply-chain guard (WS5.5): the OIDC publish job must SHA-pin its actions.

`release.yml`'s ``publish`` job carries ``id-token: write`` — the highest
blast-radius point in the repo, since a retagged third-party action could
exfiltrate the short-lived OIDC token that authenticates PyPI Trusted
Publishing. Mirroring ``security.yml``'s trivy pin, the security-critical
actions on that job must be pinned to an immutable 40-char commit SHA (a
moving branch/tag ref like ``@release/v1`` or ``@v6`` is forbidden), with a
trailing ``# vX.Y.Z`` comment so Dependabot can still bump them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

# tests/test_release_workflow_pinning.py -> repo root is two parents up.
_RELEASE_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "release.yml"

_SHA = re.compile(r"^[0-9a-f]{40}$")

# The actions the actions-sha-pin fix is responsible for pinning on the publish
# job: the OIDC publisher itself plus the checkout/setup-python steps that run
# in the same id-token:write context.
_PINNED_ACTIONS = {
    "actions/checkout",
    "actions/setup-python",
    "pypa/gh-action-pypi-publish",
}


def _publish_job() -> dict[str, Any]:
    doc = yaml.safe_load(_RELEASE_WORKFLOW.read_text())
    job: dict[str, Any] = doc["jobs"]["publish"]
    # Guard the premise of this test: this really is the OIDC job.
    assert (job.get("permissions") or {}).get("id-token") == "write"
    return job


def test_oidc_publish_job_pins_security_critical_actions_by_sha() -> None:
    """Every security-critical `uses:` on the publish job is a 40-char SHA."""
    seen: set[str] = set()
    for step in _publish_job().get("steps") or []:
        uses = step.get("uses")
        if not uses or "@" not in uses:
            continue
        name, ref = uses.split("@", 1)
        if name not in _PINNED_ACTIONS:
            continue
        seen.add(name)
        assert _SHA.match(ref), (
            f"{name} must be pinned to a 40-char commit SHA on the id-token:write "
            f"publish job (a moving ref like {ref!r} can be retagged to exfiltrate "
            "the OIDC token); use 'gh api repos/<owner>/<repo>/git/refs/tags/<tag>'."
        )

    missing = _PINNED_ACTIONS - seen
    assert not missing, f"expected the publish job to use these actions: {sorted(missing)}"


def test_pinned_publish_actions_carry_a_version_comment() -> None:
    """Pinned actions keep a trailing `# vX.Y.Z` comment for Dependabot bumps."""
    # Parsed YAML drops comments, so read the raw lines around the pinned `uses:`.
    pinned_names = {n.split("/", 1)[1] for n in _PINNED_ACTIONS}
    checked = 0
    for line in _RELEASE_WORKFLOW.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("- uses:") and "uses:" not in stripped:
            continue
        if "@" not in stripped:
            continue
        name = stripped.split("uses:", 1)[1].split("@", 1)[0].strip()
        ref_part = stripped.split("@", 1)[1]
        if not any(name.endswith(short) for short in pinned_names):
            continue
        if not _SHA.match(ref_part.split("#", 1)[0].split()[0]):
            continue  # tag-pinned (e.g. setup-node) — not in scope here
        assert "#" in ref_part, f"SHA pin for {name} is missing its `# vX.Y.Z` comment: {stripped!r}"
        checked += 1
    assert checked, "no SHA-pinned actions found to validate comments on"
