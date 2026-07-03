"""Himmy Enterprise Edition licensing (offline, fail-closed).

See :mod:`himmy.licensing.core` for the wire format and verification model. This package
is the single import point for the monetization seam::

    from himmy.licensing import require_entitlement, FEATURE_OIDC_SSO

    require_entitlement(FEATURE_OIDC_SSO)  # raises EnterpriseFeatureError on community
"""

from __future__ import annotations

from himmy.licensing.core import (
    FEATURE_AIRGAP_BUNDLE,
    FEATURE_AUDIT_EXPORT,
    FEATURE_CUSTOM_RBAC_POLICY,
    FEATURE_ENTERPRISE_CONSOLE,
    FEATURE_OIDC_SSO,
    FEATURE_REGISTRY,
    HIMMY_LICENSE_KEY_ENV,
    LICENSE_PUBLIC_KEY_HEX,
    LICENSE_SECRET_NAME,
    Edition,
    EnterpriseFeatureError,
    License,
    current_edition,
    current_entitlements,
    current_license,
    has_entitlement,
    license_file_path,
    require_entitlement,
    reset_license_cache,
    resolution_reason,
    verify_license,
)

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
]
