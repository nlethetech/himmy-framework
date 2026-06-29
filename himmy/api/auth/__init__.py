"""API authentication & authorization: who is calling, and what may they touch.

The seam mirrors the inference ``ClientManager`` pattern: an
:class:`~himmy.api.auth.base.Authenticator` resolves a request into a verified
:class:`~himmy.api.auth.principal.Principal`, and the BFF picks an implementation
(API key today; OIDC/mTLS plug in) from config. Routers then derive the tenant
from the principal via :func:`~himmy.api.auth.context.resolve_workspace`, so
tenant isolation is enforced from a trusted source rather than client input.
"""

from __future__ import annotations

from himmy.api.auth.base import Authenticator, AuthError
from himmy.api.auth.context import (
    authorize_object,
    authorize_studio_object,
    build_authenticator,
    enforce_subject_write,
    get_principal,
    is_multi_tenant,
    narrow_subject,
    principal_dependency,
    require_workspace,
    resolve_workspace,
    studio_tenant_filter,
)
from himmy.api.auth.principal import ANONYMOUS, TENANT_ADMIN_ROLE, Principal
from himmy.api.auth.rbac import (
    DEFAULT_POLICY,
    AccessPolicy,
    build_access_policy,
    policy_fingerprint,
    policy_source,
    require_permission,
)
from himmy.api.auth.scope_marker import scoped_read
from himmy.api.auth.service_principal import (
    connector_service_principal,
    routine_service_principal,
)

__all__ = [
    "Authenticator",
    "AuthError",
    "Principal",
    "ANONYMOUS",
    "TENANT_ADMIN_ROLE",
    "build_authenticator",
    "is_multi_tenant",
    "principal_dependency",
    "get_principal",
    "resolve_workspace",
    "require_workspace",
    "authorize_object",
    "narrow_subject",
    "enforce_subject_write",
    "studio_tenant_filter",
    "authorize_studio_object",
    "scoped_read",
    "AccessPolicy",
    "DEFAULT_POLICY",
    "build_access_policy",
    "policy_fingerprint",
    "policy_source",
    "require_permission",
    "connector_service_principal",
    "routine_service_principal",
]
