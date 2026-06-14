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
    build_authenticator,
    get_principal,
    is_multi_tenant,
    principal_dependency,
    require_workspace,
    resolve_workspace,
)
from himmy.api.auth.principal import ANONYMOUS, Principal
from himmy.api.auth.rbac import (
    DEFAULT_POLICY,
    AccessPolicy,
    build_access_policy,
    require_permission,
)

__all__ = [
    "Authenticator",
    "AuthError",
    "Principal",
    "ANONYMOUS",
    "build_authenticator",
    "is_multi_tenant",
    "principal_dependency",
    "get_principal",
    "resolve_workspace",
    "require_workspace",
    "AccessPolicy",
    "DEFAULT_POLICY",
    "build_access_policy",
    "require_permission",
]
