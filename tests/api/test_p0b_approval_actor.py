"""P0-B: the Studio approve/reject endpoints resolve the resolving actor.

Anonymous (the zero-config loopback default) maps to the meaningful "human"; an
authenticated principal stamps its real subject onto APPROVAL_* events.
"""

from __future__ import annotations

import types

from himmy.api.auth.principal import ANONYMOUS, Principal
from himmy.api.routers.studio import _approval_actor


def _request_with(principal: Principal) -> object:
    return types.SimpleNamespace(state=types.SimpleNamespace(principal=principal))


def test_anonymous_actor_is_human() -> None:
    assert _approval_actor(_request_with(ANONYMOUS)) == "human"


def test_authenticated_subject_is_stamped() -> None:
    principal = Principal.build(
        "user-a", tenant_ids=["t"], roles=["operator"], auth_method="apikey"
    )
    assert _approval_actor(_request_with(principal)) == "user-a"
