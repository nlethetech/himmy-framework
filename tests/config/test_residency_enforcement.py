"""WS4.3 — residency enforcement is wired at the storage + inference boundaries.

The policy primitives (``enforce_region`` etc.) are covered in ``test_residency.py``;
these tests pin the actual call sites so ``HIMMY_REGION`` stops being a silent no-op.
"""

from __future__ import annotations

from typing import Any

import pytest

from himmy.config.residency import ResidencyError, storage_region
from himmy.services.storage.factory import StoreFactory
from tests.conftest import run_async

# ------------------------------------------------------------------- storage_region


def test_storage_region_defaults_to_home(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.delenv("HIMMY_STORE_REGION", raising=False)
    assert storage_region() == "eu"


def test_storage_region_explicit_override(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.setenv("HIMMY_STORE_REGION", "eu-west")
    assert storage_region() == "eu-west"


# ---------------------------------------------------------- StoreFactory boundary


def test_for_server_refuses_disallowed_store_region(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.setenv("HIMMY_STORE_REGION", "us")  # outside the policy
    monkeypatch.delenv("HIMMY_ALLOWED_REGIONS", raising=False)
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    with pytest.raises(ResidencyError):
        run_async(StoreFactory.for_server())


def test_for_server_allows_in_region_store(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.setenv("HIMMY_STORE_REGION", "eu")
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "storage.db"))
    monkeypatch.delenv("HIMMY_DATABASE_URL", raising=False)
    store = run_async(StoreFactory.for_server())
    assert store is not None  # built, in-region
    run_async(store.close())  # type: ignore[attr-defined]


def test_for_context_server_enforces_region(monkeypatch: Any) -> None:
    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.setenv("HIMMY_STORE_REGION", "us")
    monkeypatch.delenv("HIMMY_ALLOWED_REGIONS", raising=False)
    with pytest.raises(ResidencyError):
        StoreFactory.for_context(server=True)


def test_for_context_not_pinned_is_noop(monkeypatch: Any, tmp_path: Any) -> None:
    monkeypatch.delenv("HIMMY_REGION", raising=False)
    monkeypatch.setenv("HIMMY_STORE_PATH", str(tmp_path / "storage.db"))
    store = StoreFactory.for_context(server=True)
    assert store is not None  # no pinning ⇒ builds normally


# ----------------------------------------------------- inference gateway boundary


def test_inference_gateway_refuses_disallowed_region(monkeypatch: Any) -> None:
    from himmy.api.deps import ApiContainer

    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.setenv("HIMMY_ALLOWED_REGIONS", "eu")
    monkeypatch.setenv("HIMMY_GATEWAY_REGION", "us")  # gateway in a disallowed region
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "k")
    with pytest.raises(ResidencyError):
        ApiContainer._build_inference()


def test_inference_gateway_allows_in_region(monkeypatch: Any) -> None:
    from himmy.api.deps import ApiContainer

    monkeypatch.setenv("HIMMY_REGION", "eu")
    monkeypatch.setenv("HIMMY_ALLOWED_REGIONS", "eu")
    monkeypatch.setenv("HIMMY_GATEWAY_REGION", "eu")
    monkeypatch.setenv("PYDANTIC_AI_GATEWAY_API_KEY", "k")
    svc = ApiContainer._build_inference()
    assert svc is not None
