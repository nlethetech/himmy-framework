"""API-suite fixtures.

The notification sink persists every ``record_notification`` call to the
project's ``.himmy/notify.db`` (notifications survive restarts). Any API test
that triggers a notification — a project created, an approval pause, … —
would otherwise leak test rows into the developer's real Studio inbox. Keep
the whole suite pointed at a per-test temp store; tests that probe the store
location themselves simply override the same env var.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_notify_store(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
):
    from himmy.api.routers import studio_notify

    db_dir: Path = tmp_path_factory.mktemp("notify")
    monkeypatch.setenv("HIMMY_NOTIFY_PATH", str(db_dir / "notify.db"))
    studio_notify.reset_notify_state()
    yield
    studio_notify.reset_notify_state()
    # Leave the ring marked as seeded so a later non-API test can't rehydrate
    # its in-memory deque from the repo's real on-disk store mid-suite.
    studio_notify._HYDRATED = True
