"""GET /metrics: Prometheus exposition over the FastAPI app.

A live request through the TestClient must (1) produce valid Prometheus text at
``/metrics`` and (2) be reflected in the request counter and the duration
histogram — keyed by the low-cardinality *route template*, never a raw path
param. Cardinality safety (templated label, status class) is asserted explicitly.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from himmy.api.app import create_app
from himmy.services.observability import metrics as metrics_mod


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    for var in (
        "HIMMY_MULTI_TENANT",
        "HIMMY_AUTH_MODE",
        "HIMMY_INTERNAL_API_KEY",
        "HIMMY_DATABASE_URL",
        "HIMMY_DURABLE_STORAGE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    # Fresh registry per test so counts are deterministic.
    metrics_mod.reset_registry()


def test_metrics_endpoint_exposes_valid_prometheus_and_counts_requests() -> None:
    app = create_app()
    client = TestClient(app)

    # Make a real request that hits a known route template.
    health = client.get("/health")
    assert health.status_code == 200

    scrape = client.get("/metrics")
    assert scrape.status_code == 200
    assert scrape.headers["content-type"].startswith("text/plain")
    body = scrape.text

    # Valid exposition: HELP/TYPE headers for each metric family present.
    assert "# HELP http_requests_total" in body
    assert "# TYPE http_requests_total counter" in body
    assert "# TYPE http_request_duration_seconds histogram" in body
    assert "# TYPE http_requests_in_flight gauge" in body
    # Trailing newline is required by the exposition format.
    assert body.endswith("\n")

    # The counter incremented for the /health request, labeled by the ROUTE
    # TEMPLATE (here a literal path) and status CLASS — not a raw path param.
    assert 'http_requests_total{method="GET",route="/health",status="2xx"}' in body

    # The histogram observed the request (count >= 1 for that route template).
    count_line = (
        'http_request_duration_seconds_count{method="GET",route="/health"}'
    )
    assert count_line in body
    observed = metrics_mod.get_registry().http_request_duration_seconds.count(
        ("GET", "/health")
    )
    assert observed >= 1.0

    # Direct registry read corroborates the scrape (counter incremented).
    assert (
        metrics_mod.get_registry().http_requests_total.value(
            ("GET", "/health", "2xx")
        )
        >= 1.0
    )


def test_metrics_uses_route_template_not_raw_path_param() -> None:
    app = create_app()
    client = TestClient(app)

    # Hit a parameterized route with a unique id; the label must be the TEMPLATE.
    client.get("/v1/runs/some-unique-run-id-123")

    body = client.get("/metrics").text
    # The high-cardinality raw id must NOT appear as a label.
    assert "some-unique-run-id-123" not in body
    # A templated route label (contains a {param}) should be present for /v1/runs.
    assert any(
        'route="/v1/runs/{' in line
        for line in body.splitlines()
        if line.startswith("http_requests_total{")
    )
