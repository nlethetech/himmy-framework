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
        "HIMMY_METRICS_TOKEN",
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


# ------------------------------------------- red-team r2: opt-in /metrics scrape token
def test_metrics_unauthenticated_by_default_byte_unchanged() -> None:
    """INVARIANT: with no HIMMY_METRICS_TOKEN, /metrics is open exactly as before."""
    app = create_app()
    client = TestClient(app)
    assert client.get("/metrics").status_code == 200


def test_metrics_token_required_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With HIMMY_METRICS_TOKEN set, an unauthenticated scrape is 401."""
    monkeypatch.setenv("HIMMY_METRICS_TOKEN", "scrape-secret")
    app = create_app()
    client = TestClient(app)
    assert client.get("/metrics").status_code == 401
    # A wrong token is also 401.
    assert (
        client.get("/metrics", headers={"x-metrics-token": "nope"}).status_code == 401
    )


def test_metrics_token_accepts_matching_header_and_bearer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The configured token is accepted via X-Metrics-Token OR Authorization: Bearer."""
    monkeypatch.setenv("HIMMY_METRICS_TOKEN", "scrape-secret")
    app = create_app()
    client = TestClient(app)
    via_header = client.get("/metrics", headers={"x-metrics-token": "scrape-secret"})
    assert via_header.status_code == 200
    assert "# HELP http_requests_total" in via_header.text
    via_bearer = client.get(
        "/metrics", headers={"authorization": "Bearer scrape-secret"}
    )
    assert via_bearer.status_code == 200
