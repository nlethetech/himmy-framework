# Developer entry points — the same checks CI runs, runnable locally (WS5).
.PHONY: help install lint format types test test-load gate security sbom audit integration bench-gate bench-rebaseline

help:
	@echo "make install     - editable install with dev + enterprise extras"
	@echo "make gate        - lint + format-check + types + tests (the CI quality gate)"
	@echo "make lint        - ruff check (incl. flake8-bandit SAST)"
	@echo "make types       - mypy"
	@echo "make test        - pytest (fast; excludes the @slow load/profiling tests)"
	@echo "make test-load   - run the offline load/concurrency + profiling harness (tests/load, incl. slow)"
	@echo "make security    - pip-audit + sbom (supply-chain checks)"
	@echo "make audit       - pip-audit (known-CVE dependency scan)"
	@echo "make sbom        - generate a CycloneDX SBOM (sbom.cdx.json)"
	@echo "make integration - run the real-provider integration tests (needs Ollama)"
	@echo "make bench-gate  - run the PR-sized benchmark gate vs benchmarks/baseline.json (needs Ollama)"
	@echo "make bench-rebaseline - re-measure and rewrite benchmarks/baseline.json (review + commit the diff)"

install:
	python -m pip install -e ".[api,postgres,knowledge,connectors,nepal,auth,dev]"

lint:
	ruff check .

format:
	ruff format .

types:
	mypy himmy

test:
	pytest -q -m "not slow"

test-load:
	pytest tests/load -v

gate: lint
	ruff format --check .
	mypy himmy
	pytest -q -m "not slow"

audit:
	python -m pip install --quiet pip-audit
	pip-audit --desc

sbom:
	python -m pip install --quiet cyclonedx-bom
	cyclonedx-py environment -o sbom.cdx.json
	@echo "wrote sbom.cdx.json"

security: audit sbom

integration:
	pytest -q -m integration

# The same gate CI runs on every PR (benchmarks/baseline.json `gate` block).
bench-gate:
	python scripts/bench_gate.py run

# After an INTENTIONAL quality shift: re-measure, rewrite the baseline (floors =
# measured - margin, stamped with the current SHA), then review + commit the diff.
bench-rebaseline:
	python scripts/bench_gate.py run --rebaseline --sha $$(git rev-parse --short=12 HEAD)
