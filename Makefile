# Developer entry points — the same checks CI runs, runnable locally (WS5).
.PHONY: help install lint format types test gate security sbom audit integration

help:
	@echo "make install     - editable install with dev + enterprise extras"
	@echo "make gate        - lint + format-check + types + tests (the CI quality gate)"
	@echo "make lint        - ruff check (incl. flake8-bandit SAST)"
	@echo "make types       - mypy"
	@echo "make test        - pytest"
	@echo "make security    - pip-audit + sbom (supply-chain checks)"
	@echo "make audit       - pip-audit (known-CVE dependency scan)"
	@echo "make sbom        - generate a CycloneDX SBOM (sbom.cdx.json)"
	@echo "make integration - run the real-provider integration tests (needs Ollama)"

install:
	python -m pip install -e ".[api,postgres,knowledge,connectors,nepal,auth,dev]"

lint:
	ruff check .

format:
	ruff format .

types:
	mypy himmy

test:
	pytest -q

gate: lint
	ruff format --check .
	mypy himmy
	pytest -q

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
