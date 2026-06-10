# Developer entry points — the same checks CI runs, runnable locally (WS5).
.PHONY: help install lint format types test test-load gate security sbom audit integration bench-gate bench-rebaseline \
        docker-build compose-up compose-up-ollama compose-down ops-health ops-backup helm-lint airgap-bundle

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
	@echo "--- deployment (deploy/, docs/enterprise/deployment.md) ---"
	@echo "make docker-build      - build the Studio image (himmy-studio:dev) from Dockerfile"
	@echo "make compose-up        - bring up the full-stack compose (studio + postgres)"
	@echo "make compose-up-ollama - same, plus a bundled Ollama (pulls models, several GB)"
	@echo "make compose-down      - tear the full-stack compose down"
	@echo "make ops-health        - probe a running deployment (scripts/ops_health.py)"
	@echo "make ops-backup        - snapshot durable state to a tar.gz (scripts/ops_backup.py)"
	@echo "make helm-lint         - lint + render the himmy-studio Helm chart"
	@echo "make airgap-bundle     - build the offline install bundle (scripts/airgap_bundle.py)"

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

# --- deployment ------------------------------------------------------------
# The full-stack compose, ops scripts, Helm chart, and air-gap bundle. See the
# runbook at docs/enterprise/deployment.md for the end-to-end story.

COMPOSE := docker compose -f deploy/compose/docker-compose.yml

# Build the deployable Studio image (frontend + runtime, non-root). Tagged :dev
# for local use; the compose stack tags its own image off ${HIMMY_VERSION}.
docker-build:
	docker build -t himmy-studio:dev .

# Bring up studio + postgres (run from the repo root so the relative build/secret
# paths resolve). Copy deploy/compose/.env.example -> .env and set the password +
# secrets file first (see the file header).
compose-up:
	$(COMPOSE) up -d

# Same, plus a bundled Ollama that pulls the default models on first start.
compose-up-ollama:
	$(COMPOSE) --profile ollama up -d

compose-down:
	$(COMPOSE) down

# Cheap, offline health probe for a running deployment (HTTP /health, disk, SQLite
# integrity, Postgres, Ollama). Exit 0 ok / 1 warn / 2 fail.
ops-health:
	python3 scripts/ops_health.py

# Snapshot the durable .himmy SQLite stores (WAL-safe via the sqlite3 backup API)
# + secrets + optional pg_dump into a checksummed himmy-backup-*.tar.gz.
ops-backup:
	python3 scripts/ops_backup.py backup

# Lint + render the Helm chart. `helm` must be on PATH; when it is not, the same
# checks run dockerized, e.g.:
#   docker run --rm -v "$(PWD)/deploy/helm:/charts" alpine/helm:3.16.2 \
#     lint /charts/himmy-studio
#   docker run --rm -v "$(PWD)/deploy/helm:/charts" alpine/helm:3.16.2 \
#     template /charts/himmy-studio
helm-lint:
	helm lint deploy/helm/himmy-studio
	helm template deploy/helm/himmy-studio >/dev/null

# Assemble the offline, no-network install bundle (images + wheelhouse + Ollama
# models). Pass DRY_RUN=1 (or run the script directly with --dry-run) to print the
# plan + size estimate without downloading anything.
airgap-bundle:
	python3 scripts/airgap_bundle.py build $(if $(DRY_RUN),--dry-run)
