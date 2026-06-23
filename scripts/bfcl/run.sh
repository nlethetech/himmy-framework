#!/usr/bin/env bash
# Reproducible driver for the BFCL OpenRouter harness.
#
# Usage:
#   scripts/bfcl/run.sh models                 # list models (proves registration)
#   scripts/bfcl/run.sh generate --model or-raw-gpt-4o-mini --test-category simple_python --run-ids --temperature 0 --num-threads 1
#   scripts/bfcl/run.sh evaluate --model or-raw-gpt-4o-mini --test-category simple_python --partial-eval
#   scripts/bfcl/run.sh scores
#
# NOTE: when running a bounded subset (--run-ids), `evaluate` MUST be passed
#   --partial-eval, otherwise BFCL aborts with "Length of model result (N) does
#   not match length of test entries (400)".
#
# It:
#   * pins the bfcl venv python,
#   * puts scripts/bfcl/ on PYTHONPATH so sitecustomize.py auto-registers models,
#   * sets BFCL_PROJECT_ROOT to scripts/bfcl/workdir so result/score/.env live in-repo,
#   * ensures the workdir .env carries OPENROUTER_API_KEY (copied from repo .env;
#     value never printed).
#
# Everything after the first arg is passed straight to the `bfcl` CLI.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BFCL_DIR="$REPO_ROOT/scripts/bfcl"
WORKDIR="$BFCL_DIR/workdir"
BFCL_VENV="/Users/samriddhagc/.bfcl-venv"
BFCL_BIN="$BFCL_VENV/bin/bfcl"

mkdir -p "$WORKDIR"

# Make sure the workdir .env has the OpenRouter key (BFCL loads DOTENV_PATH).
# We copy only OPENROUTER_API_KEY; never echo its value.
if [ -f "$REPO_ROOT/.env" ]; then
  if grep -q '^OPENROUTER_API_KEY=' "$REPO_ROOT/.env"; then
    # Rewrite workdir .env with just the key line, preserving the value verbatim.
    grep '^OPENROUTER_API_KEY=' "$REPO_ROOT/.env" > "$WORKDIR/.env"
  fi
fi

export PYTHONPATH="$BFCL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export BFCL_PROJECT_ROOT="$WORKDIR"

exec "$BFCL_BIN" "$@"
