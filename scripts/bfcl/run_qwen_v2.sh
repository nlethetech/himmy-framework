#!/usr/bin/env bash
# Complete the qwen himmy-v2 (hermes_chatml_xml) arm: generate ALL 6 categories
# on the fixed 170-task subset over the bounded NON-STREAMING transport, then
# evaluate each category with --partial-eval. RAW baseline is reused (untouched).
#
# Transport can no longer hang (handlers_himmy._himmy_generate_bounded:
# stream=False + 90s per-request timeout + bounded 3-attempt retry/backoff), so
# this runs to completion or records a REAL failure per task; never a fake pass.
set -uo pipefail

BFCL_DIR="/Users/samriddhagc/LocalProjects/himmy-agent-test/scripts/bfcl"
WORKDIR="$BFCL_DIR/workdir"
BFCL=/Users/samriddhagc/.bfcl-venv/bin/bfcl
MODEL="or-himmy-qwen2.5-7b"
CATS=(simple_python multiple parallel parallel_multiple irrelevance multi_turn_base)

export PYTHONPATH="$BFCL_DIR${PYTHONPATH:+:$PYTHONPATH}"
export BFCL_PROJECT_ROOT="$WORKDIR"
# Bounded transport knobs (defaults already set in handler; pin explicitly).
export HIMMY_REQUEST_TIMEOUT_S="${HIMMY_REQUEST_TIMEOUT_S:-90}"
export HIMMY_MAX_ATTEMPTS="${HIMMY_MAX_ATTEMPTS:-4}"

echo "=== QWEN HIMMY-V2 FULL RUN start $(date +%FT%T) ==="
echo "model=$MODEL timeout=${HIMMY_REQUEST_TIMEOUT_S}s attempts=${HIMMY_MAX_ATTEMPTS}"

for CAT in "${CATS[@]}"; do
  echo ">>> GENERATE $MODEL / $CAT  $(date +%T)"
  "$BFCL" generate --model "$MODEL" --test-category "$CAT" \
    --run-ids --temperature 0 --num-threads 1 2>&1 \
    | grep -E 'Generating results for or-himmy.*100%|Error|giveup|gave up' | tail -2 || true
  echo ">>> EVALUATE $MODEL / $CAT  $(date +%T)"
  "$BFCL" evaluate --model "$MODEL" --test-category "$CAT" \
    --partial-eval 2>&1 | grep -iE 'Accuracy|Error' | head -3 || true
done

echo "=== QWEN HIMMY-V2 FULL RUN done $(date +%FT%T) ==="
