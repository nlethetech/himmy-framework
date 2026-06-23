#!/usr/bin/env bash
# Run the FULL paired same-model lift matrix for ONE model: generate + evaluate
# BOTH arms (RAW prompting baseline + HIMMY) on the IDENTICAL bounded subset of
# categories, then print the lift table (Wilson CI + exact McNemar via himmy stats).
#
# Usage:
#   scripts/bfcl/run_matrix.sh <raw_model> <himmy_model> [cat1 cat2 ...]
# e.g.
#   scripts/bfcl/run_matrix.sh or-raw-llama3.2-3b-prompting or-himmy-llama3.2-3b
#
# The bounded subset (which task ids) is fixed in
#   workdir/test_case_ids_to_generate.json
# and is shared by BOTH arms so the comparison is paired by task id. Both arms run
# temperature 0, --num-threads 1 (deterministic; the himmy arm also relies on
# single-threaded generation for its per-task schema stash).
#
# Re-runs are incremental: BFCL skips a task that already has a result row, so to
# force a fresh run for an arm delete its workdir/result/<model>/ tree first.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$REPO_ROOT/scripts/bfcl/run.sh"

RAW="${1:?raw model registry name required}"
HIMMY="${2:?himmy model registry name required}"
shift 2
if [ "$#" -gt 0 ]; then
  CATS=("$@")
else
  CATS=(simple_python multiple parallel parallel_multiple irrelevance multi_turn_base)
fi

echo "=== MATRIX: RAW=$RAW  HIMMY=$HIMMY ==="
echo "categories: ${CATS[*]}"
echo

for MODEL in "$RAW" "$HIMMY"; do
  for CAT in "${CATS[@]}"; do
    echo ">>> generate $MODEL / $CAT"
    bash "$RUN" generate --model "$MODEL" --test-category "$CAT" \
      --run-ids --temperature 0 --num-threads 1 2>&1 \
      | grep -E 'Generating|Error during' | tail -1 || true
    echo ">>> evaluate $MODEL / $CAT"
    bash "$RUN" evaluate --model "$MODEL" --test-category "$CAT" \
      --partial-eval 2>&1 | grep -iE 'Accuracy:' | head -1 || true
  done
done

echo
echo "=== LIFT ANALYSIS ==="
/Users/samriddhagc/.bfcl-venv/bin/python "$REPO_ROOT/scripts/bfcl/analyze_lift.py" \
  --raw "$RAW" --himmy "$HIMMY" --categories "${CATS[@]}" \
  --json "$REPO_ROOT/scripts/bfcl/workdir/lift_${HIMMY}.json"
