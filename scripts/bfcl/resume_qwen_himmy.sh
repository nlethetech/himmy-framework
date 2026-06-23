#!/usr/bin/env bash
# Resume helper: finish the qwen2.5-7b HIMMY arm after an interrupted matrix run.
# The 5 single-turn HIMMY categories already generated (30 each); only multi_turn_base
# generation + the per-category evaluations + the lift analysis remain. RAW arm is
# already complete. Idempotent: bfcl skips tasks that already have a result row.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$REPO_ROOT/scripts/bfcl/run.sh"
RAW=or-raw-qwen2.5-7b-prompting
HIMMY=or-himmy-qwen2.5-7b
CATS=(simple_python multiple parallel parallel_multiple irrelevance multi_turn_base)

echo "=== RESUME qwen HIMMY arm ($(date +%H:%M:%S)) ==="
# 1) Generate the only missing HIMMY category (multi_turn_base); others are no-ops.
for CAT in "${CATS[@]}"; do
  echo ">>> generate $HIMMY / $CAT"
  bash "$RUN" generate --model "$HIMMY" --test-category "$CAT" \
    --run-ids --temperature 0 --num-threads 1 2>&1 \
    | grep -E 'Generating results for or|Error during' | tail -1 || true
done

# 2) Evaluate BOTH arms for all categories (RAW scores already exist but re-eval is cheap/idempotent).
for MODEL in "$RAW" "$HIMMY"; do
  for CAT in "${CATS[@]}"; do
    echo ">>> evaluate $MODEL / $CAT"
    bash "$RUN" evaluate --model "$MODEL" --test-category "$CAT" \
      --partial-eval 2>&1 | grep -iE 'Accuracy:' | head -1 || true
  done
done

# 3) Lift analysis.
echo "=== LIFT ANALYSIS ==="
/Users/samriddhagc/.bfcl-venv/bin/python "$REPO_ROOT/scripts/bfcl/analyze_lift.py" \
  --raw "$RAW" --himmy "$HIMMY" --categories "${CATS[@]}" \
  --json "$REPO_ROOT/scripts/bfcl/workdir/lift_${HIMMY}.json"
echo "=== RESUME DONE ($(date +%H:%M:%S)) ==="
