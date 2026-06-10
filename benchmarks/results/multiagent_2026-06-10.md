# Benchmark: multiagent (4 tasks × 3 trials)

| Model | Accuracy (95% CI) | Tool-call | p50 | p95 | Cost/trial | Errors |
|---|---|---|---|---|---|---|
| ollama:qwen2.5:7b-instruct | 100% (76%–100%) | — | 8.0s | 12.5s | $0.0000 | 0% |
| ollama:qwen2.5:3b-instruct | 92% (65%–99%) | — | 3.7s | 7.8s | $0.0000 | 8% |
| ollama:qwen2.5:0.5b-instruct | 75% (47%–91%) | — | 0.7s | 2.4s | $0.0000 | 0% |

## By category

_Categories with < 5 tasks show per-task pass counts (a percentage off a handful of tasks is noise), not a rate._

| Model | multiagent |
|---|---|
| ollama:qwen2.5:0.5b-instruct | handoff_routing 3/3, delegation_synthesis 3/3, no_handoff_control 0/3, group_chat_selection 3/3 |
| ollama:qwen2.5:3b-instruct | handoff_routing 3/3, delegation_synthesis 3/3, no_handoff_control 3/3, group_chat_selection 2/3 |
| ollama:qwen2.5:7b-instruct | handoff_routing 3/3, delegation_synthesis 3/3, no_handoff_control 3/3, group_chat_selection 3/3 |

## Pairwise comparison (McNemar, exact)

Paired by `(task_id, trial)`. `b` = first model wins, `c` = second wins; only discordant pairs carry signal. Significance is **Holm-corrected across the 3 pair(s)** at family alpha 0.05 (the raw per-comparison alpha would inflate the family-wise false-positive rate).

| A vs B | b / c | n pairs | p-value | Verdict |
|---|---|---|---|---|
| ollama:qwen2.5:0.5b-instruct vs ollama:qwen2.5:3b-instruct | 1 / 3 | 12 | 0.625 | not significant at 0.05 (Holm, 3 pairs) |
| ollama:qwen2.5:0.5b-instruct vs ollama:qwen2.5:7b-instruct | 0 / 3 | 12 | 0.250 | not significant at 0.05 (Holm, 3 pairs) |
| ollama:qwen2.5:3b-instruct vs ollama:qwen2.5:7b-instruct | 0 / 1 | 12 | 1.000 | not significant at 0.05 (Holm, 3 pairs) |
