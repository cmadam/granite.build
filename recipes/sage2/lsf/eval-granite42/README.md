# sage2 / eval-granite42

Scores one checkpoint on the Sage2 **Granite 4.2** suite. Each benchmark is its own
target and step (`space://steps/sage2-<benchmark>`), so a failure or a rerun stays
local to that benchmark. Each target writes a `results.json` (`sage2_results`).

The runtime is [sage2-evals](https://github.com/laminair/sage2-evals). Run
`sage2-evals list --suite granite42` there to see which benchmarks are implemented.

| Target | Metric | Status |
|---|---|---|
| `swebench-verified` | pass@1[avg-of-3] resolve rate | implemented |
| `aime25`, `hmmt-feb25` | pass@1[avg-of-4] symbolic correct | implemented (NeMo-Skills, `SAGE2_IMAGE_NEMOSKILLS`) |
| `gpqa` (Diamond) | pass@1[avg-of-2] symbolic correct | implemented (NeMo-Skills; gated data, needs `HF_TOKEN`) |
| `livecodebench-v6` | pass@1[avg-of-2] accuracy | implemented |
| `scicode` | pass@1[avg-of-2] subtask accuracy | implemented |
| `mmlu-pro` | symbolic correct | implemented (NeMo-Skills) |
| `arena-hard-v2` | win rate | implemented (NeMo-Skills; judge `aws/claude-sonnet-5`, not the official GPT-4.1, no style control; needs `SAGE2_JUDGE_API_KEY`) |
| `ruler-128k` | accuracy | implemented |
| `ruler-64k` | accuracy | implemented |

## Running

```bash
# Smoke: 5 instances, one repeat, upstream dataset.
gb build start -f recipes/sage2/lsf/eval-granite42/build.yaml \
  --param SAGE2_IMAGE_SWEBENCH=<icr ref> \
  --param SAGE2_LIMIT=5 --param SAGE2_REPEATS=1
```

`ruler-*` targets generate their data in the job for `MODEL_PATH`'s tokenizer and
serve at 131072: with thinking on (the default) a 64k sample plus the thinking budget
needs more than 65536. At 128k the sample plus the budget does not fit in 131072, so
`RULER_128K_OPTIONS` defaults to `enable_thinking=false` (NeMo-Skills' RULER exactly).
That default is a placeholder that keeps the target runnable, not a settled experiment
config; the alternative is `sample_length=<131072 - budget>` (shorter samples, thinking
on). With `RULER_128K_OPTIONS=""` the run stops before building data and names both.

Datasets are the public upstream ones, pinned by commit in sage2-evals. Gated
datasets need `HF_TOKEN` as a space secret.
