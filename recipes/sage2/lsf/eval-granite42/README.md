# sage2 / eval-granite42

Scores one checkpoint on the Sage2 **Granite 4.2** suite. Each benchmark is its own
target and step (`space://steps/sage2-<benchmark>`), so a failure or a rerun stays
local to that benchmark. Each target writes a `results.json` (`sage2_results`).

The runtime is [sage2-evals](https://github.com/laminair/sage2-evals). Run
`sage2-evals list --suite granite42` there to see which benchmarks are implemented.

| Target | Metric | Status |
|---|---|---|
| `swebench-verified` | pass@1[avg-of-3] resolve rate | implemented |
| `swebench-pro` | pass@1[avg-of-3] resolve rate | implemented |
| `swebench-multilingual` | pass@1[avg-of-3] resolve rate | implemented |
| others | see `suites/granite42.yaml` | pending |

## Running

```bash
# Smoke: 5 instances, one repeat, upstream dataset.
gb build start -f recipes/sage2/lsf/eval-granite42/build.yaml \
  --param SAGE2_IMAGE_SWEBENCH=<icr ref> \
  --param SAGE2_LIMIT=5 --param SAGE2_REPEATS=1
```

Datasets are the public upstream ones, pinned by commit in sage2-evals. Gated
datasets need `HF_TOKEN` as a space secret.
