# sage2 / eval-granite42

Scores one checkpoint on the Sage2 **Granite 4.2** suite. Each benchmark is its own
target and step (`space://steps/sage2-<benchmark>`), so a failure or a rerun stays
local to that benchmark. Each target writes a `results.json` (`sage2_results`).

The runtime is [sage2-evals](https://github.com/laminair/sage2-evals). Run
`sage2-evals list --suite granite42` there to see which benchmarks are implemented.

| Target | Metric | Status |
|---|---|---|
| `swebench-verified` | pass@1[avg-of-3] resolve rate | implemented |
| `tau3-bench` | pass@1 (avg of 3): mean pass^1 of airline, retail, telecom | implemented |
| `tau3-airline` / `tau3-retail` / `tau3-telecom` / `tau3-banking-knowledge` | pass@1 (pass^1, 4 trials) | implemented |
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

The tau3 targets call a PAID user simulator (and, for retail, an NL-assertion judge):
`aws/claude-sonnet-5` on the IBM LiteLLM gateway by default, which needs
`SAGE2_USER_API_KEY` / `SAGE2_JUDGE_API_KEY` in the job env; calls are metered against
`SAGE2_SPEND_LEDGER` / `SAGE2_SPEND_BUDGET_USD`. `--param TAU_OPTIONS="user_model=self
judge_model=self"` uses the served model instead (no key, not comparable with published
numbers). `tau3-bench` runs the three core domains itself, so running it together with
`tau3-airline`/`-retail`/`-telecom` pays for those simulations twice.
