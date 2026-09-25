# Granite 4.0 350M — LSF (BlueVela) recipes

Build recipes for SFT training and evaluation of the Granite 4.0 350M
model on the BlueVela LSF cluster via the SkyPilot LSF backend.

## Recipes

| Recipe              | Purpose                                                     |
| ------------------- | ----------------------------------------------------------- |
| `sft-10k-test`      | Open Instruct SFT training on a 10k sample                  |
| `bfcl-eval`         | BFCLv3 function-calling evaluation                          |
| `bcb-server`        | BigCodeBench evaluation server                              |
| `general-eval`      | General-domain Sage eval suite (5 targets)                  |
| `math-eval`         | Math-domain Sage eval suite (5 targets)                     |
| `code-eval`         | Code-domain Sage eval suite (9 targets)                     |
| `multilingual-eval` | Multilingual Sage eval suite (5 targets)                    |
| `safety-eval`       | Safety Sage eval suite (2 targets)                          |
| `full-eval`         | Combined 27-target suite (26 Sage + 1 BFCL)                 |
| `sft-10k-eval-test` | SFT (2 epochs) chained to the 27-target eval suite via output binding |
| `ifrl-smoke`        | IFRL GRPO smoke test (rm-server + code-server + 2-update trainer) |
| `export-results`    | Copy results from shared FS to the configured output store  |
| `distill-probe`     | Two-question gate: does the 350m load in the distillation image, and is the tokenizer confound real |
| `distill-smoke`     | Off-policy GOLD distillation from granite-4.1-3b, end to end at smoke scale (7 targets) |
| `distill-stage1`    | Off-policy GOLD distillation, the first real run -- SUPERSEDED, see distill-stage1-v2 |
| `distill-stage1-v2` | Off-policy GOLD, anchored and instrumented: 2,000 steps, checkpoint ladder, entropy guard |
| `distill-onpolicy-v2` | On-policy GOLD from stage 1 v2's chosen checkpoint, with a vLLM server allocation (written, not yet run) |

## Distillation

`distill-*` distil the SFT checkpoint towards a `granite-4.1-3b` teacher, rather than
training it further on hard labels. They reuse the GOLD steps built by epic 61 (see
`recipes/granite4-gold/lsf/`) against a new pair, and their baseline is the **existing**
after-SFT eval row — so the student is the SFT checkpoint, and there is no control arm
because the control was already run and already measured.

Run them in this order, each gating the next:

1. [`distill-probe`](distill-probe/README.md) — ~4 GPU-minutes, nothing trained.
2. [`distill-smoke`](distill-smoke/README.md) — the full graph at 64 rows and 2 steps.
3. [`distill-stage1-v2`](distill-stage1-v2/README.md) — off-policy, 2,000 steps, run
   twice: anchored and as a pure-divergence control.
4. [`distill-onpolicy-v2`](distill-onpolicy-v2/README.md) — on-policy from the rung stage
   1 v2 selects. Written, not yet run.
5. stage 2 proper — on-policy on the IFRL and IdentityRL prompt sets rather than the SFT
   mixture, which is a prompt-set change on top of step 4's policy change.

[`distill-stage1`](distill-stage1/README.md) is step 3's first attempt and is kept only as
the record of it. It ran as build `df8512e0`: one full epoch of pure-divergence GOLD, which
completed cleanly and produced a model worse on every one of the 30+ benchmarks measured
(HumanEval 40.85 → 0.61). The objective had no ground-truth term and the student had
already nearly satisfied it, so entropy reduction was the only descent direction left for
7,640 steps. `distill-stage1-v2` is the response and is what to run.


## Defaults are BlueVela-specific

The `parameters.yaml` files in this directory carry default values for
paths, queues, and resources that are specific to the BlueVela cluster
and the granite-build project layout — for example:

```yaml
MODEL_PATH: "/proj/granite-build/g4os/granite-4.0-350m-base/r251014a"
OUTPUT_DIR: "/proj/granite-build/g4os/sft/checkpoints"
QUEUE: "normal"
ACCELERATORS: "H100:1"
```

These are placeholders for a working BlueVela deployment. To run on a
different LSF cluster, override every infrastructure-specific parameter
on the command line:

```shell
gb build start -f recipes/granite4-350m/lsf/sft-10k-test/build.yaml \
  --parameters-path recipes/granite4-350m/lsf/sft-10k-test/parameters.yaml \
  --space <your-space> \
  --param MODEL_PATH=/your/cluster/path/to/model \
  --param TOKENIZED_DATA_PATH=/your/cluster/path/to/data \
  --param OUTPUT_DIR=/your/cluster/path/to/output \
  --param QUEUE=your-lsf-queue \
  --param ACCELERATORS=H100:8
```

`--space` selects which registered space the build runs in (the space
carries the asset/environment/step bindings the recipe resolves
against). Use `--space-config-uri <uri>` instead if you're pointing at
an unregistered space config directly.

`QUEUE` maps to SkyPilot's `zone` field, which the LSF cloud backend
interprets as the LSF queue name (e.g. `normal`, `preemptable`).

## Standalone gbserver

These recipes are typically driven against a local standalone gbserver
launched from the repo root with:

```shell
gbserver standalone --space-dir configurations/spaces/local --port 8080 2>&1 | tee /tmp/gbserver.log
```

`--space-dir configurations/spaces/local` registers the local space
referenced by `--space` in the `gb build start` command above; the tee
keeps a log of the run for debugging.
