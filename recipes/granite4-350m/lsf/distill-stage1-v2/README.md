# distill-stage1-v2 — off-policy GOLD, anchored and instrumented

Off-policy GOLD distillation of the granite-4.0-350m SFT checkpoint towards a
granite-4.1-3b teacher. `lmbda 0.0`, `beta 0.5`, 2 nodes x 8 H100, effective batch 96,
8192 context — and, unlike [`distill-stage1`](../distill-stage1/README.md), a CE anchor
on the objective, entropy logged every step, a guard that stops on collapse, 2,000 steps
instead of an epoch, and every checkpoint kept.

**Run [`distill-probe`](../distill-probe/README.md) and
[`distill-smoke`](../distill-smoke/README.md) first.** Both are minutes; this is hours.

## Why there is a v2

`distill-stage1` ran as build `df8512e0`. It completed cleanly — every step exit 0, no
NaN, no OOM, no NCCL fault — and produced a broken model.

| benchmark | baseline (SFT) | df8512e0 | retained |
|---|---|---|---|
| HumanEval pass@1 | 40.85 | 0.61 | 1% |
| MBPP pass@1 | 48.94 | 5.56 | 11% |
| MultiPL-E Java pass@1 | 21.05 | 0.00 | 0% |
| GSM8K | 26.84 | 2.35 | 9% |
| MGSM average | 17.60 | 0.08 | 0% |
| IFEval average | 52.98 | 32.79 | 62% |
| MMLU (mc) | 34.71 | 27.47 | 79% |
| SALAD-Bench average | 93.47 | 79.78 | 85% |

Worse on all 30+ benchmarks measured, and the pattern *is* the diagnosis: retention
tracks how many tokens the model must emit correctly in a row. Long free-form generation
retained 0–13%, short or structured 47–88%, likelihood-ranked multiple choice 36–99%. The
knowledge is largely still there; the ability to express it is gone.

The mechanism, in one paragraph. GOLD's loss **is** the divergence — no term measures the
student against the corpus tokens. This student was already two epochs of SFT into this
corpus family, so it started 12% of the way from the teacher (JSD 0.0825 against a
`ln 2` = 0.693 ceiling) with ~68% of the realisable gain banked by step 509. For the
remaining 7,640 steps the only descent direction left was to become more certain, so it
took it: student entropy 0.684 → 0.396 nats (effective branching factor 1.49), reverse KL
moved 1.8% while forward KL improved 36%, and the generations became repetition loops. The
train loss moved 2.7% across those steps and reported none of it.

Full post-mortem: `gold-stage1-df8512e0-analysis.md`, and the recommendations it ends with
are what this recipe implements.

## What changed, and what deliberately did not

| | stage1 | v2 | why |
|---|---|---|---|
| `CE_COEF` | — | 0.05 | `loss = JSD + 0.05 * CE(corpus tokens)`; drifting from real text now costs something |
| `LOG_STUDENT_ENTROPY` | — | true | the metric the train loss was blind to |
| `ENTROPY_GUARD_DROP_FRAC` | — | 0.15 | stops gracefully at a 15% entropy drop; would have fired in hour 1 |
| `GOLD_MAX_STEPS` | 0 (epoch) | 2000 | sized to the ~12% of headroom that existed |
| `GOLD_SAVE_STEPS` / `_TOTAL_LIMIT` | 1000 / 3 | 250 / 16 | keep the whole curve, not its last 750 steps |
| `CKPT_LADDER` | — | 500,1000,1500,2000 | export + transfer-eval + generation check per rung |
| `KD_CODE_DIR` | shared checkout | `kd-sandbox-gb`, pinned | the shared tree is dirty; see below |
| `NCCL_DEBUG` | `""` | `INFO` | build `8f02b739` hung with no trace and had to be relaunched |

Held fixed on purpose: the corpus and its sampling, the geometry (effective batch 96), the
LR schedule, `beta 0.5`, `lmbda 0.0`, the model pair, `MAX_LENGTH 8192`. The comparison
against `df8512e0` is only attributable if the objective is the thing that moved.

Also not changed, and it is the one recommendation from the post-mortem not taken:
`completion_boundary: last_message` still leaves 34,650,329 assistant tokens (10.2%)
unsupervised, because every earlier assistant turn in a multi-turn row is masked out.
Changing it re-preps the corpus (~30–45 min) and changes what the loss averages over, so
v2's loss would stop being comparable to `df8512e0`'s. Worth doing; worth doing separately.

## Two arms

Run it twice.

```bash
# anchored — as shipped
gb build start -f recipes/granite4-350m/lsf/distill-stage1-v2/build.yaml --space <space>

# control — same horizon, same ladder, pure divergence
gb build start -f recipes/granite4-350m/lsf/distill-stage1-v2/build.yaml --space <space> \
  --param CE_COEF=0 --param ENTROPY_GUARD_DROP_FRAC=0 \
  --param RUN_NAME=distill-350m-stage1-v2-ctrl
```

The control exists because without it a better result is ambiguous between *the anchor
worked* and *2,000 steps instead of 8,150 worked*, and `df8512e0` cannot settle that: it
kept no checkpoint below step 7750 and evaluated nothing in between.

Two details are load-bearing:

- **`RUN_NAME` must differ.** gbserver refuses an artifact URI another build in the space
  has already registered; the target then reports SUCCESS with an empty output list and
  consumers wait forever — the `b5f030cd` failure mode.
- **The control's guard is OFF.** It is expected to collapse. Stopping it early would cut
  the comparison short, and the two arms are only readable at matched steps.

`CE_COEF=0` with the guard off renders a config with none of the five new keys in it, so
the control arm runs `df8512e0`'s exact objective — that is a property of the renderer,
asserted by `test_off_policy_key_set_is_exact`, not a claim about intent.

## The trainer is pinned, and why that is new

`train-gold` is the one target that does not read the pinned steps checkout: its launcher
execs `$KD_CODE_DIR/gold/gold.py`. `distill-stage1` pointed that at
`/proj/granite-build/g4os/kd-sandbox` — another project's tree, origin
`Takuma-Udagawa/kd-sandbox`, carrying **159 uncommitted files** including `gold/`. So
`df8512e0` recorded `kd_sandbox_commit fc7d66e` and that metadata does not describe the
code that ran, because what ran was working-tree state.

v2 points at `/proj/granite-build/g4os/kd-sandbox-gb`, a clone this project controls, and
pins it with `KD_EXPECT_REF`. A wrong commit or a dirty tree fails the target in seconds,
before the teacher loads. The base is `karve/kd-sandbox` `20b416c`, whose `gold/` tree is
byte-identical to what `df8512e0` executed — every `gold/*.py` that `gold.py` imports plus
`configs/deepspeed/accelerate_deepspeed_zero3.yaml` hash identically; the single exception
is `gold/sft.py`, which `gold.py` never imports.

Building that checkout, and the one commit on top of it, is
[`steps/gold-distill/skypilot/patches/ce_anchor_and_entropy_guard.diff`](../../../../steps/gold-distill/skypilot/patches/ce_anchor_and_entropy_guard.diff).
Without it `CE_COEF` and the guard have nothing to act on: the keys render and the trainer
rejects the config, which is the loud failure and the one to want.

## What to read, in this order

Not the train loss. It was flat for 94% of `df8512e0` while the damage accumulated.

1. **`student_entropy` per step**, in the workload stdout. Holding within ~15% of its
   step-0 value is the run behaving. Falling is the failure, and the guard will say so.
2. **`rkld` per step.** Last time it moved 1.8% while forward KL improved 36% — a student
   narrowing rather than learning the teacher's spread. It should actually move now.
   `ce_anchor_loss` is logged beside them: at `CE_COEF 0.05` its contribution should sit
   near the divergence's own magnitude (~0.07), neither term ignorable.
3. **`gen-smoke`'s table.** One line per rung: the fraction of generated lines inside a
   run of ≥3 identical lines. A collapsed model scores ~1.0 there, healthy code 0.0.
4. **The four `eval-transfer-<N>/metrics.json`** against `eval-transfer-baseline`, which
   is the t=0 column. `entropy` here is the same quantity the trainer logs, on held-out
   data.
5. **Only then `full-eval`**, on whichever rungs steps 3 and 4 say are healthy.

Workload stdout is not in `gb build log` and not in `build_job_log`; SkyPilot syncs it
back locally:

```bash
gb build log <build-id> --runner --all | grep -i "SkyPilot job.*on gb-"
grep -aE "student_entropy|rkld|ce_anchor_loss|entropy-guard" ~/sky_logs/<cluster>/1-<cluster>/run.log
```

## full-eval per surviving rung

Not wired into this recipe: it is ~4 GPU-h per model and needs `bcb-server` up for
`OE_EVAL_BCB_API_URL`. Start that first, then one launch per rung worth the spend:

```bash
gb build start -f recipes/granite4-350m/lsf/full-eval/build.yaml --space <space> \
  --param MODEL_PATH=/proj/granite-build/g4os/distill/distill-350m-stage1-v2/<build-id>/export-<N> \
  --param EXPERIMENT=distill-350m-stage1-v2-ck<N>
```

The tokenizer-control column from
[`distill-probe`](../distill-probe/README.md#the-tokenizer-control-column) is still
required before any of those rows is compared to the recorded after-SFT row: the probe
measured the checkpoint preferring the pinned tokenizer by 10.9% NLL/byte, and this
export pins `tokenizer_class` while the recorded row does not.

## Cost

| target | shape | time |
|---|---|---|
| `sources` | 1 CPU node | ~5 min at 800k rows |
| `align` | 1 CPU node | ~10 min |
| `corpus` | 1 CPU node, single-process | 30–45 min at 800k rows |
| `train-gold` | 2 x 8 H100 | ~3.5 h for 2,000 steps (~4 s/it), less if the guard fires |
| `export-<N>` x4 | 1 CPU node each | minutes, concurrent |
| `eval-transfer-<N>` x5 | 1 H100 each | minutes each |
| `gen-smoke` | 1 H100 | ~2 min for all four rungs |
| `eval-bfcl` | 1 H100, `simple` only | ~10 min; a plumbing check, not a measurement |

~56 GPU-h per arm against `df8512e0`'s ~213, and both arms together still cost less than
the run they replace.

## Next

Stage 2 is on-policy, from whichever rung this selects:
[`distill-onpolicy-v2`](../distill-onpolicy-v2/README.md). It is written but unlaunched —
its student path is this run's output, and `lmbda_schedule` has never run in this
pipeline.
