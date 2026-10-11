# distill-onpolicy-switch-study

Every training arm of the G4-350m on-policy study
(`distillation-reports/20261008-2345-gkd-paper-applied-to-g4-350m-onpolicy.md`, section 5)
as one recipe plus `--param` overrides. The defaults are arm **A2**: off-policy ck2000, a
fresh optimizer, a 1,000-step warm-up to a 1e-6 peak, λ linear 0 → 1, jsd at β 0.5,
`MIN_COMPLETION_LENGTH` 16, 2,500 steps, screening evals on rungs 1,000–2,000 and the full
battery on 2,500. `parameters.yaml` gives the reason for each value.

Four recipes serve the study:

| Recipe | Serves |
|---|---|
| `distill-onpolicy-switch-study` (this one) | every training arm: A0–A6, B0–B2, C1–C2, D1–D3 |
| `distill-safety-frontload-corpus` | Track 0 check 6: the corpus A6 and Track B read |
| `distill-model-screening-eval` | Track 0 checks 2, 3, 4 and 7, and the rest of the battery on an arm's best screening rung |

Two existing recipes are reused: `bcb-server` must be up for any full battery, and
`distill-onpolicy-checkpoint-eval-rerun` re-runs evals that fail inside an arm.

## What the study needed beyond the existing recipes

- **A trainer patch**, `steps/distill/gold/skypilot/patches/onpolicy_study.diff`, on top
  of `lmbda_linear_schedule.diff`. It adds a λ ramp window (B1 ramps over the first half;
  A1 ramps from its resumed step), the generation floor sent with each vLLM request, and
  `stop_at_step` (A1 ends at 4,500 on c2000's own 8,150-step LR schedule). The result is
  commit `00edb0db`.
- **Two new trainer keys in the GOLD step**, both stock `TrainingArguments` fields:
  `adam_beta2` (A5, A6) and `train_sampling_strategy` (`sequential` for the front-loaded
  corpus). The step also accepts `distillm2_like` under a 0 → 1 ramp, which it refused
  before.
- **The eval protocol**: `SCREEN_EVALS` on every rung, the full battery only on
  `FULL_EVAL_RUNGS`.

## Findings from Track 0 that shape the commands

1. **c2000's optimizer state exists.** Build 1e2e969e has two run trees. The exported
   ck2000 is run `b6b92d72`'s checkpoint-2000 (md5 `c96487d3`), not the retry
   `4895f2e2`'s. Both trees also hold checkpoints 3000–5000, and the step resumes from the
   *highest* checkpoint it is given. So A1 seeds from a directory holding checkpoint-2000
   alone (step 0b below).
3. **λ0 step 1,000 was never re-run.** The control's re-run
   (`sage-rerun/ck2500-lmbda0-20261005`) covered rungs 500, 1,500, 2,000 and 2,500. The
   SALAD 92.4 / AttaQ 81.6 reading is the original single measurement.
6. **The pinned corpus has no safety label.** Its rows are `{messages, row_id}`, and its
   granite-4.0 source labels rows only general, tools or rag. The 12,817 `domain: "safety"`
   rows are in the granite-4.1 SFT set, the same rows the 09-24 report mixed in 5×. The
   trainer's default sampler shuffles the whole corpus under its seed, so front-loading
   needs `TRAIN_SAMPLING_STRATEGY=sequential`. On-policy steps take their prompts from the
   step's own batch, so safety rows reach on-policy steps too.

## Commands

All from the repo root. Each arm gets its own `RUN_NAME` and `EXPERIMENT`.

```bash
cd /home/cma/de/cma/gbserver-worktrees/epic-61-onpolicy-followup
L=recipes/granite4-350m/lsf
R=$L/distill-onpolicy-switch-study/build.yaml
P=$L/distill-onpolicy-switch-study/parameters.yaml
TS=$(date +%Y%m%d_%H%M)
BCB=http://<bcb-host>:7860/evaluate/          # from step 0c
CK2000=/proj/granite-build/g4os/distill/distill-350m-ckpt-eval-20260929/1e2e969e-0a26-46f2-be2b-74e351e6ca24/export-2000
CK2500=/proj/granite-build/g4os/distill/distill-350m-onpolicy-c2000_20261003_1119/68b7c4b2-7d39-449d-9f0b-fb448ac2f5ad/export-2500
SFT=/proj/granite-build/g4os/skypilot-test/sft/checkpoints/v0-20260529-20260529_212150-hf/epoch_hf_2
TEACHER=/proj/granite-build/g4os/models/granite-4.1-3b-pinned
L0_CK1000=/proj/granite-build/g4os/distill/distill-350m-ck2500-lmbda0-20261005_0412/dfe3642d-aea5-49f1-8356-0a1cde8521ee/export-1000
FRONT=/proj/granite-build/g4os/distill/corpora/safety-frontload-x5-w1000
SEED=/proj/data-eng/llmb-read-write/resume-src/c2000-b6b92d72
SHORT="--param CKPT_LADDER=250,500,1000,1500,2000,2500 --param GOLD_SAVE_STEPS=250"
A5="--param ADAM_BETA2=0.95 --param WARMUP_RATIO=0.02 $SHORT"
FRONTP="--param CORPUS_DIR=$FRONT --param TRAIN_SAMPLING_STRATEGY=sequential"
arm() { A=$1; shift; N=distill-350m-study-$A-$TS
  gb build start -f $R --space public --param OE_EVAL_BCB_API_URL=$BCB \
    --param RUN_NAME=$N --param EXPERIMENT=$N "$@"; }
```

### 0. One-time setup

**0-. The space must serve this tree's GOLD step.** `gb space list` shows which
`configurations/spaces/local` the `public` space reads. If it is not this checkout's,
sync this branch's `configurations/assets/environments/skypilot/steps/distill/gold/` there,
or repoint the space. A step from before this change has no flag for `adam_beta2`,
`train_sampling_strategy` or the λ-ramp keys, so it drops them silently: A5 would run at
β2 0.999, and A6 and Track B on a shuffled corpus.

**0a. The patched trainer checkout.** It must print `00edb0db1978c385e1469361f8bcb9293b396222`.

```bash
scp -F $HOME/.lsf/config steps/distill/gold/skypilot/patches/{lmbda_linear_schedule,onpolicy_study}.diff bluevela:
ssh -F $HOME/.lsf/config bluevela 'set -e
D=/proj/granite-build/g4os/gb-steps-distillation-onpolicy-study
ID="GIT_AUTHOR_NAME=granite.build GIT_AUTHOR_EMAIL=granite.build@localhost GIT_COMMITTER_NAME=granite.build GIT_COMMITTER_EMAIL=granite.build@localhost"
git clone -q https://github.com/laminair/gb-steps-distillation.git $D
git -C $D checkout -q a5d59bc45524a8d75706e20d44ae1a254f273f23
git -C $D apply ~/lmbda_linear_schedule.diff
env $ID GIT_AUTHOR_DATE=2026-10-05T00:00:00+0000 GIT_COMMITTER_DATE=2026-10-05T00:00:00+0000 \
  git -C $D -c commit.gpgsign=false commit -qam "gold: restore the linear lmbda schedule"
git -C $D apply ~/onpolicy_study.diff
env $ID GIT_AUTHOR_DATE=2026-10-11T00:00:00+0000 GIT_COMMITTER_DATE=2026-10-11T00:00:00+0000 \
  git -C $D -c commit.gpgsign=false commit -qam "gold: lmbda ramp window, generation floor, stop_at_step"
git -C $D rev-parse HEAD
rm ~/lmbda_linear_schedule.diff ~/onpolicy_study.diff'
```

**0b. A1's resume source.** It must print 16, one ZeRO optimizer shard per rank. The
hardlinks also keep the checkpoint alive if the build tree is ever reaped.

```bash
ssh -F $HOME/.lsf/config bluevela 'set -e
S=/proj/data-eng/llmb-read-write/builds/builds/1e2e969e-0a26-46f2-be2b-74e351e6ca24/runs/b6b92d72-6637-4b90-853a-fa52db503961/checkpoints/distill-350m-ckpt-eval-20260929_node2
D=/proj/data-eng/llmb-read-write/resume-src/c2000-b6b92d72
mkdir -p $D && cp -al $S/checkpoint-2000 $D/
find $D/checkpoint-2000/global_step2000 -name "*optim_states.pt" | wc -l'
```

**0c. bcb-server**, then put its host in `BCB`. Confirm that `curl -s -o /dev/null -w '%{http_code}' http://<host>:7860/` answers 307.

```bash
gb build start -f $L/bcb-server/build.yaml --parameters-path $L/bcb-server/parameters.yaml --space public
```

### Track 0

```bash
# 2. Interpolate ck2000 with the λ0.5 arm's step 2,500 at 0.3 / 0.5 / 0.7 (weight on ck2000); screening evals.
gb build start -f $L/distill-model-screening-eval/build.yaml --parameters-path $P --space public \
  --param EXPERIMENT=distill-350m-interp-ck2000-l05s2500-$TS --param INTERP_RATIOS=0.3,0.5,0.7

# 3 + 4. λ0 step 1,000 re-read (id 1000), and the teacher's ceiling (id 1): SALAD and AttaQ only.
gb build start -f $L/distill-model-screening-eval/build.yaml --parameters-path $P --space public \
  --param EXPERIMENT=distill-350m-safety-check-$TS --param INTERP_RATIOS= \
  --param "EVAL_MODELS=1000=$L0_CK1000;1=$TEACHER" --param EVALS=salad-bench,attaq

# 6. The front-loaded safety corpus (CPU only; reads the 41 GB 4.1 SFT file once).
gb build start -f $L/distill-safety-frontload-corpus/build.yaml --space public

# 7. One screening launch on ck2000 alone: its measured cost, and the screening baseline row.
gb build start -f $L/distill-model-screening-eval/build.yaml --parameters-path $P --space public \
  --param EXPERIMENT=distill-350m-screen-ck2000-$TS --param INTERP_RATIOS= --param "EVAL_MODELS=2000=$CK2000"
```

A ratio from check 2 that holds SALAD ≥ 92.3 gets its full battery. Use a fresh
`EXPERIMENT`, because the screening run already registered the screening evals' URIs:

```bash
ALL=$(python3 -c "import yaml;print(','.join([e['name'] for e in yaml.safe_load(open('$P'))['EVAL_CATALOG']]+['bfcl']))")
gb build start -f $L/distill-model-screening-eval/build.yaml --parameters-path $P --space public \
  --param EXPERIMENT=distill-350m-interp-full-$TS --param INTERP_RATIOS=<ratio> --param EVALS=$ALL \
  --param OE_EVAL_BCB_API_URL=$BCB
```

### Track A (first wave: A0, A2, A3, A5, and A1 once 0b is done)

```bash
arm A0 --param LMBDA=0
arm A1 --param RESUME_FROM_CHECKPOINT_DIR=$SEED --param RESUME_EMIT_SEEDED=false \
  --param GOLD_MAX_STEPS=0 --param STOP_AT_STEP=4500 --param GOLD_DATA_SEED=-1 \
  --param GOLD_LEARNING_RATE=2.0e-06 --param WARMUP_RATIO=0.05 \
  --param LMBDA_RAMP_START_STEP=2000 --param LMBDA_RAMP_STEPS=2500 \
  --param CKPT_LADDER=2250,2500,3000,3500,4000,4500 --param GOLD_SAVE_STEPS=250 --param FULL_EVAL_RUNGS=4500
arm A2
arm A3 --param WARMUP_RATIO=0.05 $SHORT
arm A5 $A5
```

Second wave, after the corpus (check 6) has built. A6 takes the switch settings of
whichever of A2 and A5 dips less:

```bash
arm A6 $A5 $FRONTP                                     # if A5 dipped less
arm A6 $FRONTP                                         # if A2 dipped less (rungs start at 1,000)
arm A4 --param LMBDA=0.25 --param LMBDA_SCHEDULE=constant   # drop it if A2 or A5 shows no dip
```

### Track B (after check 6)

```bash
B="--param STUDENT_MODEL=$SFT --param GOLD_DATA_SEED=42 --param WARMUP_RATIO=0.4 $FRONTP"
arm B0 $B --param LMBDA=0 --param GOLD_LEARNING_RATE=2.0e-06
arm B1 $B --param LMBDA_RAMP_STEPS=1250 --param GOLD_LEARNING_RATE=1.0e-06
arm B2 $B --param LMBDA_SCHEDULE=constant --param GOLD_LEARNING_RATE=1.0e-06
```

### Track C (on the winner of A and B; `W` = the winner's overrides, as above)

```bash
arm C1 $W --param LOSS_ARM=distillm2_like
arm C2 $W --param BETA=0.9
```

### Track D (optional: continue ck2500 with the existing arms' settings)

```bash
D="--param STUDENT_MODEL=$CK2500 --param GOLD_DATA_SEED=44 --param WARMUP_RATIO=0.05 --param LMBDA_INIT=0.25 --param MIN_COMPLETION_LENGTH=0 --param CKPT_LADDER=500,1000,1500,2000,2500"
arm D1 $D --param LMBDA=0.5 --param LOSS_ARM=distillm2_like
arm D2 $D --param LMBDA=0.5 --param BETA=0.9
arm D3 $D --param LMBDA=1.0
```

### After an arm: the full battery on its best screening rung

`<N>` is the rung, `<ARM_RUN>` the arm's `RUN_NAME`, and `<BUILD>` its build id. Only the
evals the screening subset left out run. They run against the rung's export under a FRESH
experiment, so their table is `<ARM_RUN>-ck<N>-rest/combined.csv` beside the arm's own.
Re-running them under the arm's experiment would collide with the per-rung CSV URIs the
arm already registered, and the roll-up behind them would then never dispatch.

```bash
REST=$(python3 -c "import yaml;p=yaml.safe_load(open('$P'));s=set(p['SCREEN_EVALS'].split(','));print(','.join(e['name'] for e in p['EVAL_CATALOG'] if e['name'] not in s))")
gb build start -f $L/distill-model-screening-eval/build.yaml --parameters-path $P --space public \
  --param EXPERIMENT=<ARM_RUN>-ck<N>-rest --param INTERP_RATIOS= \
  --param "EVAL_MODELS=<N>=/proj/granite-build/g4os/distill/<ARM_RUN>/<BUILD>/export-<N>" \
  --param EVALS=$REST --param OE_EVAL_BCB_API_URL=$BCB
```

`distill-onpolicy-checkpoint-eval-rerun` stays the tool for evals that FAILED inside an arm
(pass `--parameters-path $P`).

## What to read

- `train/current_lmbda` every step. It should read 0 at step 1 and 1.0 at the ramp's end
  (step 1,250 for B1, 4,500 for A1).
- SALAD, GlobalMMLU and BFCL at each arm's first rungs: 250 and 500 in the short-warm-up
  arms, 1,000 in the others. That is where the first run dipped.
- `t_gen` and the rollout length in the first hour of λ-near-1 arms (B2, late A arms).
