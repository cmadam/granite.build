# distill-onpolicy-ck2500-continue

On-policy GOLD for the granite-4.0-350m student and the granite-4.1-3b teacher,
continued for 2,500 steps from the on-policy run's step-2,500 export (ck2500), as three
arms that differ only in `LMBDA`:

| Arm | `LMBDA` | What it does |
|---|---|---|
| control | `0` | Off-policy only. No vLLM server, no teardown. `train-gold` renders exactly as `distill-checkpoint-eval`'s. |
| ramp 0.5 (default) | `0.5` | Linear from `LMBDA_INIT` 0.25 to 0.5, averaging 0.375. |
| ramp 0.75 | `0.75` | Linear from 0.25 to 0.75, averaging 0.5. |

All three share the ck2500 init, the 1.0e-06 learning-rate peak, `GOLD_DATA_SEED` 44, the
pinned corpus and the trainer checkout. The control shows what 2,500 more steps under
this schedule buy on their own. The gap between it and a ramp arm is what on-policy
buys. `parameters.yaml` gives the reason for each changed value; this recipe is
`distill-onpolicy-checkpoint-eval` otherwise, and a test asserts it.

## Before launching

**1. The patched trainer checkout.** At the step's trainer pin, the linear lmbda schedule
is unreachable code, so a ramp arm dies on its first step. The fix is
`steps/distill/distill-gold/skypilot/patches/lmbda_linear_schedule.diff`. Build the
checkout once, from the repo root. Copy the patch over as a file: piped into `ssh` on
stdin, it does not reach `git apply`, which then waits for input indefinitely.

```bash
scp -F $HOME/.lsf/config steps/distill/distill-gold/skypilot/patches/lmbda_linear_schedule.diff \
  bluevela:lmbda_linear_schedule.diff

ssh -F $HOME/.lsf/config bluevela 'set -e
D=/proj/granite-build/g4os/gb-steps-distillation-lmbda-ramp
git clone -q https://github.com/laminair/gb-steps-distillation.git $D
git -C $D checkout -q a5d59bc45524a8d75706e20d44ae1a254f273f23
git -C $D apply ~/lmbda_linear_schedule.diff
GIT_AUTHOR_NAME=granite.build GIT_AUTHOR_EMAIL=granite.build@localhost \
GIT_COMMITTER_NAME=granite.build GIT_COMMITTER_EMAIL=granite.build@localhost \
GIT_AUTHOR_DATE=2026-10-05T00:00:00+0000 GIT_COMMITTER_DATE=2026-10-05T00:00:00+0000 \
  git -C $D -c commit.gpgsign=false commit -qam "gold: restore the linear lmbda schedule"
git -C $D rev-parse HEAD
rm ~/lmbda_linear_schedule.diff'
```

It must print `288bfc6bdab589eca394ca9f7229f1138a6da1c4`, the recipe's `CODE_EXPECT_REF`.
The step refuses a checkout at any other commit, or a dirty one. All three arms read it.

**2. bcb-server.** `OE_EVAL_BCB_API_URL` must name a live server, or the bigcodebench eval
fails on every rung. The default in `parameters.yaml` is a dead host.

## Launch

Each arm needs its own `RUN_NAME` and `EXPERIMENT`. The vLLM server's `mem://` URIs are
suffixed with `RUN_NAME`, and the result CSVs land under `EXPERIMENT`.

```bash
TS=$(date +%Y%m%d_%H%M)
BCB=http://<bcb-host>:7860/evaluate/
R=recipes/granite4-350m/lsf/distill-onpolicy-ck2500-continue/build.yaml
for A in 0 0.5 0.75; do
  N=$(echo $A | tr -d .)
  gb build start -f $R --space <space> \
    --param LMBDA=$A \
    --param RUN_NAME=distill-350m-ck2500-lmbda$N-$TS \
    --param EXPERIMENT=distill-350m-ck2500-lmbda$N-$TS \
    --param OE_EVAL_BCB_API_URL=$BCB
done
```

Each ramp arm holds 2 x 8 H100 for the trainer plus a vLLM server; the control holds only
the trainer allocation.

## What to read

- **The ramp actually ramps:** the trainer logs `train/current_lmbda` every step. It should
  read 0.25 at step 1 and the arm's `LMBDA` at step 2,500.
- **Rungs 500 and 1,000 of every arm,** beside ck2500's own row. IFEval and SALAD first:
  the first run lost both and did not recover them.
- **The 0.75 arm late in the ramp:** the immediate-EOS collapse a generation floor would
  prevent is still unmitigated (`MIN_COMPLETION_LENGTH` 0).
