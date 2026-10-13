# distill-safety-uniform-corpus

The pinned GOLD corpus that every arm of the G4-350m on-policy study trains on
(df8512e0's, 782,388 train rows), with the 12,817 labelled safety rows of the granite-4.1
SFT set mixed through the **whole file**. It is the uniform counterpart of
`distill-safety-frontload-corpus`, which puts the extra safety rows only in the rows the
first 1,000 steps read. Both corpora use the same safety rows, prep and density
(3.8%); they differ only in where those rows sit.

CPU only, BlueVela. The output is a drop-in `CORPUS_DIR` for
`distill-onpolicy-switch-study`.

## What the build does

| Target | Does |
|---|---|
| `safety-source` | filters the 41 GB `general.jsonl` of the 4.1 SFT set to `domain == "safety"`, spelled as `{role, content}` messages |
| `safety-prep` | runs `distill/corpus-prep` on those rows with the base corpus's tokenizer and policies (`max_length` 8192, `length_policy` drop, think and documents kept), so a safety row survives on the terms a base row did |
| `mix` | adds the safety rows to the base rows until they are `SAFETY_FRACTION` of the result, shuffles everything once under `SHUFFLE_SEED`, and writes the corpus |

The safety rows are **added**, not swapped in for base rows: every base row is kept. At
the default 0.038 that means 30,905 safety rows and 813,293 rows in all, with each of the
12,817 safety rows appearing 2 or 3 times (2.41× on average). Whole shuffled passes over the
safety rows are taken, so no row repeats more than once more than any other.

The output directory (`OUTPUT_DIR`, default
`/proj/granite-build/g4os/distill/corpora/safety-uniform-d038`) holds:

- `train.jsonl`: the mixed corpus, in the order a sequential sampler will read it
- `eval.jsonl`: the base corpus's held-out split, copied unchanged, so eval-transfer
  numbers stay comparable with c2000 and every other arm
- `corpus_rows.jsonl`: one line per training row giving `index`, `row_id` and whether
  its `origin` is `base` or `safety`
- `corpus_manifest.json`: the base corpus's manifest with its train split and row
  counts rewritten (the trainer refuses a row count that disagrees with it), plus a
  `uniform_mix` section recording how the file was built
- `safety-source/`, `safety-prep/`: the intermediate safety rows

## Create the corpus

```bash
cd /home/cma/de/cma/gbserver-worktrees/epic-61-onpolicy-followup
L=recipes/granite4-350m/lsf
gb build start -f $L/distill-safety-uniform-corpus/build.yaml --space public
```

The build log's `mix:` line reports the result. Check it against the defaults:

```
mix: 813293 rows; 30905 safety rows (3.80%); 3766 in the first 96000 (3.92%); 12817 available, 2099 already in the base corpus; each used 2.41x
```

The front-loaded corpus puts 3,648 safety rows in its first 96,000. The "first 96,000"
count here (`CHECK_STEPS` × `EFFECTIVE_BATCH`) should land close to that. If it does,
the first 1,000 steps of both corpora see the same safety density and only the rest of
the epoch differs.

### Variants

Override any value in `parameters.yaml` with `--param`. **Rename `OUTPUT_DIR` for each
variant**: the directory is a pinned input that other builds read, so two variants must
never share one.

```bash
# The 09-24 repeat count (each safety row 5x) instead of its density: 64,085 rows, 7.57%.
gb build start -f $L/distill-safety-uniform-corpus/build.yaml --space public \
  --param SAFETY_FRACTION=0.0757 \
  --param OUTPUT_DIR=/proj/granite-build/g4os/distill/corpora/safety-uniform-d0757
```

At any `SAFETY_FRACTION` above 12,817 / 795,205 = 0.0161, some safety rows repeat.

## Train an arm on it

Point `distill-onpolicy-switch-study` at the output directory with `CORPUS_DIR`. The
corpus-pin check reads the new manifest, and the trainer's row-count check passes
against it.

The safety rows are spread by one shuffle of the whole file, so the trainer's default
shuffled sampler would keep the 3.8% density too. Set
`TRAIN_SAMPLING_STRATEGY=sequential` anyway. The run then reads the rows in the order
`corpus_rows.jsonl` records, so you can say exactly which safety rows any step saw, and its
first 1,000 steps match the count the manifest reports. It also makes the arm differ from
its front-loaded twin only in the corpus.

Using the variables and the `arm` helper from
[`distill-onpolicy-switch-study/README.md`](../distill-onpolicy-switch-study/README.md)
(do its step 0 setup first):

```bash
UNIFORM=/proj/granite-build/g4os/distill/corpora/safety-uniform-d038
UNIFORMP="--param CORPUS_DIR=$UNIFORM --param TRAIN_SAMPLING_STRATEGY=sequential"

# A6's uniform twin: same switch settings as A6, uniform corpus in place of the front-load.
arm A6u $A5 $UNIFORMP                    # if A6 took A5's settings
arm A6u $UNIFORMP                        # if A6 took A2's settings

# Track B on the uniform corpus: swap $FRONTP for $UNIFORMP in B.
BU="--param STUDENT_MODEL=$SFT --param GOLD_DATA_SEED=42 --param WARMUP_RATIO=0.4 $UNIFORMP"
arm B0u $BU --param LMBDA=0 --param GOLD_LEARNING_RATE=2.0e-06
```

Arm names only label `RUN_NAME` and `EXPERIMENT`. The `u` suffix keeps them apart from
the front-loaded runs.

`GOLD_MAX_STEPS` fixes the length of the run, not the corpus, so a 2,500-step arm still
reads 240,000 rows (about 9,100 of them safety rows). The extra 30,905 rows lengthen
the epoch but leave the LR schedule unchanged.
