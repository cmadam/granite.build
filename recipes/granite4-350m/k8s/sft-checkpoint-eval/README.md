# sft-checkpoint-eval — per-epoch checkpoint-fanout k8s SFT generator

The shared generator + eval catalog behind the two native-k8s SFT recipes
([`../sft-eval-smoke`](../sft-eval-smoke) and
[`../sft-eval-full-dataset`](../sft-eval-full-dataset)). It emits a `build.yaml`
that fine-tunes Granite‑4 350M, evaluates a **selected eval suite against each
selected epoch's checkpoint** as that epoch completes, and rolls all per-epoch
results into one combined benchmark×epoch CSV.

This mirrors [`../../lsf/rl-checkpoint-eval`](../../lsf/rl-checkpoint-eval) (which
fans out over RL *checkpoints*); here the fanout dimension is **training
epochs**. Because the build engine dispatches each downstream target exactly once
(keyed by its binding id), the fanout can't be open-ended: `generate_build.py`
computes the epoch schedule up front and emits one training output + one eval
target-set per evaluated epoch.

## Files

- `generate_build.py` — the generator. Reads a `parameters.yaml` + `--param`
  overrides + `eval-catalog.yaml`, writes a plain `build.yaml` and a
  `parameters-resolved.yaml`.
- `eval-catalog.yaml` — the 27 evals (transcribed from the k8s
  `sft-eval-full-dataset` byoi targets) and named sets (`code-eval`,
  `general-eval`, `math-eval`, `safety-eval`, `multilingual-eval`, `bfcl`,
  `full-eval`). The generator's source of truth; eval names/categories/sets match
  `../../lsf/rl-checkpoint-eval/eval-catalog.yaml` exactly.
- `templates/` — the large, invariant command literals kept verbatim: the
  base64 chat-template + `run_data_prep_v2.sh` tokenize command, and the
  `run-bfcl.sh` bfcl command. The small parameterized commands (sage eval,
  per-epoch/combined export) are built in Python.
- `parameters.yaml` — the generator's **default** `--parameters-path` (a
  smoke-shaped base). Real runs pass a recipe-specific `--parameters-path`.
- `test_generate_build.py` — unit tests for the generator.

## What the generated build does

1. **`tokenize`** — pre-tokenizes the tuning data once on a CPU byoi pod
   (`run_data_prep_v2.sh`, Granite template). open-instruct's on-the-fly loader
   can't ingest tulu-3's multi-turn `messages`, so `sft-training` consumes the
   pre-tokenized dataset.
2. **`sft-training`** — full fine-tune for `NUM_EPOCHS`, emitting **one HF
   checkpoint output per evaluated epoch** (`epoch_<M>`, 1-indexed).
3. **per-epoch evals** — for each selected epoch, each selected eval runs as a
   byoi target bound to `sft-training.epoch_<M>`. The k8s monitor streams pod
   logs live, so epoch-N evals dispatch **mid-training** as soon as that epoch's
   checkpoint is emitted.
4. **`export-ep<M>`** — one exporter per epoch, gated (`wait_for_push`) on that
   epoch's eval outputs, producing a per-epoch CSV.
5. **`export-combined`** — gated on **all** per-epoch exports; a single
   `exporter.py -in-folder <ep_1> <ep_2> …` join pivots them into a
   **benchmark × epoch** table (`combined.csv`).

## Usage

```shell
python generate_build.py \
  --parameters-path ../sft-eval-smoke/parameters.yaml \
  --catalog-path eval-catalog.yaml \
  --num-epochs 3 --eval-epochs all --eval-sets 'multilingual-eval,bfcl' \
  --output ../sft-eval-smoke/build.yaml \
  --params-out ../sft-eval-smoke/parameters-resolved.yaml
```

The generator writes **two** files and prints the epoch list, eval list, target
count, and the exact start command to stderr:

- `build.yaml` — the generated build (committed for these two recipes).
- `parameters-resolved.yaml` — the base parameters merged with your flags and
  `--param` overrides. **Pass this to `gb build start`** so those overrides are
  honored when the build's `$${...}` placeholders are resolved.
  (`parameters-resolved.yaml` is git-ignored — it is regenerated per run.)

Then start the build:

```shell
gb build start -f ../sft-eval-smoke/build.yaml \
  --parameters-path ../sft-eval-smoke/parameters-resolved.yaml --space <your-space>
```

### Common flags

Frequently-changed knobs each have a flag; the flag and `--param` both override
the parameters file, with `--param` winning on conflict:

| Concept | Parameter | Flag |
|---|---|---|
| Base model to fine-tune | `MODEL_URI` | `--model` |
| Tuning dataset | `TUNING_DATA_URI` | `--tuning-data` |
| Number of training epochs | `NUM_EPOCHS` | `--num-epochs` |
| Which 1-indexed epochs to evaluate | `EVAL_EPOCHS` | `--eval-epochs` |
| Which evaluations to run | `EVAL_SETS` | `--eval-sets` |
| Experiment namespace | `EXPERIMENT` | `--experiment` |
| Fraction/count of tuning data | `DATA_FRACTION` | `--data-fraction` |

Anything else is set via `--param KEY=VALUE` (dot notation supported).

## Selecting evaluations — `EVAL_SETS`

A list of **named sets and/or individual eval names** (see `eval-catalog.yaml`):

- `[full-eval]` — all 27 evaluations.
- `[multilingual-eval, bfcl]` — the 5 multilingual evals + the single BFCL eval.
- `[olmes-gsm8k, math-eval]` — an individual eval plus a whole set (de-duped).

## Selecting epochs — `EVAL_EPOCHS`

`all` (default) evaluates every epoch `1..NUM_EPOCHS`; or pass a CSV `"2,4"` or a
YAML list `[2, 4]` (validated to `1..NUM_EPOCHS`). The generator forces
`KEEP_LAST_N_CHECKPOINTS >= NUM_EPOCHS` so no evaluated epoch dir is pruned
before its eval reads it.

## How many eval targets? — the warning

Each evaluated epoch is fanned out to every selected eval, so

```
total eval targets = #epochs × #evals
```

The generator prints this — watch it before a `full-eval` run over many epochs
(e.g. `full-eval` × 3 epochs = 81 eval pods). Smoke keeps it to 1 × 6 = 6.

## Aggregation

- **per-epoch CSV** (`export-ep<M>`) — `exporter.py -in-folder
  /gb-read-write/sage/$${EXPERIMENT}-ep_<M>` over the single epoch folder.
- **combined CSV** (`export-combined`) — one `exporter.py` call over **all** epoch
  folders (ascending). exporter.py's native multi-`-in-folder` join emits one
  column per folder, so rows are benchmarks (the sage `model`+`metric`, plus a
  `BFCL-…` row per bfcl eval) and columns are `ep_<M>` — each row reads
  left-to-right as a metric's trajectory across the run. It reads whichever of
  sage/bfcl actually ran.

## open-instruct step change + caveat

Per-epoch fanout depends on a change to the `open-instruct` step's `log_monitor`
(`step_uri` resolves `git+ssh://github.ibm.com/cmadam/assets.git@gbspace-config-dev#…/open-instruct`).
It now emits a **distinct `binding_id: epoch_<M>`** (1-indexed:
`epoch_hf_0 → epoch_1`) per epoch's HF checkpoint, instead of the single fixed
`tuned_checkpoint`. The generated training outputs and eval bindings
(`sft-training.epoch_<M>`) key off these ids.

> **Caveat:** this step change must be pushed to
> `cmadam/assets@gbspace-config-dev` for the k8s build to pick it up. Once it
> lands, **single-checkpoint recipes that bind `sft-training.tuned_checkpoint`
> break** — that binding id no longer exists. Update any such recipe to bind a
> specific `epoch_<M>` (or regenerate it through this generator).

**Verify against a real run:** confirm `finetune.py` (image
`open-instruct-tuning:0.1.0-conda`) writes/logs `epoch_hf_<k>` after *each* epoch
during the run (not only at the end). If it only converts at the end, the fanout
is still correct but all evals dispatch post-training.
