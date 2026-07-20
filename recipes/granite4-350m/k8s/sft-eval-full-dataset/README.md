# Granite 4.0 350M — Kubernetes SFT per-epoch checkpoint-eval (full)

Native Granite.build Kubernetes recipe (`type: K8s`, helm launchers). It
pre-tokenizes the tuning data, runs a **full fine-tune** of Granite‑4 350M with
OpenInstruct for `NUM_EPOCHS`, evaluates **each epoch's checkpoint** with the full
27‑target eval suite (26 SAGE + 1 BFCL), and rolls the results up per-epoch + into
one combined benchmark×epoch CSV.

For a fast end-to-end smoke of the same path (1 epoch, multilingual + bfcl only),
see [`../sft-eval-smoke`](../sft-eval-smoke). The generator and eval catalog live
in [`../sft-checkpoint-eval`](../sft-checkpoint-eval/README.md).

## build.yaml is generated

`build.yaml` is **generated** from `parameters.yaml` by
`../sft-checkpoint-eval/generate_build.py` — do not hand-edit it. Regenerate:

```shell
cd recipes/granite4-350m/k8s/sft-checkpoint-eval
python generate_build.py \
  --parameters-path ../sft-eval-full-dataset/parameters.yaml \
  --catalog-path eval-catalog.yaml \
  --output ../sft-eval-full-dataset/build.yaml \
  --params-out ../sft-eval-full-dataset/parameters-resolved.yaml
```

This writes `build.yaml` **and** `parameters-resolved.yaml` (base params + any
`--param`/flag overrides). `parameters-resolved.yaml` is git-ignored and
regenerated per run.

## Run

Start with the **resolved** params (they carry any generation-time overrides):

```shell
gb build start \
  -f recipes/granite4-350m/k8s/sft-eval-full-dataset/build.yaml \
  --parameters-path recipes/granite4-350m/k8s/sft-eval-full-dataset/parameters-resolved.yaml \
  --space <your-k8s-space>
```

## Pipeline

| Stage | Step | Notes |
| --- | --- | --- |
| Tokenize | `space://steps/byoi` | Pre-tokenizes raw chat data on a CPU pod (`run_data_prep_v2.sh`, Granite template). `sft-training` consumes `--tokenized_data`; open-instruct's on-the-fly loader can't ingest tulu-3's multi-turn `messages`. |
| SFT training | `open-instruct` | Full fine-tune (no LoRA / no merge), `NUM_EPOCHS`. Emits **one HF checkpoint output per evaluated epoch** (`epoch_<M>`, 1-indexed). Pinned image `SFT_IMAGE`. |
| Evals (26 SAGE) | `space://steps/byoi` | Per epoch: each eval runs `sage/cluster/gb/scripts/<benchmark>.sh` (baked into its category image) and tees a log; bound to `sft-training.epoch_<M>`. |
| BFCL | `space://steps/byoi` | Per epoch: `run-bfcl.sh` in the bfcl image. |
| Export (per epoch) | `space://steps/byoi` | `export-ep<M>` binds that epoch's evals with `wait_for_push`; rolls its folder into a per-epoch CSV via `sage/exporters/exporter.py`. |
| Export (combined) | `space://steps/byoi` | `export-combined` binds **all** per-epoch exports; one multi-`-in-folder` `exporter.py` call joins them into a benchmark×epoch `combined.csv`. |

Ordering is expressed through input bindings (`wait_for_push`); the build schema
has no `depends_on`. The k8s monitor streams pod logs live, so epoch-N evals
dispatch **mid-training** as soon as that epoch's checkpoint is emitted.

## Selecting evals / epochs

- `EVAL_SETS` (`[full-eval]`) — named sets and/or individual eval names from
  `../sft-checkpoint-eval/eval-catalog.yaml`. `full-eval` = all 27.
- `EVAL_EPOCHS` (`all`) — `all`, a CSV `"2,4"`, or a YAML list `[2, 4]`
  (1-indexed, `1..NUM_EPOCHS`). The generator forces
  `KEEP_LAST_N_CHECKPOINTS >= NUM_EPOCHS` so no evaluated epoch dir is pruned.

**`NUM_EPOCHS` is 2** by default; the full-eval suite fans out over both.

## How many eval targets? — the warning

```
total eval targets = #epochs × #evals
```

With the defaults that is **2 epochs × 27 evals = 54** eval pods (build total 59
targets: tokenize + training + 54 evals + 2 per-epoch exports + 1 combined).
Watch this before raising `NUM_EPOCHS` — `full-eval` × 3 epochs = 81 eval pods.
The generator prints the count.

## Aggregation

- `export-ep<M>` runs `exporter.py` over the single epoch folder
  `/gb-read-write/sage/$${EXPERIMENT}-ep_<M>` → a per-epoch CSV.
- `export-combined` runs one multi-`-in-folder` `exporter.py` over all epoch
  folders → a **benchmark × epoch** table (rows = benchmarks — the sage
  `model`+`metric` plus a `BFCL-…` row per bfcl eval; columns = `ep_<M>`), so each
  row reads left-to-right as a metric's trajectory across the run.

## Prerequisites

### A space that provides the native-k8s assets

`gb build start ... --space <your-space>` where the space:

- Sets **`DEFAULT_ENVIRONMENT`** to a `type: K8s` environment (namespace, PVC
  volumes, image pull secrets, and an `HF_TOKEN` secret keyed `HF_TOKEN`).
- Registers the steps `open-instruct` and `byoi`.
- Defines the Lakehouse output variables: `DEFAULT_LH_ENVIRONMENT`,
  `DEFAULT_LH_NAMESPACE`, `DEFAULT_LH_FILESET_TABLE`.

### Recipe parameters (`parameters.yaml`)

Override at generation time with a flag or `--param KEY=VALUE`. Most likely to
change: `MODEL_URI` / `TUNING_DATA_URI`, `DATA_FRACTION` (`1.0` = full),
`NUM_EPOCHS`, `EVAL_SETS`, `EVAL_EPOCHS`, `SFT_IMAGE`, `SAGE_*_IMAGE` /
`BFCL_IMAGE`, `SFT_NUM_GPUS` / `EVAL_NUM_GPUS`, `EXPERIMENT`.

## open-instruct step change + caveat

Per-epoch fanout depends on the `open-instruct` step's `log_monitor` emitting a
distinct `binding_id: epoch_<M>` (1-indexed; `epoch_hf_0 → epoch_1`) per epoch's
HF checkpoint, instead of the fixed `tuned_checkpoint`. This must be pushed to
`cmadam/assets@gbspace-config-dev` (the `sft-training` `step_uri` resolves that
branch). **Once it lands, single-checkpoint recipes that bind
`sft-training.tuned_checkpoint` break** — that id no longer exists; rebind them to
a specific `epoch_<M>` or regenerate them through the generator.

**Verify against a real run:** confirm `finetune.py`
(`open-instruct-tuning:0.1.0-conda`) writes `epoch_hf_<k>` after *each* epoch
during the run, not only at the end. If it converts only at the end the fanout is
still correct, but all evals dispatch post-training.

## Notes to review before relying on results

1. **Full fine-tune, not LoRA.** No LoRA adapter, no `merge_model` step. (An
   earlier LoRA variant's unpinned merge `pip install` pulled a safetensors build
   that wrote `model.safetensors` `0600`, unreadable by the hfpush pod's arbitrary
   UID → `[Errno 13] Permission denied`. The pinned training image writes `0644`.)
2. **byoi evals.** Each eval runs the gb_script baked into its category image; no
   repo clone. Per-eval overrides (`multiple-*` `MAX_LENGTH=512` + `MULTIPLE_LANG`,
   `ifeval` `BATCH_SIZE=30`, bigcodebench `OE_EVAL_BCB_API_URL`) are set inline
   from `../sft-checkpoint-eval/eval-catalog.yaml`.
3. **Images pinned to BlueVela.** Training `open-instruct-tuning:0.1.0-conda`;
   SAGE `sage-py311-{olmes,code,safety,multilingual}:0.025`; bfcl
   `bfcl-py311:0.02`.
