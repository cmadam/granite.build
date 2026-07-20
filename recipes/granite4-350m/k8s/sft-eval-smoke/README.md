# Granite 4.0 350M — Kubernetes SFT per-epoch checkpoint-eval (smoke)

A fast, end-to-end **smoke test** of the native-k8s per-epoch checkpoint-eval
pipeline. It pre-tokenizes a *tiny* fraction of the dataset, fine-tunes Granite‑4
350M for **1 epoch**, evaluates that epoch's checkpoint with the **multilingual
(5) + BFCL (1)** evals, and rolls the results up per-epoch + into a combined CSV.

Its purpose is to exercise the whole `tokenize → train → per-epoch eval → export`
path quickly. It is **not** a real training run — for the full run over all
epochs, use [`../sft-eval-full-dataset`](../sft-eval-full-dataset). The generator
and eval catalog live in
[`../sft-checkpoint-eval`](../sft-checkpoint-eval/README.md).

## build.yaml is generated

`build.yaml` is **generated** from `parameters.yaml` by
`../sft-checkpoint-eval/generate_build.py` — do not hand-edit it. Regenerate:

```shell
cd recipes/granite4-350m/k8s/sft-checkpoint-eval
python generate_build.py \
  --parameters-path ../sft-eval-smoke/parameters.yaml \
  --catalog-path eval-catalog.yaml \
  --output ../sft-eval-smoke/build.yaml \
  --params-out ../sft-eval-smoke/parameters-resolved.yaml
```

This writes `build.yaml` **and** `parameters-resolved.yaml` (base params + any
`--param`/flag overrides). `parameters-resolved.yaml` is git-ignored and
regenerated per run.

## Run

Start with the **resolved** params (they carry any generation-time overrides):

```shell
gb build start \
  -f recipes/granite4-350m/k8s/sft-eval-smoke/build.yaml \
  --parameters-path recipes/granite4-350m/k8s/sft-eval-smoke/parameters-resolved.yaml \
  --space <your-k8s-space>
```

## What it runs (10 targets)

- `tokenize` — CPU byoi, `run_data_prep_v2.sh` (Granite template), `DATA_FRACTION`
  tiny.
- `sft-training` — full fine-tune, `NUM_EPOCHS=1`, emits one HF checkpoint output
  `epoch_1`.
- 6 per-epoch evals bound to `sft-training.epoch_1`: `global-mmlu`, `mgsm`,
  `include-ar-de-es-fr`, `include-hi-bn-ta-te`, `include-it-ja-ko-nl-pt-zh`
  (image `SAGE_MULTILINGUAL_IMAGE`) + `bfcl` (image `BFCL_IMAGE`, `run-bfcl.sh`).
- `export-ep1` — per-epoch exporter, gated on all 6 evals.
- `export-combined` — gated on `export-ep1`, emits the combined benchmark×epoch
  CSV (a single epoch column here).

`total eval targets = #epochs × #evals` → **1 × 6 = 6** here. Widen `EVAL_SETS`
or `NUM_EPOCHS` and this multiplies (see the
[generator README](../sft-checkpoint-eval/README.md)).

## Selecting evals / epochs

- `EVAL_SETS` (`[multilingual-eval, bfcl]`) — named sets and/or individual eval
  names from `../sft-checkpoint-eval/eval-catalog.yaml` (`full-eval` = all 27).
- `EVAL_EPOCHS` (`all`) — `all`, a CSV `"2,4"`, or a YAML list `[2, 4]`
  (1-indexed, `1..NUM_EPOCHS`).

Override at generation time, e.g. `--eval-sets full-eval` or
`--param 'EVAL_EPOCHS=[1]'`.

## Aggregation

- `export-ep<M>` runs `exporter.py` over the single epoch folder
  `/gb-read-write/sage/$${EXPERIMENT}-ep_<M>` → a per-epoch CSV.
- `export-combined` runs one multi-`-in-folder` `exporter.py` over all epoch
  folders → a **benchmark × epoch** table (rows = benchmarks, one column per
  epoch), so each row reads as a metric's trajectory across the run.

## open-instruct step change + caveat

Per-epoch fanout depends on the `open-instruct` step's `log_monitor` emitting a
distinct `binding_id: epoch_<M>` (1-indexed; `epoch_hf_0 → epoch_1`) per epoch's
HF checkpoint, instead of the fixed `tuned_checkpoint`. This must be pushed to
`cmadam/assets@gbspace-config-dev` (the `sft-training` `step_uri` resolves that
branch). **Once it lands, single-checkpoint recipes that bind
`sft-training.tuned_checkpoint` break** — that id no longer exists.

## Success criteria

1. `tokenize` and `sft-training` complete; the checkpoint pushes to HF **without**
   a `Permission denied` error (the training image writes `model.safetensors`
   `0644`).
2. The 5 multilingual evals and bfcl run against the pushed `epoch_1` checkpoint.
3. `export-ep1` and `export-combined` produce the per-epoch + combined CSVs.

See the [full recipe's README](../sft-eval-full-dataset/README.md) for
prerequisites (space setup, secrets, Lakehouse variables) — they are identical.
