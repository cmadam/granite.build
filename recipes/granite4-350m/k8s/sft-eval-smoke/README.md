# Granite 4.0 350M — Kubernetes SFT smoke test

A fast, end-to-end **smoke test** of the native-k8s SFT pipeline: it fine-tunes
Granite‑4 350M on a *tiny* fraction of the dataset, pushes the checkpoint to
HuggingFace, and runs only the **multilingual (5) + BFCL (1)** evals against it.

Its purpose is to exercise the whole `train → hfpush → eval → export` path quickly —
in particular the hfpush step, which previously failed with
`[Errno 13] Permission denied: .../model.safetensors`. It is **not** a real training
run. For the full run, use [`../sft-eval-full-dataset`](../sft-eval-full-dataset).

## What it runs

- `sft-training` (`open-instruct`, full fine-tune) with `DATA_FRACTION=0.00001` and
  `NUM_EPOCHS=1`; checkpoint pushed to `hf://…ibm-research/…`.
- 5 multilingual byoi evals: `global-mmlu`, `mgsm`, `include-ar-de-es-fr`,
  `include-hi-bn-ta-te`, `include-it-ja-ko-nl-pt-zh` (image `SAGE_MULTILINGUAL_IMAGE`).
- `bfcl` byoi eval (image `BFCL_IMAGE`, runs `run-bfcl.sh`).
- `export-results` — binds all evals with `wait_for_push` and rolls results up to
  `lh://`.

Everything (steps, images, training-target shape) matches `../sft-eval-full-dataset`;
only the dataset fraction, epoch count, GPU count, and eval set differ.

## Run

```shell
gb build start \
  -f recipes/granite4-350m/k8s/sft-eval-smoke/build.yaml \
  --parameters-path recipes/granite4-350m/k8s/sft-eval-smoke/parameters.yaml \
  --space <your-k8s-space>
```

Validate first:

```shell
gb build validate \
  -f recipes/granite4-350m/k8s/sft-eval-smoke/build.yaml \
  --parameters-path recipes/granite4-350m/k8s/sft-eval-smoke/parameters.yaml \
  --space <your-k8s-space>
```

## Success criteria

1. `sft-training` completes and pushes the checkpoint to HF **without** a
   `Permission denied` error (the training image writes `model.safetensors` `0644`).
2. The 5 multilingual evals and bfcl run against the pushed checkpoint.
3. `export-results` uploads the rolled-up results to `lh://`.

See the [full recipe's README](../sft-eval-full-dataset/README.md) for prerequisites
(space setup, secrets, Lakehouse variables) — they are identical.
