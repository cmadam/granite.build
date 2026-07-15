# Granite 4.0 350M — Kubernetes (native) SFT + full eval

Native Granite.build Kubernetes recipe (`type: K8s`, helm launchers). It runs a
**full fine-tune** of Granite‑4 350M with OpenInstruct, pushes the checkpoint to
HuggingFace, then fans out to the 27‑target eval suite (26 SAGE + 1 BFCL), each
bound to the trained checkpoint, and rolls the results up to Lakehouse.

For a fast end-to-end smoke test of the same train→push→eval path, see the sibling
[`../sft-eval-smoke`](../sft-eval-smoke) recipe.

## Pipeline

| Stage | Step | Notes |
| --- | --- | --- |
| SFT training | `space://steps/open-instruct` | Full fine-tune (no LoRA / no merge). HF base model + HF SFT dataset in, checkpoint pushed to HF. Pinned image `SFT_IMAGE`. |
| Evals (26 SAGE) | `space://steps/byoi` | The SAGE repo is baked into per-category images; each target runs `sage/cluster/gb/scripts/<benchmark>.sh` directly and tees a log artifact. |
| BFCL | `space://steps/byoi` | Runs `run-bfcl.sh` in the bfcl image; emits its own result artifact. |
| Export | `space://steps/byoi` | `export-results` binds every eval with `wait_for_push`, so it runs last; rolls the shared results dir into a CSV via `sage/exporters/exporter.py` and uploads to `lh://`. |

Ordering is expressed through input bindings: each eval declares an `eval_log`
output, and `export-results` binds all of them with `wait_for_push: true` (the
build schema has no `depends_on`).

## Prerequisites

### A space that provides the native-k8s assets

`gb build start ... --space <your-space>` where the space:

- Sets **`DEFAULT_ENVIRONMENT`** to a `type: K8s` environment (namespace, PVC
  volumes, image pull secrets, and an `HF_TOKEN` secret keyed `HF_TOKEN`).
- Registers the steps `open-instruct` and `byoi`.
- Defines the Lakehouse output variables: `DEFAULT_LH_ENVIRONMENT`,
  `DEFAULT_LH_NAMESPACE`, `DEFAULT_LH_FILESET_TABLE`.

### Recipe parameters (`parameters.yaml`)

Override on the CLI with `--param KEY=VALUE`. Most likely to change:

- `MODEL_URI` / `TUNING_DATA_URI` — HF base model and raw SFT dataset.
- `DATA_FRACTION` — fraction of the dataset to train on (`1.0` = full).
- `SFT_IMAGE` — pinned OpenInstruct training image.
- `SAGE_*_IMAGE` / `BFCL_IMAGE` — per-category eval images (SAGE repo baked in).
- `SFT_NUM_GPUS` / `EVAL_NUM_GPUS` — GPUs requested from the cluster.
- `EXPERIMENT` — namespaces the shared results dir `/gb-read-write/sage/<EXPERIMENT>`.

## Run

```shell
gb build start \
  -f recipes/granite4-350m/k8s/sft-eval-full-dataset/build.yaml \
  --parameters-path recipes/granite4-350m/k8s/sft-eval-full-dataset/parameters.yaml \
  --space <your-k8s-space>
```

Validate first:

```shell
gb build validate \
  -f recipes/granite4-350m/k8s/sft-eval-full-dataset/build.yaml \
  --parameters-path recipes/granite4-350m/k8s/sft-eval-full-dataset/parameters.yaml \
  --space <your-k8s-space>
```

## Notes to review before relying on results

1. **Full fine-tune, not LoRA.** This recipe trains a full model, so there is no
   LoRA adapter and no `merge_model` step. (An earlier LoRA variant added a
   `custom_code` merge step whose unpinned `pip install` pulled a newer safetensors
   build that wrote `model.safetensors` as `0600` — unreadable by the hfpush pod's
   arbitrary UID, causing `[Errno 13] Permission denied`. The pinned OpenInstruct
   training image writes `0644`, so the full-FT path avoids that failure. If a future
   training image regresses, the fix is to make the output group/other-readable —
   e.g. `chmod -R a+rX` on the checkpoint dir before the push.)
2. **Tokenization.** `open-instruct` tokenizes raw chat data on the fly via
   `dataset_transform_fn` (`sft_tulu_tokenize_and_truncate_v1`, `sft_tulu_filter_v1`);
   `TUNING_DATA_URI` must be a raw SFT dataset (tulu-3 format: a `messages` column).
3. **byoi evals.** Each eval runs the gb_script baked into its category image; no
   repo clone. Per-eval overrides (`multiple-*` `MAX_LENGTH=512` + `MULTIPLE_LANG`,
   `ifeval` `BATCH_SIZE=30`, bigcodebench `OE_EVAL_BCB_API_URL`) are set inline,
   transcribed from `../../lsf/rl-checkpoint-eval/eval-catalog.yaml`.
4. **Images pinned to BlueVela.** Training `open-instruct-tuning:0.1.0-conda`; SAGE
   `sage-py311-{olmes,code,safety,multilingual}:0.025`; bfcl `bfcl-py311:0.02`.
   Confirm each SAGE category image contains its `*.sh` scripts under
   `/workspace/sage`, and the bfcl image ships `/workspace/scripts/run-bfcl.sh`.
