# Granite 4.0 350M — Kubernetes (native) SFT + full eval

Port of [`../../lsf/sft-eval-full-dataset`](../../lsf/sft-eval-full-dataset)
to a **native Granite.build Kubernetes environment** (`type: K8s`, helm
launchers) — **not** the SkyPilot-on-k8s backend.

It runs OpenInstruct SFT on Granite‑4 350M, then fans out to the same
27‑target eval suite (26 SAGE + 1 BFCL), each bound to the trained
checkpoint.

## What changed from the LSF recipe (environment-specific only)

| Concern | LSF recipe | This k8s recipe |
| --- | --- | --- |
| Environment | `space://environments/skypilot/lsf/ibm-bluevela` | `space://environments/{{ space.variables.DEFAULT_ENVIRONMENT }}` (a `type: K8s` env) |
| SFT step | `space://steps/openinstruct-sft` (SkyPilot) | `space://steps/open-instruct` (native k8s) |
| Eval step | `space://steps/sage-eval` (SkyPilot) | `space://steps/custom_code` running the SAGE repo |
| BFCL step | `space://steps/bfcl-eval` | `space://steps/custom_code` (no native BFCL step yet — see below) |
| Compute | `launcher_config.resources` (`accelerators`/`cluster`/`zone`/`memory`) | `compute_config` (`num_gpus_per_node`/`num_nodes`) + `k8s.image` |
| Storage | `env://` paths on the `/proj` GPFS share | `lh://` / `hf:///` artifacts wired through output→input bindings |

The SAGE **framework and benchmarks are unchanged**: `custom_code` clones
`github.ibm.com/ai-models-architectures/sage` and runs the same
`sage/cluster/gb/scripts/*_slim.sh` gb_scripts the LSF `sage-eval` step ran.
The eval structure was modeled on the `BYOSSage` / `SageLeaderboard_v2` k8s
templates, and the SFT target on the `OpenInstruct` template
(`/home/cma/de/cma/assets/templates`).

## Prerequisites / configuration parameters you must supply

### 1. A space that provides the native-k8s assets

`gb build start ... --space <your-space>` where the space:

- Sets **`DEFAULT_ENVIRONMENT`** to a `type: K8s` environment (namespace,
  PVC volumes, COS/LH/HF artifact stores, image pull secrets). The
  standalone `configurations/spaces/local` space in this repo points
  `DEFAULT_ENVIRONMENT` at `skypilot/kubernetes` and does **not** register
  the native-k8s steps — it will not resolve this build.
- Registers the steps `open-instruct` and `custom_code`.
- Defines the Lakehouse output variables referenced by the build:
  `DEFAULT_LH_ENVIRONMENT`, `DEFAULT_LH_NAMESPACE`, `DEFAULT_LH_MODEL_TABLE`,
  `DEFAULT_LH_FILESET_TABLE`.

This is the Granite.build k8s deployment space (e.g. the `granite-build`
namespace / `gbspace-public`), not the local standalone space.

### 2. Recipe parameters (`parameters.yaml`)

Override on the CLI with `--param KEY=VALUE`. The ones you will most likely
need to change:

- `MODEL_URI` — base model as an `lh://` or `hf:///` artifact.
- `TUNING_DATA_URI` — SFT dataset table/fileset (tokenized on the fly by
  `open-instruct`; see Divergences).
- `SAGE_CONFIG_URI` — fileset holding `sage_wrapper.sh` + `env`
  (HF/GIT tokens) for your namespace.
- `SAGE_*_IMAGE` — the per-category SAGE container images (carry the sage
  repo + language toolchains the `multiple-*` code benchmarks need).
- `OE_EVAL_BCB_API_URL` — reachable BigCodeBench evaluation service.
- `SFT_NUM_GPUS` / `EVAL_NUM_GPUS` — GPUs requested from the cluster.

## Run

```shell
gb build start \
  -f recipes/granite4-350m/k8s/sft-eval-full-dataset/build.yaml \
  --parameters-path recipes/granite4-350m/k8s/sft-eval-full-dataset/parameters.yaml \
  --space <your-k8s-space> \
  --param MODEL_URI=lh://.../granite-4.0-350m-base/... \
  --param TUNING_DATA_URI=lh://.../your-sft-table \
  --param SAGE_CONFIG_URI=lh://.../sage_config/...
```

## Divergences to review before relying on results

These are genuine framework differences, not just plumbing — verify them
against your environment:

1. **Trainer / tokenization.** The LSF recipe consumed *pre-tokenized*
   open-instruct data (`--tokenized_data`). The native `open-instruct` step
   tokenizes from raw tables via `dataset_transform_fn` +
   `dataset_mixer_list`. `TUNING_DATA_URI` should therefore be a raw SFT
   table. If you must reuse an already-tokenized dataset, adjust the
   `tuning_config` accordingly (the transform fns assume raw chat data).
2. **SAGE gb_script invocation.** The `custom_code` `start_command` exports
   the same env vars the LSF `sage-eval` step set (`MODEL_PATH`, `EXPERIMENT`,
   `BATCH_SIZE`, `MAX_LENGTH`, `NUM_GPUS`, plus `MULTIPLE_LANG` /
   `OE_EVAL_BCB_API_URL`) and calls the gb_script through `sage_wrapper.sh`.
   Confirm your `sage_wrapper.sh` honors that env-var contract (the shipped
   wrapper does; a customized one may not).
3. **BFCL.** There is no native-k8s BFCL step or template. The `bfcl` target
   is a best-effort `custom_code` invocation; confirm `BFCL_REPO_URL` and the
   entrypoint (`sage/cluster/gb/scripts/bfcl_slim.sh` is a placeholder) exist
   in the repo you point at, or drop the target.
4. **Per-category images.** `multiple-*` code benchmarks need Rust/Java/Node
   toolchains; keep them on `SAGE_CODE_IMAGE`. Safety and multilingual use
   their own images, matching the LSF `image_id` per benchmark.

Validate before running: `gb build validate -f .../build.yaml --parameters-path .../parameters.yaml --space <your-k8s-space>`.
