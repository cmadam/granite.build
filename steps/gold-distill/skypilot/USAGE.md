# `gold-distill` — GOLD knowledge distillation (SkyPilot / LSF)

Distils a small **student** from a larger **teacher** with the `kd-sandbox` GOLD
trainer (generalized JSD), multi-node on BlueVela via the SkyPilot LSF cloud.

## What supplies what

| Piece | Source |
|---|---|
| Pinned deps (torch, transformers, trl, vllm, deepspeed) | the container image, venv at `/stage/.venv` |
| Trainer (`gold/*.py`, DeepSpeed config) | the live `/proj` checkout named by `kd_code_dir` |
| `bsub`, `blaunch`, enroot, per-node rank/master | SkyPilot's LSF provisioner |
| Config rendering, node roles, `accelerate` launch | this step |

The trainer being a live checkout means a trainer fix needs no change here. The
cost is that it is unpinned state, so the step records `kd_sandbox_commit` as step
metadata on every run.

## Minimal build

```yaml
steps:
  - step_uri: space://steps/gold-distill
    config:
      compute_config:
        num_nodes: 2            # accelerators below are PER NODE
        num_gpus_per_node: 8
      launcher_config:
        resources:
          accelerators: "H100:8"
      gold_config:
        model_name_or_path: /proj/.../student_overlays/granite-4.1-3b-base-hub
        teacher_model_name_or_path: /proj/.../teacher_overlays/granite-4.2-30b
        dataset_name: /proj/.../subsampled_0.4_shuffled_nothink.jsonl
        gradient_accumulation_steps: 12
```

## Choosing a student/teacher pair

GOLD compares student and teacher distributions position-by-position, so the two
**must share a tokenizer** — a same-sized vocabulary is not sufficient, the token
IDs must agree.

On BlueVela the `/proj` overlays fall into several tokenizer families, and the
family does **not** track the version number: `granite-4.0-1b-instruct-clean` and
`granite-4.1-3b_r260401a` ship byte-identical tokenizers, while
`granite-4.1-3b-base-hub` and `granite-4.1-3b-base-ct` do not share one. Group by
the hash of `tokenizer.json`, not by name:

```shell
for d in /proj/granite-build/g4os/kd-sandbox/{student,teacher}_overlays/*; do
  echo "$(md5sum "$d/tokenizer.json" | cut -c1-8)  $(basename "$d")"
done | sort
```

Two families additionally matter for the default `response_template`
(`<|im_start|>assistant`): only families whose vocabulary contains `<|im_start|>`
can use it. Others (the granite-4 markup families, using `<think_off>`,
`<documents>`, …) need a `response_template` matching their own chat format —
otherwise the template matches nothing, the completion span is never located, and
the loss is computed over the wrong tokens **without any error**.

As of this writing the reference pair (`granite-4.1-3b-base-hub` +
`granite-4.2-30b`) is the only pair in its family, so there is no smaller
drop-in substitute. Scale down context and data instead.

## Things that will bite

- **Effective batch is `per_device x grad_accum x nodes x gpus_per_node`.** The
  reference run is 192 (1 x 6 x 4 x 8) and the LR schedule was tuned against it,
  so halving the nodes wants `gradient_accumulation_steps: 12` to hold it. Change
  the node count without this and you have changed the optimization, not just the
  throughput.
- **The dataset must be think-filtered** (`*_nothink.jsonl`). An inline
  `<think>...</think>` in an assistant turn breaks completion extraction.
- **Leave `use_liger_fused_jsd: false`.** granite's `logits_scaling=10` overflows
  the fused bf16 JSD kernel and gives NaN loss.
- **Keep `save_total_limit` high.** An early, less-forgotten checkpoint is often
  the best one to evaluate; a small limit deletes it irrecoverably.
- **The image is an SM90 (H100) build** and will not run on A100.
- **Multi-node needs a SkyPilot with LSF multi-node support.** gbserver refuses
  the launch otherwise rather than silently running on one node.

## Outputs

`checkpoint` (type `model`) — `$GB_BUILD_WORKDIR/checkpoints/<run_name>_node<N>`,
registered by rank 0. The node count in that path comes from the allocation, not
from a parameter, so it always describes the run that produced it.

## On-policy

`vllm_num_servers > 0` dedicates the **last** N nodes to serving the student under
vLLM and trains on the rest; the renderer rejects a count that leaves no trainers,
or on-policy on a single node. On-policy continues from a good off-policy
checkpoint — point `model_name_or_path` at that checkpoint, not the base overlay.
