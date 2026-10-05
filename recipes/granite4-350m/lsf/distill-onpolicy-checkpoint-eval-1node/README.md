# distill-onpolicy-checkpoint-eval-1node — the c2000 on-policy run, trainer on one node

This is [`distill-onpolicy-checkpoint-eval`](../distill-onpolicy-checkpoint-eval/README.md)
with `train-gold` on **one 8-GPU node instead of two**. The rest of that README applies
here: why c2000, the data sampler, the unmitigated failure modes, how to read
`combined.csv`.

## What differs, and nothing else does

| | parent | this recipe |
|---|---|---|
| `GOLD_NUM_NODES` | 2 | **1** |
| `GOLD_GRADIENT_ACCUMULATION_STEPS` | 3 | **6** |
| effective batch (2 × accum × nodes × 8) | 96 | 96 |
| `VLLM_MAX_LIFETIME_SECONDS` | 86400 (24 h) | **129600 (36 h)**, because the run takes about twice as long |
| `RUN_NAME`, `EXPERIMENT`, build name, `mem://` URIs | — | the parent's, with a `-1node` suffix |

The per-device batch stays 2, so per-GPU memory, which the `[batch, seq, 100352]`
logits tensor limits, is the same. Each optimizer step still covers 96 rows; one node
just takes about twice as long per step. `test_distill_350m_onpolicy_1node_recipes.py`
asserts that only these keys differ from the parent.

## Why it exists

The vLLM server is its own 2-GPU LSF job. It starts in about two minutes and then sits
idle until the trainer gets nodes. BlueVela kills a GPU job left idle for two hours
(exit 130 at 2h01–2h03 on three on-policy builds, 2026-10-02). Meanwhile the parent's
`2 × 8` trainer needs two fully free hosts at the same moment, at a fairshare priority
of 0.001 in `normal`, and it waited 32 min to more than 3.5 h. A single 8-GPU host is
much easier to get.

It does **not** remove the problem: the server still has its own allocation, so a
trainer wait over two hours still kills it.

## Launch

```bash
gb build start -f recipes/granite4-350m/lsf/bcb-server/build.yaml --space <space>

gb build start -f recipes/granite4-350m/lsf/distill-onpolicy-checkpoint-eval-1node/build.yaml \
  --space <space> --param OE_EVAL_BCB_API_URL=http://<host>:7860/evaluate/
```

Shakedown, as in the parent:

```bash
gb build start -f recipes/granite4-350m/lsf/distill-onpolicy-checkpoint-eval-1node/build.yaml \
  --space <space> \
  --param RUN_NAME=distill-350m-onpolicy-c2000-1node-shakedown \
  --param EXPERIMENT=distill-350m-onpolicy-c2000-1node-shakedown \
  --param GOLD_MAX_STEPS=4 --param GOLD_SAVE_STEPS=2 --param CKPT_LADDER=2,4 \
  --param EVAL_CATEGORIES=bfcl --param VLLM_MAX_LIFETIME_SECONDS=7200
```

`RUN_NAME` must differ per launch, as in the parent.

## Cost (estimated)

`train-gold` is `1 × 8 H100` for ~16–25 h. That is the same ~130–200 GPU-h as the
parent, spread over twice the wall-clock. The server holds 2 H100s for longer, ~32–50
GPU-h instead of ~16–24. The eval fan-out is unchanged.
