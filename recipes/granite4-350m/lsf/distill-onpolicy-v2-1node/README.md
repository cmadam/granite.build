# distill-onpolicy-v2-1node — on-policy GOLD from stage 1 v2, trainer on one node

This is [`distill-onpolicy-v2`](../distill-onpolicy-v2/README.md) with `train-gold` on
**one 8-GPU node instead of two**. The rest of that README applies here: choosing the
student, the topology and its four silent failure modes, teardown, what is held fixed.

## What differs, and nothing else does

| | parent | this recipe |
|---|---|---|
| `GOLD_NUM_NODES` | 2 | **1** |
| `GOLD_GRADIENT_ACCUMULATION_STEPS` | 3 | **6** |
| effective batch (2 × accum × nodes × 8) | 96 | 96 |
| `RUN_NAME`, build name, `mem://` URIs | — | the parent's, with a `-1node` suffix |

The per-device batch stays 2, so per-GPU memory is the same. Each optimizer step still
covers 96 rows; one node just takes about twice as long per step. `VLLM_MAX_LIFETIME_SECONDS`
stays at 5 h: 300 steps on one node is budgeted at ~3 h. `test_distill_350m_onpolicy_1node_recipes.py`
asserts that only these keys differ from the parent.

## Why it exists

The vLLM server is its own LSF job. It starts within minutes and then sits idle until the
trainer gets nodes, and BlueVela kills a GPU job left idle for two hours. A `2 × 8`
trainer needs two fully free hosts at once and has waited from 32 min to more than 3.5 h;
one host is much easier to get. A trainer wait over two hours still kills the server.
See [`distill-onpolicy-checkpoint-eval-1node`](../distill-onpolicy-checkpoint-eval-1node/README.md#why-it-exists).

## Launch

```bash
gb build start -f recipes/granite4-350m/lsf/distill-onpolicy-v2-1node/build.yaml --space <space> \
  --param STUDENT_MODEL=/proj/granite-build/g4os/distill/distill-350m-stage1-v2/<build-id>/export-<N>
```
